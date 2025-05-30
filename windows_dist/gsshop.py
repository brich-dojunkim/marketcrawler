"""
gs_crawler_human_dbg.py – GS SHOP 인피니티스크롤 크롤러
  • 사람처럼 400~800px 랜덤 스크롤
  • 스피너 감지 + 상품 증가 모니터
  • 풍부한 DEBUG 로그 & 스크린샷
"""
from __future__ import annotations
import ssl; ssl._create_default_https_context = ssl._create_unverified_context

import argparse, random, time, re, datetime, sys, pytz, logging, math, uuid
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
from bs4 import BeautifulSoup

import undetected_chromedriver as uc
from selenium_stealth import stealth
from fake_useragent import UserAgent
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException

# ───── 로깅 ──────────────────────────────────
logger = logging.getLogger("gsdbg")
def init_logger(level:str):
    logging.basicConfig(
        format="%(asctime)s [%(levelname)5s] %(message)s",
        datefmt="%H:%M:%S", level=getattr(logging, level.upper(), "INFO"))
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("selenium").setLevel(logging.WARNING)

# ───── 상수 ──────────────────────────────────
KST = pytz.timezone("Asia/Seoul")
COLS = ["순위","상품명","상품ID","정가","최종가","할인율","가격내역",
        "판매자명","카테고리코드","카테고리라벨","노출코드",
        "평점","리뷰수","프로모션태그","수집시각","상품URL"]

# ───── 헬퍼 ──────────────────────────────────
def txt(el): return el.get_text(strip=True) if el else None
def num_only(s:str|None):
    if not s: return None
    d = re.sub(r"[^\d]","",s); return int(d) if d else None
def nap(base=0.25, jitter=0.2): time.sleep(base + random.random()*jitter)

# ───── 드라이버 ──────────────────────────────
def get_driver(headless=False):
    ua = UserAgent().chrome
    opts = uc.ChromeOptions()
    if headless: opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--lang=ko-KR")
    opts.add_argument(f"user-agent={ua}")
    drv = uc.Chrome(options=opts, use_subprocess=True)
    stealth(drv, languages=["ko-KR","ko"], vendor="Google Inc.",
            platform="Win32", webgl_vendor="Intel Inc.",
            renderer="Intel Iris OpenGL", fix_hairline=True)
    drv.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument",
        {"source": "Object.defineProperty(navigator,'webdriver',{get:()=>undefined});"})
    logger.debug("Chrome driver ready (headless=%s)", headless)
    return drv

# ───── 파싱 ──────────────────────────────────
def split_brand_title(raw:str):
    if m := re.match(r"\s*\[([^\]]+)]\s*(.*)", raw):
        return m.group(1), m.group(2).strip()
    parts = raw.split()
    return (parts[0] if parts else ""), raw

def parse(html:str, top:int)->List[Dict[str,Any]]:
    soup  = BeautifulSoup(html,"html.parser")
    nodes = soup.select("section.prd-list li a.prd-item")[:top]
    logger.info("🔍 최종 탐색 상품 %d개", len(nodes))
    out=[]
    for idx, a in enumerate(nodes, 1):
        link  = a["href"]
        prdid = a.get("data-prdid")
        brand, title = split_brand_title(txt(a.select_one("dt.prd-name")) or "")
        oldp = num_only(txt(a.select_one("del.price-upper")))
        newp = num_only(txt(a.select_one(".set-price strong") or a.select_one(".set-price")))
        disc = num_only(txt(a.select_one(".price-discount span")))
        rv   = num_only(txt(a.select_one(".user-comment")) or txt(a.select_one(".selling-count")))
        promo = ";".join(t.get_text(strip=True) for t in a.select(".advantage span,.badge-abs span"))
        out.append({
            "순위": idx, "상품명": title, "상품ID": prdid,
            "정가": oldp, "최종가": newp, "할인율": disc, "가격내역": "",
            "판매자명": brand, "카테고리코드": "", "카테고리라벨": "", "노출코드": "",
            "평점": None, "리뷰수": rv, "프로모션태그": promo,
            "수집시각": datetime.datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S"),
            "상품URL": link,
        })
    return out

# ───── 스피너 대기 ───────────────────────────
def wait_spinner(driver, timeout=12):
    try:
        t0=time.perf_counter()
        WebDriverWait(driver, 2).until(
            EC.visibility_of_element_located((By.ID, "viewLoading")))
        t1=time.perf_counter()
        WebDriverWait(driver, timeout).until(
            EC.invisibility_of_element_located((By.ID, "viewLoading")))
        t2=time.perf_counter()
        logger.debug("SPINNER show@%.2fs hide@%.2fs  Δ=%.2fs",
                     t1-t0, t2-t0, t2-t1)
    except TimeoutException:
        logger.debug("SPINNER none/timeout")

# ───── 스크린샷 ──────────────────────────────
def snap(driver, directory:Path, tag:str):
    fname = directory/f"{datetime.datetime.now(KST).strftime('%H%M%S')}_{tag}_{uuid.uuid4().hex[:4]}.png"
    driver.save_screenshot(str(fname))

# ───── 사람처럼 스크롤 ────────────────────────
def human_scroll(driver, top:int, stall_max:int,
                 screenshot_dir:Path|None=None):
    last_len, stall = 0, 0
    logger.info("🚶‍♂️ 스크롤 시작 → 목표 %d개 (stall_max=%d)", top, stall_max)
    i=0
    while True:
        curr = len(driver.find_elements(By.CSS_SELECTOR,
                                        "section.prd-list li a.prd-item"))
        if curr >= top:
            logger.info("✅ 목표 달성 %d/%d개", curr, top)
            break

        i+=1
        step = random.randint(400, 800)
        driver.execute_script("window.scrollBy(0, arguments[0]);", step)
        y = driver.execute_script("return window.pageYOffset")
        h = driver.execute_script("return document.body.scrollHeight")
        logger.debug("SCROLL #%d +%dpx  y=%d  h=%d", i, step, y, h)

        if screenshot_dir: snap(driver, screenshot_dir, f"{i:02d}")

        nap()
        wait_spinner(driver)
        nap(0.1,0.15)

        new_len = len(driver.find_elements(
                By.CSS_SELECTOR, "section.prd-list li a.prd-item"))
        logger.debug("ITEMS %d→%d (+%d)", curr, new_len, new_len-curr)

        if new_len == last_len:
            stall += 1
            logger.debug("STALL %d/%d", stall, stall_max)
        else:
            stall = 0
        last_len = new_len

        if stall >= stall_max:
            logger.warning("⛔ %d회 연속 증가 無 – 스크롤 종료", stall_max)
            break

# ───── 메인 ──────────────────────────────────
def crawl(url:str, top:int, headless:bool, out_dir:str,
          stall_max:int, snap_dir:str|None):
    drv = get_driver(headless)
    try:
        logger.info("🌐 GET %s", url)
        drv.get(url)
        wait_spinner(drv)

        sdir = None
        if snap_dir:
            sdir = Path(snap_dir).expanduser().resolve()
            sdir.mkdir(parents=True, exist_ok=True)
            logger.debug("📸 스크린샷 저장 경로: %s", sdir)

        human_scroll(drv, top, stall_max, sdir)
        data = parse(drv.page_source, top)
    finally:
        drv.quit()
        logger.debug("드라이버 quit")

    df = pd.DataFrame(data)[COLS]
    ts = datetime.datetime.now(KST).strftime("%Y%m%d_%H%M")
    path = Path(out_dir)/f"gs_top{top}_{ts}.xlsx"
    df.to_excel(path, index=False)
    logger.info("[DONE] %d개 저장 → %s", len(df), path.resolve())

# ───── CLI ──────────────────────────────────
if __name__ == "__main__":
    def_url="https://www.gsshop.com/shop/sect/sectL.gs?sectid=1660575&eh=eyJwYWdlTnVtYmVyIjo3LCJzZWxlY3RlZCI6Im9wdC1wYWdlIiwibHNlY3RZbiI6IlkifQ%3D%3D"
    ap = argparse.ArgumentParser("GS SHOP 크롤러 – deep debug")
    ap.add_argument("--list-url", default=def_url)
    ap.add_argument("--top-n", type=int, default=100)
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--stall-max", type=int, default=5,
                    help="연속 무증가 허용 횟수(기본 5)")
    ap.add_argument("--snap-dir", default=None,
                    help="DEBUG 모드 스크린샷 저장 폴더")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    init_logger(args.log_level)
    try:
        crawl(args.list_url, args.top_n, args.headless, args.out_dir,
              args.stall_max, args.snap_dir)
    except KeyboardInterrupt:
        logger.warning("사용자 강제 종료")
        sys.exit()
