"""
ssg_crawler.py  –  SSG.COM 리스트 크롤러 (빈 할인율 대응)
──────────────────────────────────────────────────────
• 출력 열(G마켓 호환 16개)
pip install -U selenium undetected-chromedriver selenium-stealth \
               fake-useragent beautifulsoup4 pandas openpyxl pytz
"""
from __future__ import annotations

# ── SSL 우회 ────────────────────────────────
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import argparse, re, time, random, datetime, json, sys, pytz
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
from bs4 import BeautifulSoup

# ── Selenium & 스텔스 ───────────────────────
import undetected_chromedriver as uc
from selenium_stealth import stealth
from fake_useragent import UserAgent
from selenium.webdriver.common.by import By

KST = pytz.timezone("Asia/Seoul")

# ─────────── 헬퍼 ───────────────────────────
def txt(el): return el.get_text(strip=True) if el else None

def num_only(s: str | None):
    """숫자만 추출해 int 변환; 없으면 None"""
    if not s:
        return None
    digits = re.sub(r"[^\d]", "", s)
    return int(digits) if digits else None

def human_wait(base=1.0, jitter=0.7):
    time.sleep(base + random.random() * jitter)

# ────────── 드라이버 ────────────────────────
def get_driver(headless=False):
    ua = UserAgent().chrome
    opts = uc.ChromeOptions()
    if headless: opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox"); opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--lang=ko-KR"); opts.add_argument(f"user-agent={ua}")
    driver = uc.Chrome(options=opts, use_subprocess=True)
    stealth(driver, languages=["ko-KR","ko"], vendor="Google Inc.", platform="Win32",
            webgl_vendor="Intel Inc.", renderer="Intel Iris OpenGL", fix_hairline=True)
    driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument",
        {"source":"Object.defineProperty(navigator,'webdriver',{get:()=>undefined})"})
    return driver

def safe_get(driver, url):
    cur = driver.current_window_handle
    driver.get(url); human_wait(1.3,1)
    if driver.current_window_handle!=cur and driver.window_handles:
        driver.switch_to.window(driver.window_handles[-1])

# ────────── 리스트 파서 ─────────────────────
def parse_ssg_list(html:str, top_n:int) -> List[Dict[str,Any]]:
    soup = BeautifulSoup(html, "html.parser")
    out=[]
    li_tags = soup.select('li.cunit_t232[data-unittype="item"]')[:top_n]
    for idx,li in enumerate(li_tags,1):
        unit     = li.select_one(".ssgitem_unit")
        mkt_info = json.loads(unit["data-react-mkt-info"])
        link     = li.select_one(".ssgitem_thmb_link")["href"]

        brand     = txt(li.select_one(".ssgitem_tit em")) or ""
        raw_title = txt(li.select_one(".ssgitem_tit")) or ""
        title     = raw_title.replace(brand,"").lstrip("/ ").strip()

        rating_t  = txt(li.select_one(".ssgitem_review_score")) or ""
        m         = re.search(r"(\d+(?:\.\d+)?)", rating_t)
        rating_v  = float(m.group(1)) if m else None

        promo_txt = ";".join(b.text.strip() for b in li.select(".ssgitem_bottom_label"))
        rank_val  = num_only(li.get("data-adidx")) or idx

        out.append({
            "순위": rank_val,
            "상품명": title,
            "상품ID": mkt_info["item_id"],
            "정가": num_only(txt(li.select_one(".ssgitem_price.ty_old em"))),
            "최종가": num_only(txt(li.select_one(".ssgitem_price.ty_new em"))) \
                      or int(mkt_info["lwst_sellprc"]),
            "할인율": num_only(txt(li.select_one(".ssgitem_sale_rate span"))),
            "가격내역": "",
            "판매자명": brand,
            "카테고리코드": "",
            "카테고리라벨": "",
            "노출코드": "",
            "평점": rating_v,
            "리뷰수": num_only(txt(li.select_one(".ssgitem_review_num"))),
            "프로모션태그": promo_txt,
            "수집시각": datetime.datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S"),
            "상품URL": link,
        })
    return out

# ────────── 크롤링 메인 ───────────────────
def crawl_ssg(list_url:str, top_n:int, headless:bool, out_dir:str) -> Path:
    drv = get_driver(headless)
    try:
        safe_get(drv, list_url)
        # 스크롤 6회 (필요시 늘림)
        for _ in range(6):
            drv.execute_script("window.scrollBy(0, document.body.scrollHeight/3);")
            human_wait(0.8,0.6)
            if len(drv.find_elements(By.CSS_SELECTOR,
                   'li.cunit_t232[data-unittype="item"]')) >= top_n:
                break
        html = drv.page_source
        data = parse_ssg_list(html, top_n)
    finally:
        drv.quit()

    df = pd.DataFrame(data)
    COLS=["순위","상품명","상품ID","정가","최종가","할인율","가격내역",
          "판매자명","카테고리코드","카테고리라벨","노출코드",
          "평점","리뷰수","프로모션태그","수집시각","상품URL"]
    df = df[COLS]
    ts = datetime.datetime.now(KST).strftime("%Y%m%d_%H%M")
    path = Path(out_dir)/f"ssg_top{top_n}_{ts}.xlsx"
    df.to_excel(path, index=False)
    print(f"[DONE] {len(df)}개 저장 → {path.resolve()}")
    return path

# ────────── CLI ────────────────────────
if __name__=="__main__":
    default_url="https://www.ssg.com/disp/category.ssg?dispCtgId=6000188618&pageSize=100"
    ap=argparse.ArgumentParser("SSG 리스트 크롤러")
    ap.add_argument("--list-url", default=default_url, help="리스트 URL")
    ap.add_argument("--top-n", type=int, default=100, help="수집 개수 (default 100)")
    ap.add_argument("--headless", action="store_true", help="헤드리스 모드")
    ap.add_argument("--out-dir", default=".", help="엑셀 저장 폴더")
    args = ap.parse_args()

    try:
        crawl_ssg(args.list_url, args.top_n, args.headless, args.out_dir)
    except KeyboardInterrupt:
        sys.exit("\n[EXIT] 사용자 중단")
