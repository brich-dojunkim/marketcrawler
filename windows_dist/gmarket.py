"""
gmarket.py – G마켓 베스트 Top-N 크롤러 (스텔스 우회 + 버그패치)
────────────────────────────────────────────────────────
• 기본 URL  : 여성의류 베스트 100위
• 기본 모드 : 창을 띄우고(headless = False) 실행
• 필드 순서 : 순위, 상품명, 상품ID … 상품URL(마지막)
pip install -U undetected-chromedriver selenium-stealth fake-useragent \
               pandas openpyxl beautifulsoup4
"""
from __future__ import annotations

# ── SSL 검증 해제 (루트 인증서 문제 우회) ─────────────────
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
# ───────────────────────────────────────────────────────

import argparse, re, time, sys, random
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Any

import pandas as pd
from bs4 import BeautifulSoup

# ───────── Selenium & 스텔스 ─────────
import undetected_chromedriver as uc
from selenium_stealth import stealth
from fake_useragent import UserAgent
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


# ───────────────────────── 공통 유틸
def txt(el) -> str | None:
    return el.get_text(strip=True) if el else None


def num_only(s: str | None) -> int | None:
    return int(re.sub(r"[^\d-]", "", s)) if s else None


def human_wait(base=1.0, jitter=0.7):
    time.sleep(base + random.random() * jitter)


# ───────────────────────── 드라이버 생성
def get_driver(headless: bool = True):
    ua = UserAgent().chrome
    options = uc.ChromeOptions()
    if headless:
        options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--lang=ko-KR")
    options.add_argument(f"user-agent={ua}")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--window-size=1280,960")

    driver = uc.Chrome(options=options, use_subprocess=True)   # version_main 자동 감지

    # Selenium-Stealth
    stealth(
        driver,
        languages=["ko-KR", "ko"],
        vendor="Google Inc.",
        platform="Win32",
        webgl_vendor="Intel Inc.",
        renderer="Intel Iris OpenGL",
        fix_hairline=True,
    )
    driver.execute_cdp_cmd(
        "Page.addScriptToEvaluateOnNewDocument",
        {"source": "Object.defineProperty(navigator,'webdriver',{get:()=>undefined})"},
    )
    return driver


# ───────────────────────── 새 탭 보호 get()
def safe_get(driver, url: str):
    origin = driver.current_window_handle
    driver.get(url)
    human_wait(1.5, 1.2)
    if driver.current_window_handle != origin and driver.window_handles:
        driver.switch_to.window(driver.window_handles[-1])


# ───────────────────────── ① 랭킹 페이지
def parse_rank_page(driver, list_url: str, top_n: int = 100) -> List[Dict]:
    print(f"[INFO] 랭킹 페이지 → {list_url}")
    safe_get(driver, list_url)
    driver.maximize_window()

    last = driver.execute_script("return document.body.scrollHeight")
    while True:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        human_wait(0.8, 0.6)
        new = driver.execute_script("return document.body.scrollHeight")
        if new == last:
            break
        last = new
        if len(driver.find_elements(By.CSS_SELECTOR, "a[data-montelena-goodscode]")) >= top_n:
            break

    soup = BeautifulSoup(driver.page_source, "html.parser")
    tiles = soup.select("a[data-montelena-goodscode]")[:top_n]

    rows: List[Dict] = []
    for tile in tiles:
        rows.append(
            {
                "순위": num_only(txt(tile.select_one(".box__label-rank"))) or (len(rows) + 1),
                "상품명": txt(tile.select_one(".box__item-title")),
                "상품ID": tile["data-montelena-goodscode"],
                "정가": num_only(txt(tile.select_one(".box__price-original .text__value"))),
                "최종가": num_only(txt(tile.select_one(".box__price-seller .text__value"))),
                "할인율": num_only(txt(tile.select_one(".box__discount"))),
                "가격내역": "",
                "판매자명": "",
                "카테고리코드": tile["data-montelena-categorycode"],
                "카테고리라벨": "",
                "노출코드": tile["data-montelena-acode"],
                "평점": None,
                "리뷰수": None,
                "프로모션태그": ";".join(img["alt"] for img in tile.select(".box__lmo-tags img")),
                "수집시각": "",
                "상품URL": tile["href"],
            }
        )
    return rows


# ───────────────────────── ② 상세 페이지 (평점 파싱 수정)
def parse_detail(driver, url: str, max_try: int = 3) -> Dict[str, Any]:
    for attempt in range(max_try):
        try:
            safe_get(driver, url)
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "h1.itemtit"))
            )
            soup = BeautifulSoup(driver.page_source, "html.parser")

            # ── 평점 숫자만 추출 ─────────────────────
            rating_txt = txt(soup.select_one(".text__score"))  # 예: "평점4.5"
            rating_val = None
            if rating_txt:
                m = re.search(r"(\d+(?:\.\d+)?)", rating_txt)
                if m:
                    rating_val = float(m.group(1))
            # ───────────────────────────────────────

            items = soup.select(".box__layer-coupon-information li")
            breakdown = [
                f"{txt(li.select_one('.text'))}:{num_only(txt(li.select_one('.num')))*-1}"
                for li in items[1:-1]
                if txt(li.select_one(".text")) and txt(li.select_one(".num"))
            ]
            return {
                "판매자명"   : txt(soup.select_one(".text__seller .link__seller")),
                "카테고리라벨": txt(soup.select_one(".text__category-best")),
                "평점"       : rating_val,
                "리뷰수"     : num_only(txt(soup.select_one(".text__num"))),
                "가격내역"   : ";".join(breakdown),
                "최종가"     : num_only(txt(soup.select_one(".price_innerwrap-coupon .price_real")))\
                               or num_only(txt(soup.select_one(".price_real"))),
            }
        except Exception as e:
            print(f"[RETRY] {attempt+1}/{max_try} {e}")
            human_wait(2, 2)
    print(f"[FAIL] 상세 파싱 실패 : {url}")
    return {}


# ───────────────────────── ③ 메인
def crawl(list_url: str, top_n: int, headless: bool, out_dir: str) -> Path:
    drv = get_driver(headless)
    data: List[Dict] = []
    try:
        rank_rows = parse_rank_page(drv, list_url, top_n)
        for i, row in enumerate(rank_rows, 1):
            print(f"[{i}/{len(rank_rows)}] 상세 → {row['상품URL']}")
            detail = parse_detail(drv, row["상품URL"])
            row.update(detail)
            row["수집시각"] = datetime.now(timezone(timedelta(hours=9))).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            data.append(row)
            human_wait(1.2, 0.8)
    finally:
        drv.quit()

    df = pd.DataFrame(data)
    col_order = [
        "순위", "상품명", "상품ID",
        "정가", "최종가", "할인율", "가격내역",
        "판매자명",
        "카테고리코드", "카테고리라벨",
        "노출코드",
        "평점", "리뷰수",
        "프로모션태그", "수집시각",
        "상품URL",
    ]
    df = df[col_order]

    ts = datetime.now(timezone(timedelta(hours=9))).strftime("%Y%m%d_%H%M")
    out_path = Path(out_dir) / f"gmarket_top{top_n}_{ts}.xlsx"
    df.to_excel(out_path, index=False)
    print(f"[DONE] 엑셀 저장 → {out_path.resolve()}")
    return out_path


# ───────────────────────── CLI
if __name__ == "__main__":
    default_url = "https://www.gmarket.co.kr/n/best?groupCode=100000001&subGroupCode=200000004"

    ap = argparse.ArgumentParser("G마켓 베스트 TopN 크롤러")
    ap.add_argument("--list-url", default=default_url, help="랭킹 페이지 URL")
    ap.add_argument("--top-n", type=int, default=100, help="수집 상위 랭크")
    ap.add_argument("--headless", action="store_true", help="Headless 모드 켜기")
    ap.add_argument("--out-dir", default=".", help="엑셀 저장 폴더")
    args = ap.parse_args()

    try:
        crawl(args.list_url, args.top_n, args.headless, args.out_dir)
    except KeyboardInterrupt:
        sys.exit("\n[EXIT] 사용자 중단")
