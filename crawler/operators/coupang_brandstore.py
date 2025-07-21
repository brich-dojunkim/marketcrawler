#!/usr/bin/env python3
"""Coupang Brand‑store crawler

크롤링 대상  : https://shop.coupang.com/***  브랜드스토어 상품 listing
주요 기능
-----------
* 원하는 정렬 옵션(랭킹·낮은가격·판매량 …)을 클릭해 결과를 재로딩
* 가상 스크롤(recycling list) 환경에 맞춰 **창 높이만큼 조금씩 내려가며** 상품 로드
* "최근 10 라운드 동안 새 상품 URL이 추가되지 않으면" 종료

CLI 예시
~~~~~~~~~
```
python3 coupang_brandstore.py \
  --url   "https://shop.coupang.com/A01321495/314039?locale=ko_KR&platform=p" \
  --sort  "판매량순" \
  --pause 1.0 \
  --headless 0 \
  --outdir ./output --fmt csv
```
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import logging
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Set

import bs4
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
import undetected_chromedriver as uc

# ────────────────────────────── 설정 ──────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)5s | %(message)s",
    datefmt="%H:%M:%S",
)

# ────────────────────────────── 데이터 모델 ──────────────────────────────
@dataclass
class Product:
    name: str
    price: int | None = None              # 원화, 정수 (할인가)
    rating: float | None = None           # 0.0 – 5.0
    review_count: int | None = None
    discount_percentage: int | None = None  # 55 -> 55 %
    image_url: str | None = None
    product_url: str | None = None
    rank: int | None = None               # 스크롤 순서 기준 1 … N

# ────────────────────────────── 유틸 ──────────────────────────────
_RE_ONLY_DIGIT = re.compile(r"[^0-9]")

def _strip_digits(text: str) -> int | None:
    """문자열에서 숫자만 추출 → int, 없으면 None"""
    if not text:
        return None
    m = _RE_ONLY_DIGIT.sub("", text)
    return int(m) if m else None

def _get_total_count(driver: webdriver.Chrome) -> int:
    """상단 "전체 (<num>)" 부분에서 총 갯수 추출"""
    try:
        elem = driver.find_element(By.CSS_SELECTOR, ".total-count .number")
        return _strip_digits(elem.text or "") or 0
    except Exception:
        return 0

def _click_sort_option(driver: webdriver.Chrome, label: str) -> None:
    """정렬 옵션(텍스트 완전 일치)을 찾아 클릭. 없으면 RuntimeError"""
    try:
        ul = driver.find_element(By.CSS_SELECTOR, "ul.pc-sortkey-filter")
    except Exception as e:
        raise RuntimeError("정렬 UL을 찾지 못했습니다.") from e

    for li in ul.find_elements(By.CSS_SELECTOR, "li.sortkey"):
        if (li.text or "").strip() == label.strip():
            li.click()
            return
    raise RuntimeError(f"정렬 옵션 '{label}'을 찾지 못했습니다.")

# ───────────────────────  파싱 (BeautifulSoup)  ──────────────────────

def _parse_visible_products(html: str) -> List[Product]:
    soup = bs4.BeautifulSoup(html, "html.parser")
    out: List[Product] = []
    for li in soup.select("ul.products-list li.product-wrap"):
        a = li.select_one("a.product-wrapper")
        if not a:
            continue
        name            = a.select_one("div.name")
        price_val       = a.select_one("strong.price-value")
        rating_div      = a.select_one("div.rating-light")
        rating_total    = a.select_one("span.rating-total-count")
        disc_span       = a.select_one("span.discount-percentage")
        img             = a.select_one("img")

        out.append(Product(
            name=(name.text or "").strip() if name else "",
            price=_strip_digits(price_val.text) if price_val else None,
            rating=float(rating_div["data-rating"]) if rating_div and rating_div.has_attr("data-rating") else None,
            review_count=_strip_digits(rating_total.text) if rating_total else None,
            discount_percentage=_strip_digits(disc_span.text) if disc_span else None,
            image_url=img["data-src"] if img and img.has_attr("data-src") else None,
            product_url=a["href"] if a.has_attr("href") else None,
        ))
    return out

# ────────────────────────────── 크롤러 ──────────────────────────────
class CoupangCrawler:
    """브랜드스토어 상품 Listing 전용 크롤러"""

    def __init__(
        self,
        url: str,
        sort_label: str = "판매량순",
        headless: bool  = True,
        pause: float    = 1.0,
    ) -> None:
        self.url        = url
        self.sort_label = sort_label
        self.pause      = pause

        opts = uc.ChromeOptions()
        if headless:
            opts.add_argument("--headless=new")
        opts.add_argument("--disable-gpu")
        opts.add_argument("--no-sandbox")
        opts.add_argument("--lang=ko-KR")

        self.driver: webdriver.Chrome = uc.Chrome(options=opts)
        self.driver.set_window_size(1400, 1000)

    # ─────────── 핵심 로직 ───────────
    def fetch_all(self) -> List[Product]:
        drv = self.driver
        drv.get(self.url)

        WebDriverWait(drv, 15).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "ul.pc-sortkey-filter"))
        )

        _click_sort_option(drv, self.sort_label)
        time.sleep(1.2)                     # 정렬 후 초기 로딩 대기

        total_expected = _get_total_count(drv)
        logging.info("총 %d개 상품이 표시된 페이지 – 스크롤 시작", total_expected)

        seen: Set[str]      = set()
        products: List[Product] = []
        stagnant_rounds     = 0
        max_stagnant        = 10           # 새 URL이 안 늘어나면 정지

        while len(seen) < total_expected and stagnant_rounds < max_stagnant:
            before = len(seen)

            # 1) 현재 viewport에서 파싱
            for p in _parse_visible_products(drv.page_source):
                if p.product_url and p.product_url not in seen:
                    seen.add(p.product_url)
                    products.append(p)

            # 2) 창 높이의 90% 만큼 더 내리기
            drv.execute_script("window.scrollBy(0, Math.floor(window.innerHeight*0.9));")
            time.sleep(self.pause)

            # 3) 변화 감지
            if len(seen) == before:
                stagnant_rounds += 1
            else:
                stagnant_rounds  = 0

        # 최종 랭킹 부여
        for idx, p in enumerate(products, 1):
            p.rank = idx

        logging.info("✔ 수집 결과: %d / %d", len(products), total_expected)
        return products

    def close(self):
        try:
            self.driver.quit()
        except Exception:
            pass

# ────────────────────────────── 저장 함수 ──────────────────────────────

def _save_csv(products: List[Product], outdir: Path) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    outfile = outdir / f"brandstore_{ts}.csv"

    with outfile.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(Product.__annotations__.keys()))
        writer.writeheader()
        for p in products:
            writer.writerow(asdict(p))

    logging.info("저장 완료 → %s", outfile)
    return outfile

# ────────────────────────────── CLI ──────────────────────────────

def _parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Coupang 브랜드스토어 크롤러")
    ap.add_argument("--url",      required=True, help="브랜드 스토어 listing URL")
    ap.add_argument("--sort",     default="판매량순", help="정렬 옵션 라벨 (기본: 판매량순)")
    ap.add_argument("--outdir",   default="./output", help="결과 저장 폴더")
    ap.add_argument("--fmt",      choices=["csv", "json"], default="csv")
    ap.add_argument("--headless", type=int, choices=[0, 1], default=1, help="0=창 표시, 1=헤드리스")
    ap.add_argument("--pause",    type=float, default=1.0, help="스크롤 후 대기 초")
    return ap.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)

    crawler = CoupangCrawler(
        url=args.url,
        sort_label=args.sort,
        headless=bool(args.headless),
        pause=args.pause,
    )
    try:
        products = crawler.fetch_all()
    finally:
        crawler.close()

    outdir = Path(args.outdir)
    if args.fmt == "csv":
        _save_csv(products, outdir)
    else:
        outdir.mkdir(parents=True, exist_ok=True)
        ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        outfile = outdir / f"brandstore_{ts}.json"
        with outfile.open("w", encoding="utf-8-sig") as f:
            json.dump([asdict(p) for p in products], f, ensure_ascii=False, indent=2)
        logging.info("저장 완료 → %s", outfile)


if __name__ == "__main__":
    main()
