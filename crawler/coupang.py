#!/usr/bin/env python3
"""
쿠팡 캠페인 6585 (listSize=60) 1–N 페이지 스텔스 크롤러 – 단일 파일 버전
-----------------------------------------------------------------------
$ python3 crawler_single.py --pages 10 --fmt csv --headless 1 --outdir ./output
"""

from __future__ import annotations
import re, sys, time, argparse, logging
from pathlib import Path
from datetime import datetime
from typing import Sequence, List, Dict, Any

import pandas as pd
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
from dataclasses import dataclass, asdict

# ───────────────────────── 로그 설정 ─────────────────────────
log = logging.getLogger("coupang")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
)

# ───────────────────────── 데이터 모델 ─────────────────────────
@dataclass(slots=True)
class Product:
    rank: int
    name: str
    sale_price: int
    base_price: int | None
    discount_rate: int | None
    unit_price: str | None
    rating: float
    rating_count: int
    rocket: bool
    benefit_text: str | None
    product_url: str
    image_url: str

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ───────────────────────── 파서 유틸 ─────────────────────────
def _to_int(txt: str | None) -> int | None:
    if not txt:
        return None
    num = re.sub(r"[^\d]", "", txt)
    return int(num) if num else None


def parse_products(html: str, page_num: int) -> List[Product]:
    soup = BeautifulSoup(html, "lxml")
    items = soup.select("#product-list > li.ProductUnit_productUnit__Qd6sv")
    products: List[Product] = []

    for idx, li in enumerate(items, start=1):
        name = li.select_one(".ProductUnit_productName__gre7e").get_text(strip=True)
        sale_price = _to_int(li.select_one("strong.Price_priceValue__A4KOr").get_text())

        base_el = li.select_one("del.PriceInfo_basePrice__8BQ32")
        base_price = _to_int(base_el.get_text(strip=True)) if base_el else None

        disc_el = li.select_one(".PriceInfo_discountRate__EsQ8I")
        discount_rate = _to_int(disc_el.get_text(strip=True)) if disc_el else None

        unit_price = (
            li.select_one(".UnitPrice_unitPrice__R_ZcA").get_text(strip=True)
            if li.select_one(".UnitPrice_unitPrice__R_ZcA")
            else None
        )

        # ★ 별점 계산
        star_style = li.select_one(".ProductRating_star__RGSlV")["style"]
        pct = float(re.search(r"(\d+(?:\.\d+)?)%", star_style).group(1))
        rating = round(pct / 20, 1)

        rating_count = _to_int(
            li.select_one(".ProductRating_ratingCount__R0Vhz").get_text(strip=True)
        )

        rocket = bool(li.select_one("img[alt='로켓배송']"))
        benefit_text = (
            li.select_one(".BenefitBadge_cash-benefit__SmkrN span").get_text(strip=True)
            if li.select_one(".BenefitBadge_cash-benefit__SmkrN span")
            else None
        )

        anchor = li.select_one("a")
        product_url = f"https://www.coupang.com{anchor['href']}"
        image_url = li.select_one("figure img")["src"]

        products.append(
            Product(
                rank=(page_num - 1) * 60 + idx,
                name=name,
                sale_price=sale_price,
                base_price=base_price,
                discount_rate=discount_rate,
                unit_price=unit_price,
                rating=rating,
                rating_count=rating_count,
                rocket=rocket,
                benefit_text=benefit_text,
                product_url=product_url,
                image_url=image_url,
            )
        )

    return products

# ───────────────────────── 크롤러 클래스 ─────────────────────────
class CoupangCrawler:
    URL_TMPL = (
        "https://www.coupang.com/np/campaigns/6585"
        "?listSize=60&rating=0&isPriceRange=false&sorter=saleCountDesc"
        "&page={page}&fromComponent=N&channel=user"
    )

    def __init__(self, headless: bool = False, max_pages: int = 10, retry: int = 3):
        self.headless, self.max_pages, self.retry = headless, max_pages, retry
        self.driver = self._init_driver()

    def _init_driver(self):
        opts = uc.ChromeOptions()
        if self.headless:
            opts.add_argument("--headless=new")
        opts.add_argument("--disable-blink-features=AutomationControlled")
        opts.add_argument("--no-sandbox")
        opts.add_argument("--disable-dev-shm-usage")
        opts.add_argument("--window-size=1440,900")
        return uc.Chrome(options=opts)

    def close(self):
        if self.driver:
            self.driver.quit()

    # ──────────────── 수집 진입점 ────────────────
    def run(self) -> Sequence[Product]:
        last_page = min(self._detect_last_page(), self.max_pages)
        log.info("▶ 총 페이지: 1~%d", last_page)

        products: list[Product] = []
        for page in range(1, last_page + 1):
            html = self._fetch_page(page)
            items = parse_products(html, page)
            products.extend(items)
            log.info("  Page %-2d → %d 개", page, len(items))
        return products

    # ──────────────── 내부 도우미 ────────────────
    def _fetch_page(self, page: int) -> str:
        url = self.URL_TMPL.format(page=page)
        for attempt in range(1, self.retry + 1):
            try:
                self.driver.get(url)
                WebDriverWait(self.driver, 15).until(
                    EC.presence_of_element_located((By.ID, "product-list"))
                )
                time.sleep(0.7)
                return self.driver.page_source
            except Exception as e:
                if attempt == self.retry:
                    raise
                log.warning("page %d 로드 실패: %s → 재시도 %d", page, e, attempt)
                time.sleep(3)

    def _detect_last_page(self) -> int:
        html = self._fetch_page(1)
        soup = BeautifulSoup(html, "lxml")
        nums = [
            int(a["data-page"])
            for a in soup.select(".Pagination_pagination__eHDDy a[data-page]")
            if a["data-page"].isdigit()
        ]
        return max(nums) if nums else 1


# ───────────────────────── 결과 저장 ─────────────────────────
def save_products(products: Sequence[Product], outdir: str, fmt: str) -> Path:
    Path(outdir).mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([p.as_dict() for p in products])

    fname = Path(outdir) / f"campaign6585_{datetime.now():%Y%m%d_%H%M%S}.{fmt}"
    if fmt == "csv":
        df.to_csv(fname, index=False)
    else:
        df.to_json(fname, orient="records", force_ascii=False, indent=2)
    return fname


# ───────────────────────── CLI ─────────────────────────
def cli(argv: list[str] | None = None):
    p = argparse.ArgumentParser(description="쿠팡 캠페인 6585 스텔스 크롤러 (단일 파일)")
    p.add_argument("--pages", type=int, default=10, help="최대 페이지 수")
    p.add_argument("--headless", type=int, default=1, help="1=헤드리스, 0=GUI")
    p.add_argument("--fmt", choices=["csv", "json"], default="csv", help="저장 형식")
    p.add_argument("--outdir", default="./output", help="저장 폴더")
    args = p.parse_args(argv)

    crawler = CoupangCrawler(bool(args.headless), args.pages)
    try:
        products = crawler.run()
        path = save_products(products, args.outdir, args.fmt)
        log.info("✅ 완료: %s (총 %d 개)", path, len(products))
    finally:
        crawler.close()


if __name__ == "__main__":
    cli(sys.argv[1:])
