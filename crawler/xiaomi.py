# xiaomi_store_crawler.py
# --------------------------------------------------------------------------- #
#   쿠팡 샤오미 브랜드샵 크롤러 – 대분류(6종) + 소분류(예: 샤오미 패드 7)
#   2025‑07  /  Python 3.11  /  undetected‑chromedriver 3.x
# --------------------------------------------------------------------------- #
from __future__ import annotations

import argparse
import logging
import re
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Sequence

import pandas as pd
import undetected_chromedriver as uc
from bs4 import BeautifulSoup, Tag
from selenium.webdriver import ChromeOptions
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

###############################################################################
# 기본 상수 & 로그
###############################################################################
COUPANG_ROOT = "https://www.coupang.com"           # 상품 상세로 연결될 때 사용
STORE_HOME_URL = "http://shop.coupang.com/A01321495?platform=p"  # 샤오미 브랜드샵 홈
HOME_PATH = "http://shop.coupang.com/A01321495/{menu_id}?platform=p&locale=ko_KR"  # 각 메뉴 URL

log = logging.getLogger("xiaomi")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
)

###############################################################################
# 데이터 모델
###############################################################################
@dataclass(slots=True)
class Product:
    category: str            # 대분류 (스마트폰 / 태블릿 PC ...)
    subcategory: str         # 소분류 (샤오미 패드 7, 샤오미 패드 6 ...)
    name: str
    sale_price: int
    base_price: int | None
    discount_rate: int | None
    rating: float | None
    rating_count: int | None
    rocket: bool
    product_url: str
    image_url: str

    def as_dict(self):
        return asdict(self)

###############################################################################
# 파싱 유틸
###############################################################################
_INT = re.compile(r"[^\d]")

def _to_int(txt: str | None) -> int | None:
    """문자열에서 숫자만 추출해 int 변환 (없으면 None)."""
    if not txt:
        return None
    num = _INT.sub("", txt)
    return int(num) if num else None

def _extract_rating(li: Tag) -> tuple[float | None, int | None]:
    """별점 평균·리뷰수 추출."""
    star = li.select_one(".rating-light")
    if not star:
        return None, None

    # 평균 ★: 폭 % 또는 data-rating 값
    rating_avg: float | None = None
    style = star.get("style", "")
    if "width" in style:
        pct = float(style.split(":")[-1].rstrip("%; "))
        rating_avg = round(pct / 20, 1)
    elif star.has_attr("data-rating"):
        rating_avg = float(star["data-rating"])

    rating_cnt = _to_int(li.select_one(".rating-total-count").get_text()) if li.select_one(".rating-total-count") else None
    return rating_avg, rating_cnt

def _parse_one(li: Tag, category: str, subcategory: str) -> Product:
    """<li class="product-wrapper"> 하나를 Product 객체로 변환."""
    name_tag = li.select_one(".name")
    name = name_tag.get_text(" ", strip=True) if name_tag else ""

    price_area = li.select_one(".price-area")
    sale_price = _to_int(price_area.select_one(".price-value").get_text()) if price_area else 0
    base_price = _to_int(price_area.select_one(".base-price").get_text()) if price_area and price_area.select_one(".base-price") else None
    discount_rate = _to_int(price_area.select_one(".discount-percentage").get_text()) if price_area and price_area.select_one(".discount-percentage") else None

    rating_avg, rating_cnt = _extract_rating(li)
    rocket = bool(li.select_one(".badge.rocket img"))

    a_tag = li.select_one("a.product-wrapper")
    href = a_tag.get("href", "") if a_tag else ""
    product_url = href if href.startswith("http") else COUPANG_ROOT + href

    img_tag = li.select_one("img")
    img_src = img_tag.get("src", "") if img_tag else ""
    if img_src.startswith("//"):
        img_src = "https:" + img_src

    return Product(
        category=category,
        subcategory=subcategory,
        name=name,
        sale_price=sale_price,
        base_price=base_price,
        discount_rate=discount_rate,
        rating=rating_avg,
        rating_count=rating_cnt,
        rocket=rocket,
        product_url=product_url,
        image_url=img_src,
    )

###############################################################################
# 크롤러 본체
###############################################################################
class XiaomiStoreCrawler:
    """샤오미 브랜드샵 > 6개 대분류를 순회하며 모든 상품 + 소분류 정보를 수집."""

    TARGET_CATEGORIES: dict[str, str | None] = {
        "스마트폰": "318534",
        "태블릿 PC": "318540",
        "웨어러블": "318550",
        "로봇청소기": "318538",
        "TV 모니터": "318537",
        "생활가전": "318539",
    }

    # --------------------------------------------------------------
    def __init__(self, headless: bool, outdir: str, fmt: str):
        self.headless = headless
        self.outdir = Path(outdir)
        self.fmt = fmt
        self.driver = self._init_driver()

    def _init_driver(self):
        opt = ChromeOptions()
        if self.headless:
            opt.add_argument("--headless=new")
        opt.add_argument("--lang=ko-KR")
        opt.add_argument("--disable-gpu")
        opt.add_argument("--no-sandbox")
        opt.add_argument("--window-size=1440,900")
        return uc.Chrome(options=opt)

    # --------------------------------------------------------------
    # 내부 도우미
    # --------------------------------------------------------------
    def _scroll_to_bottom(self, pause: float = 0.6, max_round: int = 40):
        """스크롤 다운을 반복해 모든 lazy‑load 상품을 노출."""
        last = 0
        for _ in range(max_round):
            items = len(self.driver.find_elements(By.CSS_SELECTOR, "li.product-wrapper"))
            if items == last:
                break
            last = items
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(pause)

    def _fetch_menu_map(self) -> Dict[str, str]:
        """홈에서 data-menu-item-id 추출 (지정값 우선)."""
        # 이미 다 지정돼 있으면 끝
        if all(self.TARGET_CATEGORIES.values()):
            return self.TARGET_CATEGORIES  # type: ignore

        self.driver.get(STORE_HOME_URL)
        WebDriverWait(self.driver, 15).until(EC.presence_of_element_located((By.CSS_SELECTOR, "ul.smooth")))
        soup = BeautifulSoup(self.driver.page_source, "lxml")
        for li in soup.select("ul.smooth > li[data-menu-item-id]"):
            name = li.get_text(strip=True)
            if name in self.TARGET_CATEGORIES and not self.TARGET_CATEGORIES[name]:
                self.TARGET_CATEGORIES[name] = li["data-menu-item-id"]
        return self.TARGET_CATEGORIES  # type: ignore

    # --------------------------------------------------------------
    def _parse_category(self, cat: str, menu_id: str) -> List[Product]:
        """대분류 페이지 1장 로드 → 무한스크롤 → 상품 파싱 (소분류 포함)."""
        url = HOME_PATH.format(menu_id=menu_id)
        self.driver.get(url)
        WebDriverWait(self.driver, 15).until(EC.presence_of_element_located((By.CSS_SELECTOR, "div.modules")))

        self._scroll_to_bottom()
        soup = BeautifulSoup(self.driver.page_source, "lxml")
        products: List[Product] = []

        # part = Products / Carousel 모두 상품 모듈
        for part in soup.select('div[data-part-name="Products"], div[data-part-name="Carousel"]'):
            # 소분류명 추출
            title_tag = part.select_one(".category-title > span") or part.select_one(".carousel-title > span")
            sub = title_tag.get_text(strip=True) if title_tag else "(미지정)"

            for li in part.select("li.product-wrapper"):
                try:
                    products.append(_parse_one(li, cat, sub))
                except Exception as err:
                    log.debug("파싱 오류 – %s/%s: %s", cat, sub, err)

        log.info("  ▸ %s – %d개", cat, len(products))
        return products

    # --------------------------------------------------------------
    def run(self):
        all_items: List[Product] = []
        for cat, mid in self._fetch_menu_map().items():
            if not mid:
                log.warning("❌ 메뉴 ID 없음: %s", cat)
                continue
            try:
                all_items.extend(self._parse_category(cat, mid))
            except Exception as e:
                log.error("카테고리 %s 실패: %s", cat, e)

        self._save(all_items)

    # --------------------------------------------------------------
    def _save(self, items: Sequence[Product]):
        if not items:
            log.warning("수집된 데이터가 없습니다.")
            return
        self.outdir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame([p.as_dict() for p in items])
        ts = time.strftime("%Y%m%d_%H%M%S")
        fp = self.outdir / f"xiaomi_store_{ts}.{self.fmt}"
        if self.fmt == "csv":
            df.to_csv(fp, index=False)
        else:
            df.to_json(fp, orient="records", force_ascii=False, indent=2)
        log.info("✅ 저장 완료 → %s (%d rows)", fp, len(df))

    # --------------------------------------------------------------
    def close(self):
        try:
            self.driver.quit()
        except Exception:
            pass

###############################################################################
# CLI 진입점
###############################################################################

def cli(argv: list[str] | None = None):
    p = argparse.ArgumentParser(description="샤오미 브랜드샵 전체 크롤러 (대/소분류)懶로드 지원")
    p.add_argument("--headless", type=int, default=1, help="1=headless, 0=GUI")
    p.add_argument("--fmt", choices=["csv", "json"], default="csv")
    p.add_argument("--outdir", default="./output")
    args = p.parse_args(argv)

    crawler = XiaomiStoreCrawler(bool(args.headless), args.outdir, args.fmt)
    try:
        crawler.run()
    finally:
        crawler.close()


if __name__ == "__main__":
    cli(sys.argv[1:])
