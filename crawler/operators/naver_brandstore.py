#!/usr/bin/env python3
"""네이버 브랜드스토어 크롤러

크롤링 대상: 네이버 브랜드스토어 상품 목록
주요 기능:
- 정렬 옵션 선택 (인기도순, 최신등록순, 낮은가격순 등)
- 페이지네이션 지원
- 상품 정보 수집 (이름, 가격, 평점, 리뷰수 등)

CLI 예시:
python3 naver_brandstore.py \
  --url "https://brand.naver.com/lgcaremall/category/50001937?cp=1" \
  --sort "인기도순" \
  --pages 5 \
  --pause 1.0 \
  --headless 0 \
  --outdir ./output --fmt csv
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
from urllib.parse import urljoin, urlparse, parse_qs

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
    price: int | None = None              # 할인가
    original_price: int | None = None     # 원가
    discount_percentage: int | None = None # 할인율
    rating: float | None = None           # 평점
    review_count: int | None = None       # 리뷰수
    product_url: str | None = None
    image_url: str | None = None
    rank: int | None = None               # 페이지 내 순서
    is_best: bool = False                 # BEST 상품 여부
    delivery_info: str | None = None      # 배송 정보

# ────────────────────────────── 유틸 ──────────────────────────────
_RE_ONLY_DIGIT = re.compile(r"[^0-9]")

def _strip_digits(text: str) -> int | None:
    """문자열에서 숫자만 추출 → int, 없으면 None"""
    if not text:
        return None
    m = _RE_ONLY_DIGIT.sub("", text)
    return int(m) if m else None

def _extract_price_info(price_container: bs4.Tag) -> tuple[int | None, int | None, int | None]:
    """가격 정보 추출: (할인가, 원가, 할인율)"""
    # 할인가 (현재 가격)
    current_price = None
    price_elem = price_container.select_one("._2dOfJD1Omd")
    if price_elem:
        current_price = _strip_digits(price_elem.get_text())
    
    # 원가 (할인 전 가격)
    original_price = None
    original_elem = price_container.select_one("del.f35ydjnAaf")
    if original_elem:
        original_price = _strip_digits(original_elem.get_text())
    
    # 할인율
    discount_rate = None
    discount_elem = price_container.select_one("._2QqrIC4j9z")
    if discount_elem:
        discount_rate = _strip_digits(discount_elem.get_text())
    
    return current_price, original_price, discount_rate

def _extract_rating_info(rating_container: bs4.Tag) -> tuple[float | None, int | None]:
    """평점과 리뷰수 추출"""
    rating = None
    review_count = None
    
    # 평점
    rating_elem = rating_container.select_one("._2OCXSai3Xk")
    if rating_elem:
        try:
            rating = float(rating_elem.get_text().strip())
        except ValueError:
            pass
    
    # 리뷰수
    review_elems = rating_container.select("._2OCXSai3Xk")
    if len(review_elems) >= 2:
        review_count = _strip_digits(review_elems[1].get_text())
    
    return rating, review_count

# ───────────────────────  파싱 (BeautifulSoup)  ──────────────────────

def _parse_product_card(card: bs4.Tag, rank: int) -> Product:
    """개별 상품 카드 파싱"""
    
    # 상품명
    name_elem = card.select_one("._1BUPOjzreS")
    name = name_elem.get_text(strip=True) if name_elem else ""
    
    # BEST 상품 여부
    is_best = bool(card.select_one(".SfPwCdpBQs.lpmOVO_hHZ"))
    
    # 상품 URL
    product_url = None
    link_elem = card.select_one("a._2FR8OVp4H2")
    if link_elem and link_elem.has_attr("href"):
        href = link_elem["href"]
        if href.startswith("/"):
            product_url = "https://brand.naver.com" + href
        else:
            product_url = href
    
    # 이미지 URL
    image_url = None
    img_elem = card.select_one("img._23LvYmA1iE")
    if img_elem and img_elem.has_attr("src"):
        image_url = img_elem["src"]
    
    # 가격 정보
    price_container = card.select_one("._1zSEbOFiGj")
    current_price, original_price, discount_rate = (None, None, None)
    if price_container:
        current_price, original_price, discount_rate = _extract_price_info(price_container)
    
    # 평점 및 리뷰수
    rating_container = card.select_one("._3l-eQrIvNu")
    rating, review_count = (None, None)
    if rating_container:
        rating, review_count = _extract_rating_info(rating_container)
    
    # 배송 정보
    delivery_info = None
    delivery_elem = card.select_one("._2HBP0hbnFH")
    if delivery_elem:
        delivery_info = delivery_elem.get_text(strip=True)
    
    return Product(
        name=name,
        price=current_price,
        original_price=original_price,
        discount_percentage=discount_rate,
        rating=rating,
        review_count=review_count,
        product_url=product_url,
        image_url=image_url,
        rank=rank,
        is_best=is_best,
        delivery_info=delivery_info
    )

def _parse_products_from_html(html: str) -> List[Product]:
    """HTML에서 모든 상품 파싱"""
    soup = bs4.BeautifulSoup(html, "html.parser")
    products: List[Product] = []
    
    # 상품 카드들 찾기
    product_cards = soup.select(".oUod__EwiU.CbbA4PRfJd._1R-bXfdr6w")
    
    for idx, card in enumerate(product_cards, 1):
        try:
            product = _parse_product_card(card, idx)
            if product.name:  # 이름이 있는 경우만 추가
                products.append(product)
        except Exception as e:
            logging.debug(f"상품 파싱 오류 (순서 {idx}): {e}")
    
    return products

# ────────────────────────────── 크롤러 ──────────────────────────────
class NaverBrandStoreCrawler:
    """네이버 브랜드스토어 상품 목록 크롤러"""

    SORT_OPTIONS = {
        "인기도순": "POPULAR",
        "최신등록순": "RECENT", 
        "낮은가격순": "LOW_DISP_PRICE",
        "높은가격순": "HIGH_DISP_PRICE",
        "할인율순": "DISCOUNT_RATE",
        "누적판매순": "TOTALSALE",
        "리뷰많은순": "REVIEW",
        "평점높은순": "SATISFACTION"
    }

    def __init__(
        self,
        url: str,
        sort_label: str = "인기도순",
        pages: int = 1,
        headless: bool = True,
        pause: float = 1.0,
    ) -> None:
        self.url = url
        self.sort_label = sort_label
        self.pages = pages
        self.pause = pause

        opts = uc.ChromeOptions()
        if headless:
            opts.add_argument("--headless=new")
        opts.add_argument("--disable-gpu")
        opts.add_argument("--no-sandbox")
        opts.add_argument("--lang=ko-KR")

        self.driver: webdriver.Chrome = uc.Chrome(options=opts)
        self.driver.set_window_size(1400, 1000)

    def _click_sort_option(self, sort_label: str) -> None:
        """정렬 옵션 클릭"""
        try:
            # 정렬 버튼들 찾기
            sort_buttons = self.driver.find_elements(By.CSS_SELECTOR, "button._25ky2YPYcL")
            
            for button in sort_buttons:
                if button.text.strip() == sort_label:
                    button.click()
                    time.sleep(1.5)  # 정렬 후 로딩 대기
                    return
                    
            logging.warning(f"정렬 옵션 '{sort_label}'을 찾지 못했습니다.")
            
        except Exception as e:
            logging.error(f"정렬 옵션 클릭 실패: {e}")

    def _navigate_to_page(self, page_num: int) -> bool:
        """특정 페이지로 이동"""
        try:
            # 페이지네이션에서 해당 페이지 번호 클릭
            page_links = self.driver.find_elements(By.CSS_SELECTOR, "a.UWN4IvaQza")
            
            for link in page_links:
                if link.text.strip() == str(page_num):
                    link.click()
                    time.sleep(2)  # 페이지 로딩 대기
                    return True
            
            logging.warning(f"페이지 {page_num}을 찾지 못했습니다.")
            return False
            
        except Exception as e:
            logging.error(f"페이지 {page_num} 이동 실패: {e}")
            return False

    def fetch_all(self) -> List[Product]:
        """모든 페이지의 상품 수집"""
        all_products: List[Product] = []
        
        # 첫 페이지 로드
        self.driver.get(self.url)
        
        try:
            # 정렬 옵션이 로드될 때까지 대기
            WebDriverWait(self.driver, 15).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "button._25ky2YPYcL"))
            )
            
            # 정렬 옵션 선택
            if self.sort_label != "인기도순":  # 기본값이 아닌 경우만 클릭
                self._click_sort_option(self.sort_label)
            
            # 각 페이지 순회
            for page in range(1, self.pages + 1):
                logging.info(f"페이지 {page}/{self.pages} 수집 중...")
                
                if page > 1:
                    if not self._navigate_to_page(page):
                        logging.warning(f"페이지 {page} 이동 실패, 중단합니다.")
                        break
                
                # 상품 목록이 로드될 때까지 대기
                WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, ".oUod__EwiU"))
                )
                
                # 현재 페이지 상품 파싱
                page_products = _parse_products_from_html(self.driver.page_source)
                
                # 랭킹 조정 (페이지별로 순서 이어가기)
                for product in page_products:
                    if product.rank:
                        product.rank += (page - 1) * len(page_products)
                
                all_products.extend(page_products)
                logging.info(f"  페이지 {page}: {len(page_products)}개 상품 수집")
                
                time.sleep(self.pause)
        
        except Exception as e:
            logging.error(f"크롤링 중 오류 발생: {e}")
        
        logging.info(f"✔ 총 {len(all_products)}개 상품 수집 완료")
        return all_products

    def close(self):
        try:
            self.driver.quit()
        except Exception:
            pass

# ────────────────────────────── 저장 함수 ──────────────────────────────

def _save_csv(products: List[Product], outdir: Path) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    outfile = outdir / f"naver_brandstore_{ts}.csv"

    with outfile.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(Product.__annotations__.keys()))
        writer.writeheader()
        for p in products:
            writer.writerow(asdict(p))

    logging.info("저장 완료 → %s", outfile)
    return outfile

def _save_json(products: List[Product], outdir: Path) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    outfile = outdir / f"naver_brandstore_{ts}.json"
    
    with outfile.open("w", encoding="utf-8") as f:
        json.dump([asdict(p) for p in products], f, ensure_ascii=False, indent=2)
    
    logging.info("저장 완료 → %s", outfile)
    return outfile

# ────────────────────────────── CLI ──────────────────────────────

def _parse_args(argv=None):
    ap = argparse.ArgumentParser(description="네이버 브랜드스토어 크롤러")
    ap.add_argument("--url", required=True, help="브랜드스토어 카테고리 URL")
    ap.add_argument("--sort", default="인기도순", 
                   choices=list(NaverBrandStoreCrawler.SORT_OPTIONS.keys()),
                   help="정렬 옵션")
    ap.add_argument("--pages", type=int, default=1, help="크롤링할 페이지 수")
    ap.add_argument("--outdir", default="./output", help="결과 저장 폴더")
    ap.add_argument("--fmt", choices=["csv", "json"], default="csv")
    ap.add_argument("--headless", type=int, choices=[0, 1], default=1, help="0=창 표시, 1=헤드리스")
    ap.add_argument("--pause", type=float, default=1.0, help="페이지 이동 후 대기 초")
    return ap.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)

    crawler = NaverBrandStoreCrawler(
        url=args.url,
        sort_label=args.sort,
        pages=args.pages,
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
        _save_json(products, outdir)


if __name__ == "__main__":
    main()