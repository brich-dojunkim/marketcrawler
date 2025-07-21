# -*- coding: utf-8 -*-
"""단순한 네이버 브랜드스토어 리뷰 크롤러"""

from __future__ import annotations

import argparse
import csv
import logging
import random
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

import pandas as pd
import undetected_chromedriver as uc
from bs4 import BeautifulSoup
from selenium.webdriver import ChromeOptions
from selenium.webdriver.common.by import By

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s | %(levelname)7s | %(message)s", level=logging.INFO)

@dataclass
class NaverReview:
    rank: int
    rating: int
    date: str
    user: str
    content: str
    helpful_count: Optional[int]
    images: str
    option_info: str
    product_name: str

def parse_naver_reviews(html: str, product_name: str = "") -> List[NaverReview]:
    """네이버 리뷰 파싱"""
    soup = BeautifulSoup(html, "html.parser")
    review_items = soup.select("li.BnwL_cs1av")
    
    reviews = []
    for i, item in enumerate(review_items, 1):
        try:
            # 평점
            rating_elem = item.select_one("span._3hrdz1EXfJ")
            rating = 0
            if rating_elem and rating_elem.get('style'):
                width_match = re.search(r'width:\s*(\d+)%', rating_elem.get('style', ''))
                if width_match:
                    rating = round(int(width_match.group(1)) / 20)
            
            # 날짜
            date_spans = item.select("span._2L3vDiadT9")
            date = ""
            for span in date_spans:
                text = span.get_text(strip=True)
                if re.match(r'\d{2}\.\d{2}\.\d{2}', text):
                    date = text
                    break
            
            # 사용자
            user_elem = item.select_one("strong._2L3vDiadT9")
            user = user_elem.get_text(strip=True) if user_elem else ""
            
            # 리뷰 내용
            content_elem = item.select_one("div._1kMfD5ErZ6 span._2L3vDiadT9")
            content = content_elem.get_text(strip=True) if content_elem else ""
            
            # 도움수
            helpful_elem = item.select_one("span.count")
            helpful_count = 0
            if helpful_elem:
                try:
                    helpful_count = int(re.sub(r'[^0-9]', '', helpful_elem.get_text()))
                except:
                    helpful_count = 0
            
            # 이미지
            img_elems = item.select("img[alt='review_image']")
            images = []
            for img in img_elems:
                src = img.get('data-src') or img.get('src')
                if src:
                    images.append(src)
            
            # 옵션
            option_elem = item.select_one("div._2FXNMst_ak")
            option_info = option_elem.get_text(strip=True) if option_elem else ""
            
            reviews.append(NaverReview(
                rank=i,
                rating=rating,
                date=date,
                user=user,
                content=content,
                helpful_count=helpful_count,
                images=",".join(images),
                option_info=option_info,
                product_name=product_name
            ))
            
        except Exception as e:
            logger.debug(f"리뷰 파싱 오류: {e}")
            continue
    
    return reviews

class SimpleNaverReviewCrawler:
    def __init__(self, url: str, product_name: str, pages: int = 1, headless: int = 1):
        self.url = url
        self.product_name = product_name
        self.pages = pages
        
        opts = ChromeOptions()
        if headless:
            opts.add_argument("--headless=new")
        opts.add_argument("--no-sandbox")
        opts.add_argument("--disable-dev-shm-usage")
        
        self.driver = uc.Chrome(options=opts, headless=headless)
        self.driver.set_window_size(1400, 1000)

    def run(self) -> List[NaverReview]:
        """리뷰 크롤링 실행"""
        try:
            logger.info(f"크롤링 시작: {self.product_name[:30]}...")
            
            # 페이지 접속
            self.driver.get(self.url)
            time.sleep(2)
            
            # 리뷰 섹션으로 스크롤
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(1)
            
            # 리뷰 있는지 확인
            review_items = self.driver.find_elements(By.CSS_SELECTOR, "li.BnwL_cs1av")
            if not review_items:
                logger.warning("리뷰가 없습니다.")
                return []
            
            all_reviews = []
            
            # 페이지별 크롤링
            for page in range(1, min(self.pages + 1, 11)):  # 최대 10페이지
                if page > 1:
                    # 페이지 클릭
                    page_links = self.driver.find_elements(By.CSS_SELECTOR, f"a.UWN4IvaQza")
                    found = False
                    for link in page_links:
                        if link.text.strip() == str(page):
                            link.click()
                            time.sleep(1)
                            found = True
                            break
                    if not found:
                        break
                
                # 리뷰 파싱
                html = self.driver.page_source
                reviews = parse_naver_reviews(html, self.product_name)
                
                if not reviews:
                    break
                
                # 랭킹 조정
                for review in reviews:
                    review.rank = len(all_reviews) + review.rank
                
                all_reviews.extend(reviews)
                logger.info(f"페이지 {page}: {len(reviews)}개 리뷰")
            
            logger.info(f"완료: 총 {len(all_reviews)}개 리뷰")
            return all_reviews
            
        except Exception as e:
            logger.error(f"크롤링 실패: {e}")
            return []
        finally:
            self.driver.quit()

def load_products(csv_file: str, limit: int = None) -> List[tuple]:
    """CSV에서 상품 정보 로드"""
    df = pd.read_csv(csv_file)
    if limit:
        df = df.head(limit)
    
    products = []
    for _, row in df.iterrows():
        name = str(row.get('name', ''))
        url = str(row.get('product_url', ''))
        if name and url and url != 'nan':
            products.append((name, url))
    
    return products

def save_reviews(reviews: List[NaverReview], outdir: Path, product_name: str):
    """리뷰 저장"""
    if not reviews:
        return
    
    outdir.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r'[^\w\s-]', '', product_name)[:30]
    safe_name = re.sub(r'[-\s]+', '-', safe_name).strip('-')
    
    filename = outdir / f"naver_reviews_{safe_name}_{time.strftime('%H%M%S')}.csv"
    
    df = pd.DataFrame([asdict(r) for r in reviews])
    df.to_csv(filename, index=False, encoding='utf-8-sig')
    
    logger.info(f"저장: {filename} ({len(reviews)}개)")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_file", help="상품 목록 CSV 파일")
    parser.add_argument("--limit", type=int, default=100, help="상품 수 제한")
    parser.add_argument("--pages", type=int, default=5, help="리뷰 페이지 수")
    parser.add_argument("--headless", type=int, default=1, choices=[0, 1])
    parser.add_argument("--outdir", default="./output/naver_reviews")
    parser.add_argument("--delay", type=float, default=2.0, help="상품간 대기시간")
    
    args = parser.parse_args()
    
    # 상품 로드
    products = load_products(args.csv_file, args.limit)
    logger.info(f"총 {len(products)}개 상품 처리")
    
    outdir = Path(args.outdir)
    success = 0
    fail = 0
    
    for i, (name, url) in enumerate(products, 1):
        logger.info(f"[{i}/{len(products)}] {name[:50]}...")
        
        try:
            crawler = SimpleNaverReviewCrawler(url, name, args.pages, args.headless)
            reviews = crawler.run()
            
            if reviews:
                save_reviews(reviews, outdir, name)
                success += 1
            else:
                fail += 1
                
        except Exception as e:
            logger.error(f"실패: {e}")
            fail += 1
        
        # 대기
        if i < len(products):
            time.sleep(random.uniform(args.delay, args.delay + 1))
    
    logger.info(f"완료 - 성공: {success}, 실패: {fail}")

if __name__ == "__main__":
    main()