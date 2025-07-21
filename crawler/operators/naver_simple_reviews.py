# -*- coding: utf-8 -*-
"""단순한 네이버 브랜드스토어 리뷰 크롤러 (중복 제거 기능 포함)"""

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
            time.sleep(3)
            
            # REVIEW 섹션으로 스크롤
            try:
                review_section = self.driver.find_element(By.ID, "REVIEW")
                self.driver.execute_script("arguments[0].scrollIntoView();", review_section)
                time.sleep(2)
                logger.info("리뷰 섹션 찾음")
            except:
                # ID로 못 찾으면 일반적인 방법으로
                self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight * 0.7);")
                time.sleep(2)
                logger.info("페이지 하단으로 스크롤")
            
            # 리뷰 있는지 확인
            review_items = self.driver.find_elements(By.CSS_SELECTOR, "li.BnwL_cs1av")
            if not review_items:
                logger.warning("리뷰가 없습니다.")
                return []
            
            all_reviews = []
            
            # 페이지별 크롤링
            for page in range(1, min(self.pages + 1, 11)):  # 최대 10페이지
                if page > 1:
                    # 리뷰 페이지네이션 클릭 (수정된 셀렉터)
                    page_links = self.driver.find_elements(By.CSS_SELECTOR, "a.bd_12qvo")
                    found = False
                    for link in page_links:
                        if link.text.strip() == str(page):
                            link.click()
                            time.sleep(2)  # 페이지 로딩 대기 증가
                            found = True
                            break
                    
                    if not found:
                        # 다음 버튼으로 페이지 범위 확장 시도
                        next_buttons = self.driver.find_elements(By.CSS_SELECTOR, "a.bd_GEHgm.bd_3DjoC")
                        for btn in next_buttons:
                            if "다음" in btn.text:
                                btn.click()
                                time.sleep(2)
                                # 다음 버튼 클릭 후 다시 페이지 찾기
                                page_links = self.driver.find_elements(By.CSS_SELECTOR, "a.bd_12qvo")
                                for link in page_links:
                                    if link.text.strip() == str(page):
                                        link.click()
                                        time.sleep(2)
                                        found = True
                                        break
                                break
                    
                    if not found:
                        logger.warning(f"페이지 {page}를 찾을 수 없습니다.")
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

def extract_product_id(url: str) -> str:
    """URL에서 상품 ID 추출"""
    try:
        # /products/숫자 패턴에서 숫자 추출
        match = re.search(r'/products/(\d+)', url)
        return match.group(1) if match else url.split('/')[-1]
    except:
        return url

def load_products(csv_file: str, limit: int = None) -> List[tuple]:
    """CSV에서 상품 정보 로드 (상품 ID 기반 중복 제거)"""
    df = pd.read_csv(csv_file)
    if limit:
        df = df.head(limit)
    
    products = []
    seen_product_ids = set()  # 상품 ID 기반 중복 체크
    
    for _, row in df.iterrows():
        name = str(row.get('name', ''))
        url = str(row.get('product_url', ''))
        
        if not name or not url or url == 'nan':
            continue
            
        # 상품 ID 추출
        product_id = extract_product_id(url)
        
        # 중복 체크
        if product_id in seen_product_ids:
            logger.info(f"중복 상품 스킵 (ID: {product_id}): {name[:30]}...")
            continue
            
        seen_product_ids.add(product_id)
        products.append((name, url))
    
    logger.info(f"총 {len(products)}개 상품 (중복 제거 완료)")
    return products

def save_reviews(reviews: List[NaverReview], outdir: Path, product_name: str, product_url: str):
    """리뷰 저장 (URL 기반 고유 파일명)"""
    if not reviews:
        return
    
    outdir.mkdir(parents=True, exist_ok=True)
    
    # URL에서 상품 ID 추출
    product_id = extract_product_id(product_url)
    
    # 안전한 상품명
    safe_name = re.sub(r'[^\w\s-]', '', product_name)[:30]
    safe_name = re.sub(r'[-\s]+', '-', safe_name).strip('-')
    
    filename = outdir / f"naver_reviews_{product_id}_{safe_name}_{time.strftime('%H%M%S')}.csv"
    
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
    
    # 상품 로드 (중복 제거)
    products = load_products(args.csv_file, args.limit)
    logger.info(f"총 {len(products)}개 상품 처리 시작")
    
    outdir = Path(args.outdir)
    success = 0
    fail = 0
    
    for i, (name, url) in enumerate(products, 1):
        logger.info(f"[{i}/{len(products)}] {name[:50]}...")
        
        try:
            crawler = SimpleNaverReviewCrawler(url, name, args.pages, args.headless)
            reviews = crawler.run()
            
            if reviews:
                save_reviews(reviews, outdir, name, url)
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