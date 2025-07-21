# -*- coding: utf-8 -*-
"""네이버 브랜드스토어 상품 리뷰 일괄 수집 스크립트
   naver_brandstore.py 결과를 기반으로 각 상품의 리뷰를 수집
"""

from __future__ import annotations

import argparse
import csv
import logging
import random
import re
import sys
import time
from pathlib import Path
from typing import Iterable, List, Tuple

import pandas as pd

# naver_review.py에서 크롤러 임포트
try:
    from naver_review import NaverReviewCrawler, NaverReview
except ImportError:
    print("naver_review.py 파일이 필요합니다.")
    sys.exit(1)

###############################################################################
# 로깅 설정
###############################################################################
log = logging.getLogger("naver_bulk_reviews")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
)

###############################################################################
# 유틸리티 함수
###############################################################################

def slugify(text: str, maxlen: int = 40) -> str:
    """안전한 파일명을 위한 슬러그 생성"""
    s = re.sub(r'[^\w\-]+', '-', text.strip().lower())
    return s[:maxlen].strip('-') or 'item'

def extract_representative_title(src: str | float) -> str:
    """상품명에서 대표 제목 추출"""
    if not isinstance(src, str):
        return "상품"
    return src.split(',')[0].strip()

def deduplicate_urls(rows: Iterable[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """URL 중복 제거"""
    seen: set[str] = set()
    unique: list[Tuple[str, str]] = []
    
    for name, url in rows:
        # 쿼리 파라미터 제거한 베이스 URL로 중복 체크
        base = url.split('?', 1)[0]
        if base in seen:
            continue
        seen.add(base)
        unique.append((name, url))
    
    return unique

def load_product_urls(src: str, limit: int | None = None) -> List[Tuple[str, str]]:
    """CSV/JSON에서 상품 URL 목록 로드"""
    try:
        if src.endswith('.csv'):
            df = pd.read_csv(src)
        elif src.endswith('.json'):
            df = pd.read_json(src)
        else:
            raise ValueError("지원 형식: csv | json")

        if 'product_url' not in df.columns:
            raise KeyError("product_url 열이 없습니다.")

        if limit:
            df = df.head(limit)
            log.info(f"상위 {limit}개 상품만 대상")

        # 상품명 컬럼 찾기
        name_col = 'name' if 'name' in df.columns else df.columns[0]

        raw_data = [
            (extract_representative_title(row.get(name_col, "")), str(row["product_url"]))
            for _, row in df.iterrows()
            if pd.notna(row.get("product_url"))
        ]

        return deduplicate_urls(raw_data)

    except Exception as e:
        log.error(f"파일 로드 실패: {e}")
        return []

def save_individual_reviews(reviews: List[NaverReview], base_out: Path, item_slug: str, fmt: str):
    """개별 상품 리뷰 저장"""
    if not reviews:
        log.info("    ↳ 리뷰 0건 – 스킵")
        return

    base_out.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    filename = base_out / f"naver_reviews_{item_slug}_{ts}.{fmt}"

    df = pd.DataFrame.from_records([r.__dict__ for r in reviews])
    
    if fmt == "csv":
        df.to_csv(filename, index=False, encoding='utf-8-sig', quoting=csv.QUOTE_MINIMAL)
    else:
        df.to_json(filename, orient="records", force_ascii=False, indent=2)

    log.info(f"    ↳ 저장: {filename} ({len(df)}건)")

###############################################################################
# 메인 크롤링 로직
###############################################################################

def crawl_product_reviews(
    name: str, 
    url: str, 
    pages: int, 
    sort_option: str,
    headless: int, 
    fmt: str, 
    outdir: Path
) -> List[NaverReview]:
    """단일 상품 리뷰 크롤링"""
    
    item_slug = slugify(name)
    
    try:
        crawler = NaverReviewCrawler(
            url=url,
            product_name=name,
            pages=pages,
            sort_option=sort_option,
            headless=headless,
            fmt=fmt,
            outdir=outdir,
        )
        
        reviews = crawler.run()
        
        # 개별 저장
        save_individual_reviews(reviews, outdir, item_slug, fmt)
        
        return reviews
        
    except Exception as e:
        log.error(f"크롤링 실패: {e}")
        return []

###############################################################################
# CLI
###############################################################################

def main(argv: List[str] | None = None):
    parser = argparse.ArgumentParser(
        description="네이버 브랜드스토어 상품 리뷰 일괄 크롤러 – CSV/JSON 기반"
    )
    parser.add_argument("src", help="naver_brandstore 결과 파일 (csv/json)")
    parser.add_argument("--pages", type=int, default=5, help="페이지당 리뷰 페이지 수")
    parser.add_argument("--sort", default="랭킹순", 
                       choices=["랭킹순", "최신순", "평점 높은순", "평점 낮은순"],
                       help="리뷰 정렬 옵션")
    parser.add_argument("--headless", type=int, default=1, choices=[0, 1])
    parser.add_argument("--fmt", choices=["csv", "json"], default="csv")
    parser.add_argument("--outdir", default="./output/naver_reviews")
    parser.add_argument("--limit", type=int, help="처리할 상품 수 제한")
    parser.add_argument("--delay", type=float, default=3.0, help="상품 간 대기시간(초)")
    parser.add_argument("--retry", type=int, default=2, help="실패 시 재시도 횟수")
    
    args = parser.parse_args(argv)

    # 상품 URL 목록 로드
    products = load_product_urls(args.src, args.limit)
    total = len(products)
    
    if not products:
        log.error("처리할 상품이 없습니다.")
        return

    log.info(f"총 {total}개 상품 리뷰 크롤링 시작")
    log.info(f"설정: 페이지={args.pages}, 정렬={args.sort}, 헤드리스={args.headless}")

    # 통계 변수
    success_count = 0
    fail_count = 0
    total_reviews = 0
    
    # 결과 저장을 위한 디렉토리 생성
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 각 상품별 크롤링
    for idx, (name, url) in enumerate(products, 1):
        log.info(f"\n[{idx}/{total}] {name[:60]}...")
        log.info(f"    URL: {url}")
        
        retry_count = 0
        reviews = []
        
        # 재시도 로직
        while retry_count <= args.retry:
            try:
                reviews = crawl_product_reviews(
                    name=name,
                    url=url,
                    pages=args.pages,
                    sort_option=args.sort,
                    headless=args.headless,
                    fmt=args.fmt,
                    outdir=outdir
                )
                
                if reviews:
                    success_count += 1
                    total_reviews += len(reviews)
                    log.info(f"    ✅ 성공: {len(reviews)}개 리뷰 수집")
                    break
                else:
                    log.warning(f"    ⚠️ 리뷰 0건")
                    break
                    
            except Exception as e:
                retry_count += 1
                log.error(f"    ❌ 시도 {retry_count} 실패: {e}")
                
                if retry_count <= args.retry:
                    log.info(f"    🔄 {retry_count + 1}초 후 재시도...")
                    time.sleep(retry_count + 1)
                else:
                    fail_count += 1
                    log.error(f"    ❌ 최종 실패: {name}")
                    break
        
        # 상품 간 대기 (마지막 상품이 아닌 경우)
        if idx < total:
            delay = random.uniform(args.delay, args.delay + 1.5)
            log.info(f"    ⏳ {delay:.1f}초 대기...")
            time.sleep(delay)

    # 최종 결과 요약
    log.info(f"\n" + "="*50)
    log.info(f"크롤링 완료!")
    log.info(f"  • 총 상품: {total}개")
    log.info(f"  • 성공: {success_count}개")
    log.info(f"  • 실패: {fail_count}개")
    log.info(f"  • 수집된 리뷰: {total_reviews:,}개")
    log.info(f"  • 성공률: {success_count/total*100:.1f}%")
    log.info(f"  • 결과 위치: {outdir}")
    log.info(f"="*50)

    # 요약 통계 파일 저장
    summary = {
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'total_products': total,
        'success_count': success_count,
        'fail_count': fail_count,
        'total_reviews': total_reviews,
        'success_rate': round(success_count/total*100, 2) if total > 0 else 0,
        'settings': {
            'pages': args.pages,
            'sort': args.sort,
            'headless': args.headless,
            'format': args.fmt
        }
    }
    
    summary_file = outdir / f"crawling_summary_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        import json
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    log.info(f"요약 파일 저장: {summary_file}")


if __name__ == "__main__":
    main(sys.argv[1:])