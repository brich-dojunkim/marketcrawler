# -*- coding: utf-8 -*-
"""샤오미 스토어 상품 목록 → 리뷰 전수 수집 스크립트 (2025‑07‑22)
   coupang_review.CoupangReviewCrawler v5 에 맞게 약간 수정
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

from coupang_review import CoupangReviewCrawler, Review  # noqa: E402

###############################################################################
# 로그 & 유틸
###############################################################################
log = logging.getLogger("xiaomi_review")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
)

_SLUG_RE = re.compile(r"[^\w\-]+", re.U)


def slugify(text: str, maxlen: int = 40) -> str:
    s = _SLUG_RE.sub("-", text.strip().lower())
    return s[:maxlen].strip("-") or "item"


###############################################################################
# 목록 처리
###############################################################################
def _representative_title(src: str | float) -> str:
    if not isinstance(src, str):
        return "상품"
    return src.split(",")[0].strip()


def _dedup_by_product_id(rows: Iterable[Tuple[str, str]]):
    seen: set[str] = set()
    unique: list[Tuple[str, str]] = []
    for name, url in rows:
        base = url.split("?", 1)[0]
        if base in seen:
            continue
        seen.add(base)
        unique.append((name, url))
    return unique


def iter_product_urls(src: str, limit: int | None = None):
    if src.endswith(".csv"):
        df = pd.read_csv(src)
    elif src.endswith(".json"):
        df = pd.read_json(src)
    else:
        raise ValueError("지원 형식: csv | json")

    if "product_url" not in df.columns:
        raise KeyError("product_url 열이 없습니다.")

    if limit:
        df = df.head(limit)
        log.info("상위 %d개 상품만 대상", limit)

    name_col = df.columns[2] if "name" not in df.columns else "name"

    raw = [
        (_representative_title(row.get(name_col, "")), str(row["product_url"]))
        for _, row in df.iterrows()
    ]
    return _dedup_by_product_id(raw)


###############################################################################
# 저장
###############################################################################
def save_reviews(reviews: List[Review], base_out: Path, item_slug: str, fmt: str):
    if not reviews:
        log.info("    ↳ 리뷰 0건 – 스킵")
        return
    base_out.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    fn = base_out / f"reviews_{item_slug}_{ts}.{fmt}"
    df = pd.DataFrame.from_records([r.__dict__ for r in reviews])
    if fmt == "csv":
        df.to_csv(fn, index=False, quoting=csv.QUOTE_MINIMAL)
    else:
        df.to_json(fn, orient="records", force_ascii=False, indent=2)
    log.info("    ↳ 저장: %s (%d건)", fn, len(df))


###############################################################################
# 메인 루프
###############################################################################
def cli(argv: List[str] | None = None):
    ap = argparse.ArgumentParser(
        description="쿠팡 샤오미 상품 리뷰 일괄 크롤러 – 목록 CSV/JSON 기반"
    )
    ap.add_argument("src", help="xiaomi_store_crawler 결과(csv/json)")
    ap.add_argument("--pages", type=int, default=20)
    ap.add_argument("--headless", type=int, default=1)
    ap.add_argument("--fmt", choices=["csv", "json"], default="csv")
    ap.add_argument("--outdir", default="./output/reviews")
    ap.add_argument("--limit", type=int, help="상품 수 제한")
    args = ap.parse_args(argv)

    urls = list(iter_product_urls(args.src, args.limit))
    total = len(urls)
    log.info("총 %d개 상품 리뷰 수집 시작", total)

    for idx, (name, url) in enumerate(urls, 1):
        log.info("[%d/%d] %s", idx, total, name[:60])
        item_slug = slugify(name)
        try:
            crawler = CoupangReviewCrawler(
                url=url,
                pages=args.pages,
                headless=args.headless,
                fmt=args.fmt,
                outdir=args.outdir,
            )
            reviews = crawler.run()
            crawler.save(reviews)  # 부분 저장
            save_reviews(reviews, Path(args.outdir), item_slug, args.fmt)
        except Exception as e:
            log.error("⚠️ %s 리뷰 수집 실패: %s", name, e)
        finally:
            time.sleep(random.uniform(1.0, 2.5))  # 트래픽 페이스 조절

    log.info("완료 – %d개 상품", total)


if __name__ == "__main__":
    cli(sys.argv[1:])
