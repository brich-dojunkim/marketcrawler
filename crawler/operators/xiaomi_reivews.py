# xiaomi_reviews.py
# --------------------------------------------------------------------------- #
#   상품 목록 CSV/JSON  → product_url 컬럼 순회하며 리뷰 전수 수집
#   2025-07   /  Python 3.11   /  undetected-chromedriver 3.x
# --------------------------------------------------------------------------- #
from __future__ import annotations

import argparse
import csv
import logging
import re
import sys
import time
from pathlib import Path
from typing import Iterable, List, Tuple

import pandas as pd

# coupang_review.py 는 같은 디렉터리(또는 PYTHONPATH)에 있다고 가정
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
    """파일명 안전 슬러그."""
    s = _SLUG_RE.sub("-", text.strip().lower())
    return s[:maxlen].strip("-") or "item"


###############################################################################
# 대표 제목 & 중복 제거
###############################################################################
def _representative_title(src: str | float) -> str:
    """
    색상·용량 등 옵션이 붙은 풀 네임에서
    첫 번째 콤마(,) 이전까지만 잘라 ‘대표 제목’으로 사용한다.
    """
    if not isinstance(src, str):
        return "상품"
    return src.split(",")[0].strip()


def _dedup_by_product_id(rows: Iterable[Tuple[str, str]]) -> list[Tuple[str, str]]:
    """
    같은 ‘상품 번호’( …/vp/products/######## )는
    리뷰 페이지가 동일하므로 한 번만 크롤링.
    """
    seen: set[str] = set()
    unique: list[Tuple[str, str]] = []
    for name, url in rows:
        base = url.split("?", 1)[0]      # 물음표 이전 Path → “…/products/상품번호”
        if base in seen:
            continue
        seen.add(base)
        unique.append((name, url))
    return unique


###############################################################################
# 메인 처리
###############################################################################
def iter_product_urls(src: str) -> Iterable[tuple[str, str]]:
    """CSV/JSON → (상품이름, URL) Generator."""
    if src.endswith(".csv"):
        df = pd.read_csv(src)
    elif src.endswith(".json"):
        df = pd.read_json(src)
    else:
        raise ValueError("지원하지 않는 파일 형식 – csv / json")

    if "product_url" not in df.columns:
        raise KeyError("product_url 열이 없습니다.")

    name_col = df.columns[2] if "name" not in df.columns else "name"  # heuristics

    # (원본명, URL) 목록
    raw: list[tuple[str, str]] = [
        (
            _representative_title(row.get(name_col, "")),
            str(row["product_url"]),
        )
        for _, row in df.iterrows()
    ]

    dedup = _dedup_by_product_id(raw)
    if len(dedup) < len(raw):
        log.info("  ↳ 중복 제거: %d → %d", len(raw), len(dedup))
    return dedup


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
# CLI & 실행 루프
###############################################################################
def cli(argv: List[str] | None = None):
    ap = argparse.ArgumentParser(
        description="쿠팡 샤오미 상품 리뷰 일괄 크롤러 – 상품목록 CSV/JSON 기반"
    )
    ap.add_argument("src", help="xiaomi_store_crawler 가 만든 CSV/JSON 경로")
    ap.add_argument("--pages", type=int, default=20, help="상품당 리뷰 페이지 수 (기본 20)")
    ap.add_argument("--headless", type=int, default=1, help="chromedriver headless 1/0")
    ap.add_argument("--fmt", choices=["csv", "json"], default="csv")
    ap.add_argument("--outdir", default="./output/reviews", help="저장 폴더")
    args = ap.parse_args(argv)

    urls = list(iter_product_urls(args.src))
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
            crawler.save(reviews)  # 부분저장 로직 포함
            save_reviews(reviews, Path(args.outdir), item_slug, args.fmt)
        except Exception as e:
            log.error("⚠️ %s 리뷰 수집 실패: %s", name, e)
        finally:
            time.sleep(1)  # 서버 부하 방지 – 잠깐 쉼

    log.info("완료 – %d개 상품", total)


if __name__ == "__main__":
    cli(sys.argv[1:])
