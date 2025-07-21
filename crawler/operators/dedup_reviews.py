#!/usr/bin/env python3
# dedup_reviews.py ----------------------------------------------------------- #
#   리뷰 CSV 폴더 → 내용이 90 % 이상 동일한 파일은 중복 처리 + 대표 제목으로 새 이름
# --------------------------------------------------------------------------- #
from __future__ import annotations

import argparse
import hashlib
import logging
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Set

import pandas as pd

log = logging.getLogger("dedup_reviews")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
)

_SLUG_RE = re.compile(r"[^\w\-]+", re.U)
_FILE_RE = re.compile(r"^reviews_(.+?)_\d{8}_\d{6}$")
THRESHOLD = 0.90                     # 90 % 이상 동일 → 중복으로 간주
_SAMPLE = 300                        # 비교시 최대 n개 리뷰만 사용(속도용)


def slugify(txt: str, maxlen: int = 40) -> str:
    slug = _SLUG_RE.sub("-", txt.strip().lower())[:maxlen].strip("-")
    if not slug or slug.isdigit():
        slug = "item"
    return slug


def _guess_name_col(df: pd.DataFrame) -> str | None:
    for c in df.columns:
        if any(k in c.lower() for k in ("product", "item", "name", "title")):
            return c
    return None


def representative_title(csv_fp: Path) -> str:
    """CSV → 대표 제품명(콤마 앞까지만). 없으면 파일명의 slug."""
    try:
        df = pd.read_csv(csv_fp, nrows=5)
        col = _guess_name_col(df)
        if col and not df[col].dropna().empty:
            raw = str(df[col].dropna().iloc[0])
            return raw.split(",")[0].strip()
    except Exception:
        pass

    m = _FILE_RE.match(csv_fp.stem)
    return m.group(1).replace("-", " ") if m else csv_fp.stem


# --------------------------------------------------------------------------- #
def _review_signature(df: pd.DataFrame) -> Set[str]:
    """리뷰 텍스트 + 평점 기반의 해시셋. 행 수가 많으면 SAMPLE 개만 사용."""
    cols = [c for c in df.columns if re.search(r"review|text|content", c, re.I)]
    txt_col = cols[0] if cols else None
    if txt_col is None:
        # 텍스트가 없으면 행 전체를 문자열로.
        sig_series = df.astype(str).agg("|".join, axis=1)
    else:
        rating_cols = [c for c in df.columns if re.search(r"rating|score|star", c, re.I)]
        merged = (
            df[txt_col].astype(str)
            + (("|" + df[rating_cols[0]].astype(str)) if rating_cols else "")
        )
        sig_series = merged
    # 샘플링(무작위 X – 처음 N개면 충분)
    sample = sig_series.head(_SAMPLE)
    return {hashlib.md5(s.encode("utf-8")).hexdigest() for s in sample}


def is_similar(sig_a: Set[str], sig_b: Set[str]) -> bool:
    """두 리뷰 해시셋이 THRESHOLD 이상 겹치면 True."""
    if not sig_a or not sig_b:
        return False
    inter = len(sig_a & sig_b)
    union_len = max(len(sig_a), len(sig_b))
    return inter / union_len >= THRESHOLD


# --------------------------------------------------------------------------- #
def process_folder(src: Path, dst: Path, dry: bool):
    dst.mkdir(parents=True, exist_ok=True)

    kept_files: List[Dict] = []          # [{path, sig}]
    slug_count: Dict[str, int] = {}

    for fp in sorted(src.glob("*.csv")):
        try:
            df = pd.read_csv(fp)
        except Exception as e:
            log.warning("❌ 읽기 실패(%s) – 건너뜀: %s", fp.name, e)
            continue

        sig = _review_signature(df)

        # ── 중복 여부 검사 ────────────────────────────────────────────────
        dup_found = False
        for k in kept_files:
            if is_similar(sig, k["sig"]):
                log.info("🗑️  중복   %-45s ≃ %s", fp.name, k["path"].name)
                dup_found = True
                break
        if dup_found:
            continue

        # ── 새로운 파일 저장 ──────────────────────────────────────────────
        title = representative_title(fp)
        slug = slugify(title)
        slug_idx = slug_count.get(slug, 0)
        slug_count[slug] = slug_idx + 1
        if slug_idx:
            slug = f"{slug}_{slug_idx+1}"

        new_name = f"reviews_{slug}_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        dst_fp = dst / new_name

        if dry:
            log.info("⏩ DRY  keep %-45s → %s", fp.name, dst_fp.name)
        else:
            shutil.copy2(fp, dst_fp)
            log.info("✅ keep  %-45s → %s", fp.name, dst_fp.name)

        kept_files.append({"path": fp, "sig": sig})


# --------------------------------------------------------------------------- #
def cli(argv: list[str] | None = None):
    ap = argparse.ArgumentParser(
        description="리뷰 CSV 폴더 → 90 % 이상 동일 리뷰는 중복으로 제거 후 복사"
    )
    ap.add_argument("input_dir", help="원본 CSV 폴더")
    ap.add_argument("output_dir", help="저장할 폴더")
    ap.add_argument("--dry", action="store_true", help="파일을 복사하지 않고 로그만 출력")
    args = ap.parse_args(argv)

    src = Path(args.input_dir).expanduser().resolve()
    dst = Path(args.output_dir).expanduser().resolve()
    if not src.is_dir():
        log.error("입력 폴더가 존재하지 않습니다: %s", src)
        sys.exit(1)

    log.info("📂 입력:  %s", src)
    log.info("📂 출력:  %s", dst)
    process_folder(src, dst, args.dry)
    log.info("🎉 완료")


if __name__ == "__main__":
    cli(sys.argv[1:])
