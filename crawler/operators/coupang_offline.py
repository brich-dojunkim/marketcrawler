#!/usr/bin/env python3
"""
Coupang Brand-store ― **오프라인 HTML 파서**

브라우저에서 ‘웹페이지 전체 저장(Complete)’ 으로 내려받은
HTML 파일을 읽어 다음 필드를 추출해 csv/json 으로 저장한다.

* name
* price (정수, 원화)
* rating  (실수, 5점 만점)
* review_count (정수)
* discount_percentage (정수, %)
* image_url
* product_url
* rank  (HTML 상 등장 순서 1…N)
"""
from __future__ import annotations

import argparse, csv, json, pathlib, re, sys, datetime as dt
from dataclasses import dataclass, asdict
from typing import List, Optional

from bs4 import BeautifulSoup


@dataclass
class Product:  # 추출 필드 정의
    name: str
    price: Optional[int] = None
    rating: Optional[float] = None
    review_count: Optional[int] = None
    discount_percentage: Optional[int] = None
    image_url: Optional[str] = None
    product_url: Optional[str] = None
    rank: Optional[int] = None


RE_INT = re.compile(r"[0-9]+(?:,?[0-9]+)*")


def _num(text: str) -> Optional[int]:
    """숫자만 ⟶ int, 없으면 None"""
    m = RE_INT.search(text or "")
    return int(m.group(0).replace(",", "")) if m else None


def _parse_html(html: str) -> List[Product]:
    soup = BeautifulSoup(html, "html.parser")

    out: List[Product] = []
    for idx, li in enumerate(
        soup.select("ul.products-list li.product-wrap"), start=1
    ):
        a = li.select_one("a.product-wrapper")
        if not a:
            continue

        name = a.select_one("div.name")
        price = a.select_one("strong.price-value")
        rating_box = a.select_one("div.rating-light")
        reviews = a.select_one("span.rating-total-count")
        disc = a.select_one("span.discount-percentage")
        img = a.select_one("img")

        out.append(
            Product(
                name=(name.text or "").strip() if name else "",
                price=_num(price.text) if price else None,
                rating=float(rating_box["data-rating"])
                if rating_box and rating_box.has_attr("data-rating")
                else None,
                review_count=_num(reviews.text) if reviews else None,
                discount_percentage=_num(disc.text) if disc else None,
                image_url=img["data-src"] if img and img.has_attr("data-src") else None,
                product_url=a["href"] if a and a.has_attr("href") else None,
                rank=idx,
            )
        )
    return out


def write_csv(rows: List[Product], out_dir: pathlib.Path) -> pathlib.Path:
    out_dir.mkdir(exist_ok=True, parents=True)
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    fp = out_dir / f"brandstore_offline_{ts}.csv"

    with fp.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=Product.__annotations__.keys())
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))
    return fp


def write_json(rows: List[Product], out_dir: pathlib.Path) -> pathlib.Path:
    out_dir.mkdir(exist_ok=True, parents=True)
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    fp = out_dir / f"brandstore_offline_{ts}.json"

    with fp.open("w", encoding="utf-8-sig") as f:
        json.dump([asdict(r) for r in rows], f, ensure_ascii=False, indent=2)
    return fp


def cli(argv=None):
    ap = argparse.ArgumentParser(description="브랜드-스토어 HTML 오프라인 파서")
    ap.add_argument("--html", required=True, help="저장한 HTML 파일 경로")
    ap.add_argument("--outdir", default="./output", help="결과 저장 폴더")
    ap.add_argument("--fmt", choices=("csv", "json"), default="csv")
    args = ap.parse_args(argv)

    html_path = pathlib.Path(args.html)
    if not html_path.exists():
        sys.exit(f"[ERR] HTML 파일이 없습니다: {html_path}")

    html = html_path.read_text(encoding="utf-8", errors="ignore")
    products = _parse_html(html)

    if not products:
        sys.exit("[WARN] 상품 정보를 찾지 못했습니다 – HTML 저장이 완전한지 확인하세요.")

    out_dir = pathlib.Path(args.outdir)
    fp = write_csv(products, out_dir) if args.fmt == "csv" else write_json(products, out_dir)
    print(f"[OK] {len(products)}개 항목 저장 → {fp}")


if __name__ == "__main__":
    cli()
