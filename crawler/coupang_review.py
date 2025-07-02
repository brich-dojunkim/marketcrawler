# -*- coding: utf-8 -*-
"""쿠팡 상품-리뷰 크롤러 (2025-07 리뉴얼 + 오류-복구 & 부분 저장 v4)"""

from __future__ import annotations

import argparse
import logging
import os
import pathlib
import sys
import time
from dataclasses import asdict, dataclass
from typing import List

import pandas as pd
import undetected_chromedriver as uc
from bs4 import BeautifulSoup
from selenium.common.exceptions import (
    ElementClickInterceptedException,
    StaleElementReferenceException,
    TimeoutException,
)
from selenium.webdriver import ChromeOptions
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s | %(levelname)7s | %(message)s", level=logging.INFO
)

###############################################################################
# 파싱 유틸
###############################################################################


@dataclass
class Review:
    rank: int
    rating: int
    date: str
    user: str
    headline: str
    body: str
    helpful_count: int | None
    images: str


def _safe_int(text: str) -> int | None:
    try:
        return int(text.replace(",", "").strip())
    except Exception:
        return None


def parse_reviews(html: str, page: int, start_rank: int = 1) -> List[Review]:
    soup = BeautifulSoup(html, "html.parser")
    arts = soup.select("article.sdp-review__article__list")
    rs: list[Review] = []
    for i, art in enumerate(arts):
        rank = start_rank + i
        rating = int(
            art.select_one(
                ".sdp-review__article__list__info__product-info__star-orange"
            ).get("data-rating", 0)
        )
        date = (
            art.select_one(
                ".sdp-review__article__list__info__product-info__reg-date"
            ).get_text(strip=True)
            if art.select_one(
                ".sdp-review__article__list__info__product-info__reg-date"
            )
            else ""
        )
        user = (
            art.select_one(".sdp-review__article__list__info__user__name")
            or art.select_one(".js_reviewUserProfileImage")
        ).get_text(strip=True)
        headline_tag = art.select_one(
            ".sdp-review__article__list__headline, "
            ".sdp-review__article__list__review__content"
        )
        headline = headline_tag.get_text(strip=True) if headline_tag else ""
        body_tag = art.select_one(".js_reviewArticleContent")
        body = body_tag.get_text("\n", strip=True) if body_tag else ""
        helpful_tag = art.select_one(
            ".sdp-review__article__list__help__count strong"
        )
        helpful = _safe_int(helpful_tag.get_text()) if helpful_tag else None
        imgs = [
            img["src"]
            for img in art.select(".sdp-review__article__list__attachment__img")
        ]
        rs.append(
            Review(
                rank,
                rating,
                date,
                user,
                headline,
                body,
                helpful,
                ",".join(imgs),
            )
        )
    logger.info("  ↳ 파싱 완료: %d건 (page %d)", len(rs), page)
    return rs


###############################################################################
# 크롤러
###############################################################################


class CoupangReviewCrawler:
    # ------------------------------------------------------------------ #
    # init
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        url: str,
        pages: int = 1,
        headless: int = 1,
        fmt: str = "csv",
        outdir: str | os.PathLike = "./output",
    ):
        self.url = url
        self.pages = pages
        self.fmt = fmt.lower()
        self.outdir = pathlib.Path(outdir)
        self.driver = self._init_driver(headless)

    @staticmethod
    def _init_driver(headless: int):
        opt = ChromeOptions()
        if headless:
            opt.add_argument("--headless=new")
        opt.add_argument("--lang=ko-KR")
        opt.add_argument("--disable-gpu")
        opt.add_argument("--no-sandbox")
        drv = uc.Chrome(options=opt, headless=headless)
        drv.set_window_size(1200, 960)
        return drv

    def _close(self):
        try:
            self.driver.quit()
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def _safe_click(self, elem, max_try: int = 3):
        for _ in range(max_try):
            try:
                elem.click()
                return
            except ElementClickInterceptedException:
                try:
                    self.driver.execute_script("arguments[0].click();", elem)
                    return
                except Exception:
                    pass
            except StaleElementReferenceException:
                pass
            self.driver.execute_script("window.scrollBy(0, 20);")
            time.sleep(0.3)
        raise RuntimeError("요소 클릭 실패(가림/인터셉트)")

    def _scroll_until(self, css, max_try=30, pause=0.6):
        for _ in range(max_try):
            if self.driver.find_elements(By.CSS_SELECTOR, css):
                return True
            self.driver.execute_script(
                "window.scrollBy(0, window.innerHeight*0.9);"
            )
            time.sleep(pause)
        return False

    def _open_review_section(self):
        self.driver.get(self.url)
        if not self._scroll_until(".sdp-review"):
            raise RuntimeError("리뷰 섹션을 찾지 못했습니다.")
        root = self.driver.find_element(By.CSS_SELECTOR, ".sdp-review")
        self.driver.execute_script(
            "arguments[0].scrollIntoView({block:'center'});", root
        )
        WebDriverWait(self.driver, 15).until(
            EC.presence_of_element_located(
                (By.CSS_SELECTOR, ".js_reviewArticleListContainer")
            )
        )

    # ------------------------------------------------------------------ #
    # pagination
    # ------------------------------------------------------------------ #
    def _ensure_page_button_visible(self, page: int):
        btn_css = f".js_reviewArticlePageBtn[data-page='{page}']"
        pag_wrap = self.driver.find_element(
            By.CSS_SELECTOR, ".js_reviewArticlePagingContainer"
        )
        self.driver.execute_script(
            "arguments[0].scrollIntoView({block:'center'});", pag_wrap
        )
        for _ in range(60):  # 넉넉한 반복
            if self.driver.find_elements(By.CSS_SELECTOR, btn_css):
                return
            nxt = self.driver.find_elements(
                By.CSS_SELECTOR, ".sdp-review__article__page__next--active"
            )
            if not nxt:
                break
            self._safe_click(nxt[0])
            try:
                WebDriverWait(self.driver, 5).until(EC.staleness_of(nxt[0]))
            except TimeoutException:
                pass
            time.sleep(0.4)
        raise RuntimeError(f"페이지 버튼 {page} 를 노출시키지 못했습니다.")

    def _go_to_page(self, page: int):
        if page == 1:
            return
        self._ensure_page_button_visible(page)
        btn = WebDriverWait(self.driver, 10).until(
            EC.element_to_be_clickable(
                (
                    By.CSS_SELECTOR,
                    f"button.js_reviewArticlePageBtn[data-page='{page}']",
                )
            )
        )
        self._safe_click(btn)
        time.sleep(0.8)

    # ------------------------------------------------------------------ #
    # main
    # ------------------------------------------------------------------ #
    def run(self) -> List[Review]:
        logger.info("▶ 리뷰 섹션 오픈 중…")
        try:
            self._open_review_section()
        except Exception as e:
            logger.error("리뷰 섹션 열기 실패: %s", e, exc_info=True)
            self._close()
            return []

        all_rs: list[Review] = []

        for p in range(1, self.pages + 1):
            logger.info("▶ 리뷰 페이지 %d", p)
            try:
                if p > 1:
                    self._go_to_page(p)
                html = self.driver.page_source
                parsed = parse_reviews(html, p, start_rank=len(all_rs) + 1)
                if not parsed:
                    logger.warning("리뷰 파싱 0건 – 더 이상 페이지가 없을 수 있습니다.")
                    break
                all_rs.extend(parsed)
            except Exception as e:
                logger.error(
                    "페이지 %d 처리 중 오류, 크롤링을 중단합니다: %s", p, e, exc_info=True
                )
                break  # 부분 데이터라도 반환
        self._close()
        return all_rs

    # ------------------------------------------------------------------ #
    # save
    # ------------------------------------------------------------------ #
    def save(self, reviews: List[Review]):
        if not reviews:
            logger.warning("저장할 리뷰가 없습니다.")
            return
        self.outdir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame([asdict(rv) for rv in reviews])
        ts = time.strftime("%Y%m%d_%H%M%S")
        fp = self.outdir / f"coupang_reviews_{ts}.{self.fmt}"
        if self.fmt == "csv":
            df.to_csv(fp, index=False)
        elif self.fmt == "json":
            df.to_json(fp, orient="records", force_ascii=False, indent=2)
        else:
            raise ValueError("fmt 는 csv | json 중 하나여야 합니다.")
        logger.info("✅ 저장 완료 → %s (%d rows)", fp, len(df))


###############################################################################
# CLI
###############################################################################


def cli(argv: List[str] | None = None):
    parser = argparse.ArgumentParser(
        description="쿠팡 상품 리뷰 스크레이퍼 (2025-07, 오류-복구/부분-저장 대응)"
    )
    parser.add_argument("--url", required=True, help="상품 상세 URL")
    parser.add_argument("--pages", type=int, default=1, help="가져올 리뷰 페이지 수")
    parser.add_argument("--headless", type=int, choices=[0, 1], default=1)
    parser.add_argument("--fmt", default="csv", choices=["csv", "json"])
    parser.add_argument("--outdir", default="./output")
    args = parser.parse_args(argv)

    crawler = CoupangReviewCrawler(
        url=args.url,
        pages=args.pages,
        headless=args.headless,
        fmt=args.fmt,
        outdir=args.outdir,
    )

    reviews: List[Review] = []
    try:
        reviews = crawler.run()
    finally:
        # run() 이 어떤 상황이든 반환한 분량까지 저장
        crawler.save(reviews)


if __name__ == "__main__":
    cli(sys.argv[1:])
