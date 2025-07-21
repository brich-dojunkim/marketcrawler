# -*- coding: utf-8 -*-
"""쿠팡 상품‑리뷰 크롤러 v5  (2025‑07‑22)
   • alert 차단(_safe_page_source)
   • 스텔스 Chrome 프로필 & 랜덤 UA
   • 페이지·상품 간 지연으로 트래픽 페이스 완화
"""

from __future__ import annotations

import argparse
import logging
import os
import pathlib
import random
import sys
import time
from dataclasses import asdict, dataclass
from typing import List, Optional

import pandas as pd
import undetected_chromedriver as uc
from bs4 import BeautifulSoup
from selenium.common.exceptions import (
    ElementClickInterceptedException,
    StaleElementReferenceException,
    TimeoutException,
    UnexpectedAlertPresentException,
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
    helpful_count: Optional[int]
    images: str


def _safe_int(text: str) -> Optional[int]:
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
        user_tag = art.select_one(
            ".sdp-review__article__list__info__user__name, .js_reviewUserProfileImage"
        )
        user = user_tag.get_text(strip=True) if user_tag else ""
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
# 크롬 초기화 (스텔스)
###############################################################################

# ───── coupang_review.py  /  _init_driver() 수정본 ─────
def _init_driver(headless: int):
    opt = ChromeOptions()

    # ❶ 창/헤드리스
    if headless:
        opt.add_argument("--headless=new")

    # ❷ 충돌 줄이는 공통 플래그
    opt.add_argument("--lang=ko-KR")
    opt.add_argument("--no-sandbox")
    opt.add_argument("--disable-dev-shm-usage")
    opt.add_argument("--disable-gpu")
    opt.add_argument("--disable-software-rasterizer")
    opt.add_argument("--disable-extensions")
    opt.add_argument("--disable-blink-features=AutomationControlled")
    opt.add_argument("--remote-debugging-port=0")   # 임의 포트 자동 할당

    # ❸ 랜덤 UA (fake_useragent)
    from fake_useragent import UserAgent
    opt.add_argument(f"--user-agent={UserAgent().random}")

    # ※ user‑data‑dir 주석 처리 → 실행 충돌 최소화
    # profile_dir = os.path.expanduser("~/Library/Application Support/Google/Chrome")
    # opt.add_argument(f"--user-data-dir={profile_dir}")

    # ❹ 버전 자동 감지  ←  version_main 파라미터 **삭제**
    drv = uc.Chrome(options=opt, headless=headless)

    # webdriver 흔적 제거
    drv.execute_cdp_cmd(
        "Page.addScriptToEvaluateOnNewDocument",
        {"source": "Object.defineProperty(navigator,'webdriver',{get:()=>undefined});"},
    )
    drv.set_window_size(1200, 960)
    return drv

###############################################################################
# 크롤러
###############################################################################


class CoupangReviewCrawler:
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
        self.driver = _init_driver(headless)

    # ---------------------- low‑level helpers ---------------------- #
    def _close(self):
        try:
            self.driver.quit()
        except Exception:
            pass

    def _safe_click(self, elem, max_try: int = 3):
        for _ in range(max_try):
            try:
                elem.click()
                return
            except (ElementClickInterceptedException, StaleElementReferenceException):
                self.driver.execute_script("window.scrollBy(0, 40);")
                time.sleep(0.3)
                try:
                    self.driver.execute_script("arguments[0].click();", elem)
                    return
                except Exception:
                    pass
        raise RuntimeError("요소 클릭 실패")

    def _scroll_until(self, css, max_try=30, pause=0.6):
        for _ in range(max_try):
            if self.driver.find_elements(By.CSS_SELECTOR, css):
                return True
            self.driver.execute_script(
                "window.scrollBy(0, window.innerHeight*0.9);"
            )
            time.sleep(pause)
        return False

    def _safe_page_source(self) -> str | None:
        """Unexpected alert 를 처리하고 HTML 을 반환한다."""
        try:
            return self.driver.page_source
        except UnexpectedAlertPresentException:
            try:
                alert = self.driver.switch_to.alert
                logger.warning("⚠️ 쿠팡 alert: %s – 수락 후 재시도", alert.text[:60])
                alert.accept()
                time.sleep(2)
                return self.driver.page_source
            except Exception as e:
                logger.error("alert 처리 실패: %s", e)
                return None

    # ---------------------- review section ------------------------- #
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

    # ---------------------- pagination ----------------------------- #
    def _ensure_page_button_visible(self, page: int):
        btn_css = f".js_reviewArticlePageBtn[data-page='{page}']"
        pag_wrap = self.driver.find_element(
            By.CSS_SELECTOR, ".js_reviewArticlePagingContainer"
        )
        self.driver.execute_script(
            "arguments[0].scrollIntoView({block:'center'});", pag_wrap
        )
        for _ in range(60):
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
        raise RuntimeError(f"페이지 버튼 {page} 노출 실패")

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
        time.sleep(random.uniform(0.7, 1.2))

    # ---------------------- main ----------------------------- #
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

                html = self._safe_page_source()
                if not html:
                    logger.warning("페이지 소스 없음 – 중단")
                    break

                parsed = parse_reviews(html, p, start_rank=len(all_rs) + 1)
                if not parsed:
                    logger.warning("리뷰 0건 – 이후 페이지 없음")
                    break
                all_rs.extend(parsed)

                # 요청 페이스 조절
                time.sleep(random.uniform(0.8, 1.5))

            except Exception as e:
                logger.error("페이지 %d 오류 – 부분 수집 후 중단: %s", p, e, exc_info=True)
                break

        self._close()
        return all_rs

    # ---------------------- save ----------------------------- #
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
        else:
            df.to_json(fp, orient="records", force_ascii=False, indent=2)
        logger.info("✅ 저장 완료 → %s (%d rows)", fp, len(df))


###############################################################################
# CLI
###############################################################################


def cli(argv: List[str] | None = None):
    parser = argparse.ArgumentParser(
        description="쿠팡 상품 리뷰 스크레이퍼 (스텔스/alert 복구 대응)"
    )
    parser.add_argument("--url", required=True, help="상품 상세 URL")
    parser.add_argument("--pages", type=int, default=20, help="리뷰 페이지 수")
    parser.add_argument("--headless", type=int, choices=[0, 1], default=1)
    parser.add_argument("--fmt", default="csv", choices=["csv", "json"])
    parser.add_argument("--outdir", default="./output/reviews")
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
        crawler.save(reviews)


if __name__ == "__main__":
    cli(sys.argv[1:])
