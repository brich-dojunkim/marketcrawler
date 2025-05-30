# cjonstyle.py  –  CJ온스타일 여성의류 Top-100 크롤러
# ──────────────────────────────────────────────────────
#  • URL·개수·헤드리스 모드 모두 코드 안에 기본값으로 고정
#  • 결과 열 순서 :  순위, 상품명, 브랜드명, 상품ID, 정가, 최종가, 할인율,
#                  가격내역, 혜택, 리뷰/구매수, 수집시각, 상품URL
#  • SSL·페이지네이션·자연스크롤 그대로 유지
# ------------------------------------------------------
from __future__ import annotations
import dataclasses as dc, re, ssl, time, logging, datetime as dt, urllib.request
from pathlib import Path
from typing import List, Optional

import certifi, pandas as pd, undetected_chromedriver as uc
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# ────────────────────── 전역 설정 ──────────────────────
TARGET_URL  = "https://display.cjonstyle.com/p/category/categoryMain?dpCateId=G00011"
MAX_PAGES   = 10          # 페이지네이션 상한
TARGET_EACH = 120         # 페이지당 스크롤 목표
LIMIT       = 100         # 총 수집 개수
OUT_DIR     = Path("output")
HEADLESS    = True        # 화면 안 띄우기

# ──────────────── SSL 우회 (certifi) ───────────────────
ctx = ssl.create_default_context(cafile=certifi.where())
urllib.request.install_opener(
    urllib.request.build_opener(urllib.request.HTTPSHandler(context=ctx))
)

# ──────────────────── 헬퍼 함수 ────────────────────────
def digits(text: str) -> Optional[int]:
    m = re.sub(r"[^\d]", "", text or "")
    return int(m) if m else None

def join_or_blank(parts: List[str]) -> str:
    return ";".join(p for p in parts if p)

def split_brand_title(raw: str) -> tuple[str, str]:
    """'[브랜드] 상품명' 또는 '브랜드 상품명' 패턴 분리"""
    if m := re.match(r"\s*\[([^\]]+)\]\s*(.*)", raw):
        return m.group(1), m.group(2).strip()
    parts = raw.split(maxsplit=1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return "", raw

# ───────────────── 데이터 클래스 ───────────────────────
@dc.dataclass
class Product:
    rank: int
    name: str
    brand: str
    item_id: str
    list_price: Optional[int]
    sale_price: Optional[int]
    discount: Optional[int]
    price_detail: str
    benefit: str
    review_cnt: Optional[int]
    collected_at: str
    url: str

    @classmethod
    def from_li(cls, li, rank: int, ts: str) -> "Product":
        # 텍스트 안전 추출
        def txt(css: str) -> str:
            try:
                return li.find_element(By.CSS_SELECTOR, css).text.strip()
            except:
                return ""

        # a.gaclass 요소
        a_item = li.find_element(By.CSS_SELECTOR, "a.gaclass")

        # 상품ID를 a.gaclass 속성에서 가져오기
        item_id = (
            a_item.get_attribute("data-dqid")
            or a_item.get_attribute("data-item-cd")
            or ""
        )

        # 원본 이름 & 브랜드/상품명 분리
        raw_name = txt("strong.goods_name")
        brand_txt, title = split_brand_title(raw_name)
        brand = txt(".shop_area a.goBrand") or brand_txt

        # 가격 정보
        sale   = digits(txt(".wrap_price ins .num"))
        normal = digits(txt(".wrap_price del .num"))
        disc   = digits(txt(".discount .num"))
        price_detail = join_or_blank([
            f"할인가{sale:,}원~"   if sale   else "",
            f"정상가{normal:,}원~" if normal else "",
        ])

        # 혜택 정보
        benefits = [b.text.strip() for b in li.find_elements(By.CSS_SELECTOR, ".info_benefit .benefit_item")]
        if li.find_elements(By.CSS_SELECTOR, ".shop_area .ico_free"):
            benefits.append("무료배송")
        benefit = join_or_blank(benefits)

        # 리뷰/구매수
        review_cnt = digits(txt(".customer_count .num"))

        # 링크
        link = a_item.get_attribute("href") or ""

        return cls(
            rank        = rank,
            name        = title,
            brand       = brand,
            item_id     = item_id,
            list_price  = normal,
            sale_price  = sale,
            discount    = disc,
            price_detail= price_detail,
            benefit     = benefit,
            review_cnt  = review_cnt,
            collected_at= ts,
            url         = link,
        )

    def to_record(self) -> dict:
        return {
            "순위"       : self.rank,
            "상품명"     : self.name,
            "브랜드명"   : self.brand,
            "상품ID"     : self.item_id,
            "정가"       : self.list_price,
            "최종가"     : self.sale_price,
            "할인율"     : self.discount,
            "가격내역"   : self.price_detail,
            "혜택"       : self.benefit,
            "리뷰/구매수" : self.review_cnt,
            "수집시각"   : self.collected_at,
            "상품URL"    : self.url,
        }

# ──────────────── Selenium 드라이버 ───────────────────
def build_driver() -> webdriver.Chrome:
    opt = uc.ChromeOptions()
    if HEADLESS:
        opt.add_argument("--headless=new")
    opt.add_argument("--no-sandbox")
    opt.add_argument("--disable-gpu")
    opt.add_argument("--lang=ko-KR")
    return uc.Chrome(options=opt)

def smooth_scroll(driver, want: int, pause: float = .4):
    last = 0; stagnant = 0
    while True:
        driver.execute_script("window.scrollBy(0,800)")
        time.sleep(pause)
        count = len(driver.find_elements(By.CSS_SELECTOR, "ul.lst_cate_result>li"))
        if count >= want:
            break
        cur = driver.execute_script("return document.documentElement.scrollTop+window.innerHeight")
        if cur == last:
            stagnant += 1
            if stagnant >= 5:
                break
        else:
            stagnant = 0; last = cur

# ──────────────────── 메인 루프 ───────────────────────
def crawl():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    OUT_DIR.mkdir(exist_ok=True)
    drv = build_driver()
    ts = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    products: List[Product] = []

    try:
        drv.get(TARGET_URL)
        page = 1
        while True:
            WebDriverWait(drv, 20).until(
                EC.presence_of_all_elements_located((By.CSS_SELECTOR, "ul.lst_cate_result>li"))
            )
            need = min(TARGET_EACH, LIMIT - len(products))
            smooth_scroll(drv, need)
            lis = drv.find_elements(By.CSS_SELECTOR, "ul.lst_cate_result>li")[:need]
            for li in lis:
                products.append(Product.from_li(li, len(products)+1, ts))
                if len(products) >= LIMIT:
                    break
            logging.info("📄 page %d – 수집 %d개", page, len(products))
            if len(products) >= LIMIT or page >= MAX_PAGES:
                break
            try:
                nxt = drv.find_element(By.CSS_SELECTOR, "a.btn_pn_next:not(.disabled)")
                drv.execute_script("arguments[0].click()", nxt)
            except:
                break
            page += 1
            time.sleep(1)
    finally:
        drv.quit()

    # Excel 저장
    df = pd.DataFrame([p.to_record() for p in products])
    fname = OUT_DIR / f"cj_{dt.datetime.now():%Y%m%d_%H%M}.xlsx"
    df.to_excel(fname, index=False)
    logging.info("[DONE] 총 %d개 저장 → %s", len(df), fname.resolve())

if __name__ == "__main__":
    crawl()
