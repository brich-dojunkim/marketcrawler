"""
unified_crawler_exe.py - 하나의 엑셀로 통합하는 쇼핑몰 크롤러
Windows용 실행파일 빌드를 위한 완전한 통합 버전
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import sys
import os
import importlib.util
from pathlib import Path
import datetime
import json
import traceback
import pandas as pd
import platform
import tempfile
import shutil

class UnifiedCrawlerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("통합 쇼핑몰 크롤러 v1.0")
        self.root.geometry("700x650")
        
        # 시스템별 폰트 설정
        self.setup_fonts()
        
        # 기본 설정
        self.config = {
            "output_dir": str(Path.home() / "Desktop" / "쇼핑몰크롤링결과"),
            "count": 50,
            "sites": {
                "cjonstyle": {
                    "name": "CJ온스타일", 
                    "enabled": True, 
                    "url": "https://display.cjonstyle.com/p/category/categoryMain?dpCateId=G00011"
                },
                "gmarket": {
                    "name": "G마켓", 
                    "enabled": True, 
                    "url": "https://www.gmarket.co.kr/n/best?groupCode=100000001&subGroupCode=200000004"
                },
                "gsshop": {
                    "name": "GS샵", 
                    "enabled": True, 
                    "url": "https://www.gsshop.com/shop/sect/sectL.gs?sectid=1660575&eh=eyJwYWdlTnVtYmVyIjo3LCJzZWxlY3RlZCI6Im9wdC1wYWdlIiwibHNlY3RZbiI6IlkifQ%3D%3D"
                },
                "ssg": {
                    "name": "신세계몰", 
                    "enabled": True, 
                    "url": "https://www.ssg.com/disp/category.ssg?dispCtgId=6000188618&pageSize=100"
                }
            }
        }
        
        self.running = False
        self.all_results = []  # 모든 사이트 결과를 담을 리스트
        self.setup_ui()
        self.load_config()

    def setup_fonts(self):
        """시스템별 폰트 설정"""
        system = platform.system()
        if system == "Windows":
            self.font_family = "맑은 고딕"
        elif system == "Darwin":  # macOS
            self.font_family = "SF Pro Display"
        else:  # Linux
            self.font_family = "DejaVu Sans"
        
        # 기본 폰트 설정
        self.title_font = (self.font_family, 18, "bold")
        self.header_font = (self.font_family, 12, "bold")
        self.normal_font = (self.font_family, 10)
        self.small_font = (self.font_family, 9)

    def setup_ui(self):
        # 메인 프레임
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 제목
        title_label = ttk.Label(main_frame, text="통합 쇼핑몰 크롤러", font=self.title_font)
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 10))
        
        desc_label = ttk.Label(main_frame, text="여러 쇼핑몰의 상품 정보를 하나의 엑셀 파일로 수집합니다", 
                              font=self.small_font, foreground="gray")
        desc_label.grid(row=1, column=0, columnspan=2, pady=(0, 20))
        
        # 저장 폴더
        ttk.Label(main_frame, text="저장 폴더:", font=self.normal_font).grid(row=2, column=0, sticky=tk.W, pady=5)
        folder_frame = ttk.Frame(main_frame)
        folder_frame.grid(row=2, column=1, sticky=(tk.W, tk.E), padx=(10, 0))
        folder_frame.columnconfigure(0, weight=1)
        
        self.folder_var = tk.StringVar(value=self.config["output_dir"])
        folder_entry = ttk.Entry(folder_frame, textvariable=self.folder_var, width=45, font=self.small_font)
        folder_entry.grid(row=0, column=0, sticky=(tk.W, tk.E))
        ttk.Button(folder_frame, text="찾기", command=self.browse_folder).grid(row=0, column=1, padx=(5, 0))
        
        # 수집 개수
        ttk.Label(main_frame, text="각 사이트별 수집 개수:", font=self.normal_font).grid(row=3, column=0, sticky=tk.W, pady=5)
        count_frame = ttk.Frame(main_frame)
        count_frame.grid(row=3, column=1, sticky=tk.W, padx=(10, 0))
        
        self.count_var = tk.IntVar(value=self.config["count"])
        count_spin = ttk.Spinbox(count_frame, from_=10, to=200, textvariable=self.count_var, width=10, font=self.small_font)
        count_spin.pack(side=tk.LEFT)
        ttk.Label(count_frame, text="개 (권장: 50~100개)", font=self.small_font).pack(side=tk.LEFT, padx=(5, 0))
        
        # 사이트 선택
        ttk.Label(main_frame, text="수집할 쇼핑몰:", font=self.normal_font).grid(row=4, column=0, sticky=tk.W, pady=(15, 5))
        sites_frame = ttk.LabelFrame(main_frame, text="사이트 선택", padding="10")
        sites_frame.grid(row=4, column=1, sticky=(tk.W, tk.E), padx=(10, 0), pady=5)
        
        self.site_vars = {}
        for i, (site_id, site_config) in enumerate(self.config["sites"].items()):
            var = tk.BooleanVar(value=site_config["enabled"])
            self.site_vars[site_id] = var
            cb = ttk.Checkbutton(sites_frame, text=site_config["name"], variable=var, font=self.normal_font)
            cb.grid(row=i//2, column=i%2, sticky=tk.W, pady=3, padx=10)
        
        # 전체 선택/해제 버튼
        select_frame = ttk.Frame(sites_frame)
        select_frame.grid(row=2, column=0, columnspan=2, pady=(10, 0))
        ttk.Button(select_frame, text="전체선택", command=self.select_all, width=8).pack(side=tk.LEFT, padx=5)
        ttk.Button(select_frame, text="전체해제", command=self.deselect_all, width=8).pack(side=tk.LEFT, padx=5)
        
        # 버튼들
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=5, column=0, columnspan=2, pady=20)
        
        self.start_btn = ttk.Button(button_frame, text="🚀 크롤링 시작", command=self.start_crawling, width=15)
        self.start_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_btn = ttk.Button(button_frame, text="⏹️ 중지", command=self.stop_crawling, 
                                  state="disabled", width=10)
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(button_frame, text="📁 결과폴더 열기", command=self.open_folder, width=15).pack(side=tk.LEFT, padx=5)
        
        # 진행상황
        progress_frame = ttk.Frame(main_frame)
        progress_frame.grid(row=6, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        progress_frame.columnconfigure(1, weight=1)
        
        ttk.Label(progress_frame, text="상태:", font=self.small_font).grid(row=0, column=0, sticky=tk.W)
        self.status_var = tk.StringVar(value="대기 중")
        status_label = ttk.Label(progress_frame, textvariable=self.status_var, font=(self.font_family, 9, "bold"))
        status_label.grid(row=0, column=1, sticky=tk.W, padx=(5, 0))
        
        # 진행 바
        self.progress = ttk.Progressbar(progress_frame, mode='determinate')
        self.progress.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(5, 0))
        
        # 로그
        ttk.Label(main_frame, text="진행 로그:", font=self.normal_font).grid(row=7, column=0, columnspan=2, sticky=tk.W, pady=(15, 5))
        self.log_text = scrolledtext.ScrolledText(main_frame, height=15, width=80, font=self.small_font)
        self.log_text.grid(row=8, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 그리드 설정
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(8, weight=1)

    def select_all(self):
        for var in self.site_vars.values():
            var.set(True)

    def deselect_all(self):
        for var in self.site_vars.values():
            var.set(False)

    def browse_folder(self):
        folder = filedialog.askdirectory(title="결과 저장 폴더 선택")
        if folder:
            self.folder_var.set(folder)

    def open_folder(self):
        folder = self.folder_var.get()
        if os.path.exists(folder):
            if platform.system() == "Windows":
                os.startfile(folder)
            elif platform.system() == "Darwin":  # macOS
                os.system(f"open '{folder}'")
            else:  # Linux
                os.system(f"xdg-open '{folder}'")
        else:
            messagebox.showinfo("알림", "폴더가 존재하지 않습니다.")

    def log(self, msg, level="INFO"):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        if level == "ERROR":
            prefix = "❌"
        elif level == "SUCCESS":
            prefix = "✅"
        elif level == "WARNING":
            prefix = "⚠️"
        else:
            prefix = "ℹ️"
        
        log_msg = f"[{timestamp}] {prefix} {msg}\n"
        self.log_text.insert(tk.END, log_msg)
        self.log_text.see(tk.END)
        self.root.update_idletasks()

    def load_config(self):
        try:
            config_file = "crawler_config.json"
            if os.path.exists(config_file):
                with open(config_file, "r", encoding="utf-8") as f:
                    saved = json.load(f)
                    self.config.update(saved)
                    self.update_ui()
        except Exception as e:
            self.log(f"설정 로드 실패: {e}", "WARNING")

    def save_config(self):
        try:
            config_file = "crawler_config.json"
            with open(config_file, "w", encoding="utf-8") as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.log(f"설정 저장 실패: {e}", "WARNING")

    def update_ui(self):
        self.folder_var.set(self.config["output_dir"])
        self.count_var.set(self.config["count"])
        for site_id, var in self.site_vars.items():
            if site_id in self.config["sites"]:
                var.set(self.config["sites"][site_id]["enabled"])

    def start_crawling(self):
        # 설정 업데이트
        self.config["output_dir"] = self.folder_var.get()
        self.config["count"] = self.count_var.get()
        for site_id, var in self.site_vars.items():
            self.config["sites"][site_id]["enabled"] = var.get()
        
        # 검증
        enabled_sites = [s for s in self.config["sites"].values() if s["enabled"]]
        if not enabled_sites:
            messagebox.showwarning("경고", "최소 하나의 쇼핑몰을 선택하세요.")
            return
        
        if not (10 <= self.config["count"] <= 200):
            messagebox.showwarning("경고", "수집 개수는 10~200개 사이로 설정하세요.")
            return
        
        # 폴더 생성
        try:
            Path(self.config["output_dir"]).mkdir(parents=True, exist_ok=True)
        except Exception as e:
            messagebox.showerror("오류", f"폴더 생성 실패: {e}")
            return
        
        self.save_config()
        self.running = True
        self.all_results = []  # 결과 초기화
        
        # UI 상태 변경
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.progress["maximum"] = len(enabled_sites)
        self.progress["value"] = 0
        
        # 로그 클리어
        self.log_text.delete(1.0, tk.END)
        self.log("크롤링 작업을 시작합니다...")
        
        # 백그라운드 실행
        threading.Thread(target=self.crawl_worker, daemon=True).start()

    def stop_crawling(self):
        self.running = False
        self.status_var.set("중지 중...")
        self.log("중지 요청됨", "WARNING")

    def crawl_worker(self):
        enabled_sites = [(id, cfg) for id, cfg in self.config["sites"].items() if cfg["enabled"]]
        success = 0
        
        self.log(f"{len(enabled_sites)}개 사이트에서 각각 {self.config['count']}개씩 수집 시작")
        
        for i, (site_id, site_config) in enumerate(enabled_sites, 1):
            if not self.running:
                break
            
            self.status_var.set(f"{site_config['name']} 수집 중... ({i}/{len(enabled_sites)})")
            self.log(f"{site_config['name']} 크롤링 시작")
            
            try:
                result_data = self.run_crawler(site_id, site_config)
                if result_data:
                    self.all_results.extend(result_data)  # 결과 합치기
                    success += 1
                    self.log(f"{site_config['name']} 완료 - {len(result_data)}개 수집", "SUCCESS")
                else:
                    self.log(f"{site_config['name']} 실패 - 데이터 없음", "ERROR")
            except Exception as e:
                self.log(f"{site_config['name']} 오류: {str(e)}", "ERROR")
                self.log(f"상세 오류: {traceback.format_exc()}")
            
            # 진행률 업데이트
            self.progress["value"] = i
            self.root.update_idletasks()
        
        # 최종 결과 저장
        if self.running and self.all_results:
            self.save_unified_excel()
            
        # 완료 처리
        if self.running:
            self.status_var.set(f"완료! ({success}/{len(enabled_sites)}개 성공)")
            self.log(f"전체 작업 완료: {success}/{len(enabled_sites)}개 성공", "SUCCESS")
            self.log(f"총 {len(self.all_results)}개 상품 데이터 수집")
            if success > 0:
                messagebox.showinfo("완료", 
                                  f"크롤링 완료!\n성공: {success}/{len(enabled_sites)}개 사이트\n총 {len(self.all_results)}개 상품")
        
        # UI 복원
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.progress["value"] = 0
        self.running = False

    def save_unified_excel(self):
        """모든 결과를 하나의 엑셀 파일로 저장"""
        try:
            if not self.all_results:
                self.log("저장할 데이터가 없습니다", "WARNING")
                return
            
            # DataFrame 생성
            df = pd.DataFrame(self.all_results)
            
            # 컬럼 순서 통일
            columns = ["순위", "쇼핑몰", "상품명", "브랜드명", "상품ID", "정가", "최종가", "할인율",
                      "가격내역", "혜택", "리뷰/구매수", "평점", "리뷰수", "판매자명", 
                      "카테고리코드", "카테고리라벨", "노출코드", "프로모션태그", 
                      "수집시각", "상품URL"]
            
            # 누락된 컬럼 추가
            for col in columns:
                if col not in df.columns:
                    df[col] = ""
            
            # 컬럼 순서 적용
            df = df[columns]
            
            # 파일명 생성
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"통합쇼핑몰크롤링_{timestamp}.xlsx"
            filepath = Path(self.config["output_dir"]) / filename
            
            # 엑셀 저장 (여러 시트로)
            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                # 전체 데이터
                df.to_excel(writer, sheet_name='전체데이터', index=False)
                
                # 사이트별 시트
                for site in df['쇼핑몰'].unique():
                    if site:
                        site_df = df[df['쇼핑몰'] == site]
                        sheet_name = site.replace('/', '_')  # 시트명 특수문자 제거
                        site_df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            self.log(f"통합 엑셀 파일 저장 완료: {filepath.name}", "SUCCESS")
            self.log(f"저장 위치: {filepath}")
            
        except Exception as e:
            self.log(f"엑셀 저장 중 오류: {e}", "ERROR")
            self.log(f"상세 오류: {traceback.format_exc()}")

    def load_module(self, name):
        """동적으로 모듈 로드"""
        try:
            # 실행파일인 경우 임시 디렉토리에서 파일 찾기
            if getattr(sys, 'frozen', False):
                # PyInstaller로 패키징된 경우
                bundle_dir = sys._MEIPASS
                module_path = os.path.join(bundle_dir, f"{name}.py")
            else:
                # 일반 Python 실행
                current_dir = os.path.dirname(os.path.abspath(__file__))
                module_path = os.path.join(current_dir, f"{name}.py")
            
            if not os.path.exists(module_path):
                self.log(f"{name}.py 파일이 존재하지 않습니다: {module_path}", "ERROR")
                return None
            
            # 모듈 이름이 이미 있으면 제거
            if name in sys.modules:
                del sys.modules[name]
            
            # 동적 import
            spec = importlib.util.spec_from_file_location(name, module_path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            
            return module
            
        except Exception as e:
            self.log(f"{name} 모듈 로드 실패: {e}", "ERROR")
            return None

    def run_crawler(self, site_id, site_config):
        """각 사이트별 크롤러 실행하고 통합 형태로 데이터 반환"""
        if site_id == "cjonstyle":
            return self.run_cj(site_config)
        elif site_id == "gmarket":
            return self.run_gm(site_config)
        elif site_id == "gsshop":
            return self.run_gs(site_config)
        elif site_id == "ssg":
            return self.run_ssg(site_config)
        return None

    def standardize_data(self, data_list, site_name):
        """각 사이트 데이터를 통합 형식으로 표준화"""
        standardized = []
        for i, item in enumerate(data_list, 1):
            std_item = {
                "순위": i,
                "쇼핑몰": site_name,
                "상품명": item.get("상품명", ""),
                "브랜드명": item.get("브랜드명", item.get("브랜드", item.get("판매자명", ""))),
                "상품ID": item.get("상품ID", ""),
                "정가": item.get("정가"),
                "최종가": item.get("최종가"),
                "할인율": item.get("할인율"),
                "가격내역": item.get("가격내역", ""),
                "혜택": item.get("혜택", item.get("프로모션태그", "")),
                "리뷰/구매수": item.get("리뷰/구매수"),
                "평점": item.get("평점"),
                "리뷰수": item.get("리뷰수"),
                "판매자명": item.get("판매자명", ""),
                "카테고리코드": item.get("카테고리코드", ""),
                "카테고리라벨": item.get("카테고리라벨", ""),
                "노출코드": item.get("노출코드", ""),
                "프로모션태그": item.get("프로모션태그", ""),
                "수집시각": item.get("수집시각", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                "상품URL": item.get("상품URL", "")
            }
            standardized.append(std_item)
        return standardized

    def run_cj(self, site_config):
        """CJ온스타일 크롤러 실행"""
        module = self.load_module("cjonstyle")
        if not module:
            return None
        
        try:
            # 임시 결과 폴더 생성
            temp_dir = tempfile.mkdtemp()
            
            # 원본 설정 백업
            original_settings = {}
            for attr in ['LIMIT', 'HEADLESS', 'OUT_DIR', 'TARGET_URL']:
                if hasattr(module, attr):
                    original_settings[attr] = getattr(module, attr)
            
            # 설정 변경
            module.LIMIT = self.config["count"]
            module.HEADLESS = True
            module.OUT_DIR = Path(temp_dir)
            if hasattr(module, 'TARGET_URL'):
                module.TARGET_URL = site_config["url"]
            
            # 크롤링 실행
            module.crawl()
            
            # 생성된 엑셀 파일 찾기
            excel_files = list(Path(temp_dir).glob("*.xlsx"))
            if excel_files:
                df = pd.read_excel(excel_files[0])
                data = df.to_dict('records')
                result = self.standardize_data(data, "CJ온스타일")
                
                # 임시 파일 정리
                shutil.rmtree(temp_dir, ignore_errors=True)
                
                # 원본 설정 복원
                for attr, value in original_settings.items():
                    setattr(module, attr, value)
                
                return result
            
            # 임시 파일 정리
            shutil.rmtree(temp_dir, ignore_errors=True)
            
        except Exception as e:
            self.log(f"CJ온스타일 실행 오류: {e}", "ERROR")
            return None

    def run_gm(self, site_config):
        """G마켓 크롤러 실행"""
        module = self.load_module("gmarket")
        if not module:
            return None
        
        try:
            # 임시 결과 폴더 생성
            temp_dir = tempfile.mkdtemp()
            
            # 크롤링 실행
            result_path = module.crawl(
                list_url=site_config["url"],
                top_n=self.config["count"],
                headless=True,
                out_dir=temp_dir
            )
            
            # 결과 파일 읽기
            if result_path and os.path.exists(result_path):
                df = pd.read_excel(result_path)
                data = df.to_dict('records')
                result = self.standardize_data(data, "G마켓")
                
                # 임시 파일 정리
                shutil.rmtree(temp_dir, ignore_errors=True)
                return result
            
            # 임시 파일 정리
            shutil.rmtree(temp_dir, ignore_errors=True)
            
        except Exception as e:
            self.log(f"G마켓 실행 오류: {e}", "ERROR")
            return None

    def run_gs(self, site_config):
        """GS샵 크롤러 실행"""
        module = self.load_module("gsshop")
        if not module:
            return None
        
        try:
            # 임시 결과 폴더 생성
            temp_dir = tempfile.mkdtemp()
            
            # 크롤링 실행
            module.crawl(
                url=site_config["url"],
                top=self.config["count"],
                headless=True,
                out_dir=temp_dir,
                stall_max=5,
                snap_dir=None
            )
            
            # 생성된 엑셀 파일 찾기
            excel_files = list(Path(temp_dir).glob("*.xlsx"))
            if excel_files:
                df = pd.read_excel(excel_files[0])
                data = df.to_dict('records')
                result = self.standardize_data(data, "GS샵")
                
                # 임시 파일 정리
                shutil.rmtree(temp_dir, ignore_errors=True)
                return result
            
            # 임시 파일 정리
            shutil.rmtree(temp_dir, ignore_errors=True)
            
        except Exception as e:
            self.log(f"GS샵 실행 오류: {e}", "ERROR")
            return None

    def run_ssg(self, site_config):
        """SSG 크롤러 실행"""
        module = self.load_module("ssg")
        if not module:
            return None
        
        try:
            # 임시 결과 폴더 생성
            temp_dir = tempfile.mkdtemp()
            
            # 크롤링 실행
            result_path = module.crawl_ssg(
                list_url=site_config["url"],
                top_n=self.config["count"],
                headless=True,
                out_dir=temp_dir
            )
            
            # 결과 파일 읽기
            if result_path and os.path.exists(result_path):
                df = pd.read_excel(result_path)
                data = df.to_dict('records')
                result = self.standardize_data(data, "신세계몰")
                
                # 임시 파일 정리
                shutil.rmtree(temp_dir, ignore_errors=True)
                return result
            
            # 임시 파일 정리
            shutil.rmtree(temp_dir, ignore_errors=True)
            
        except Exception as e:
            self.log(f"신세계몰 실행 오류: {e}", "ERROR")
            return None

def main():
    # 필요한 파일 확인
    required = ["cjonstyle.py", "gmarket.py", "gsshop.py", "ssg.py"]
    missing = []
    
    for file in required:
        # 실행파일인 경우와 일반 실행인 경우 모두 고려
        if getattr(sys, 'frozen', False):
            # PyInstaller로 패키징된 경우
            file_path = os.path.join(sys._MEIPASS, file)
        else:
            # 일반 Python 실행
            file_path = file
            
        if not os.path.exists(file_path):
            missing.append(file)
    
    if missing:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("파일 부족", 
                           f"다음 파일들이 필요합니다:\n{chr(10).join(missing)}\n\n"
                           f"실행 파일과 같은 폴더에 위치시켜 주세요.")
        return
    
    try:
        root = tk.Tk()
        
        # Windows 아이콘 설정 (옵션)
        try:
            if platform.system() == "Windows":
                root.iconbitmap('icon.ico')  # icon.ico 파일이 있다면
        except:
            pass
        
        app = UnifiedCrawlerGUI(root)
        root.mainloop()
        
    except Exception as e:
        messagebox.showerror("실행 오류", f"프로그램 실행 중 오류가 발생했습니다:\n{e}")

if __name__ == "__main__":
    main()