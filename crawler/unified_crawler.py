"""
unified_crawler.py - 간단한 통합 쇼핑몰 크롤러
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

class CrawlerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("쇼핑몰 크롤러")
        self.root.geometry("600x500")
        
        # 기본 설정
        self.config = {
            "output_dir": str(Path.home() / "Desktop" / "크롤링결과"),
            "count": 50,
            "sites": {
                "cjonstyle": {"name": "CJ온스타일", "enabled": True, "url": "https://display.cjonstyle.com/p/category/categoryMain?dpCateId=G00011"},
                "gmarket": {"name": "G마켓", "enabled": True, "url": "https://www.gmarket.co.kr/n/best?groupCode=100000001&subGroupCode=200000004"},
                "gsshop": {"name": "GS샵", "enabled": True, "url": "https://www.gsshop.com/shop/sect/sectL.gs?sectid=1660575&eh=eyJwYWdlTnVtYmVyIjo3LCJzZWxlY3RlZCI6Im9wdC1wYWdlIiwibHNlY3RZbiI6IlkifQ%3D%3D"},
                "ssg": {"name": "신세계몰", "enabled": True, "url": "https://www.ssg.com/disp/category.ssg?dispCtgId=6000188618&pageSize=100"}
            }
        }
        
        self.running = False
        self.setup_ui()
        self.load_config()

    def setup_ui(self):
        # 메인 프레임
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 제목
        title_label = ttk.Label(main_frame, text="쇼핑몰 크롤러", font=("", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # 저장 폴더
        ttk.Label(main_frame, text="저장 폴더:").grid(row=1, column=0, sticky=tk.W, pady=5)
        folder_frame = ttk.Frame(main_frame)
        folder_frame.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=(10, 0))
        folder_frame.columnconfigure(0, weight=1)
        
        self.folder_var = tk.StringVar(value=self.config["output_dir"])
        ttk.Entry(folder_frame, textvariable=self.folder_var, width=40).grid(row=0, column=0, sticky=(tk.W, tk.E))
        ttk.Button(folder_frame, text="찾기", command=self.browse_folder).grid(row=0, column=1, padx=(5, 0))
        
        # 수집 개수
        ttk.Label(main_frame, text="각 사이트별 수집 개수:").grid(row=2, column=0, sticky=tk.W, pady=5)
        count_frame = ttk.Frame(main_frame)
        count_frame.grid(row=2, column=1, sticky=tk.W, padx=(10, 0))
        
        self.count_var = tk.IntVar(value=self.config["count"])
        ttk.Spinbox(count_frame, from_=10, to=300, textvariable=self.count_var, width=10).pack(side=tk.LEFT)
        ttk.Label(count_frame, text="개").pack(side=tk.LEFT, padx=(5, 0))
        
        # 사이트 선택
        ttk.Label(main_frame, text="수집할 쇼핑몰:").grid(row=3, column=0, sticky=tk.W, pady=(15, 5))
        sites_frame = ttk.Frame(main_frame)
        sites_frame.grid(row=3, column=1, sticky=(tk.W, tk.E), padx=(10, 0))
        
        self.site_vars = {}
        for i, (site_id, site_config) in enumerate(self.config["sites"].items()):
            var = tk.BooleanVar(value=site_config["enabled"])
            self.site_vars[site_id] = var
            ttk.Checkbutton(sites_frame, text=site_config["name"], variable=var).grid(row=i//2, column=i%2, sticky=tk.W, pady=2)
        
        # 버튼들
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=4, column=0, columnspan=2, pady=20)
        
        self.start_btn = ttk.Button(button_frame, text="크롤링 시작", command=self.start_crawling, 
                                   style="Accent.TButton")
        self.start_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_btn = ttk.Button(button_frame, text="중지", command=self.stop_crawling, state="disabled")
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(button_frame, text="결과폴더 열기", command=self.open_folder).pack(side=tk.LEFT, padx=5)
        
        # 진행상황
        self.status_var = tk.StringVar(value="대기 중")
        ttk.Label(main_frame, textvariable=self.status_var, font=("", 10)).grid(row=5, column=0, columnspan=2, pady=5)
        
        # 로그
        ttk.Label(main_frame, text="진행 로그:").grid(row=6, column=0, columnspan=2, sticky=tk.W, pady=(10, 5))
        self.log_text = scrolledtext.ScrolledText(main_frame, height=12, width=70)
        self.log_text.grid(row=7, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 그리드 설정
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(7, weight=1)

    def browse_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.folder_var.set(folder)

    def open_folder(self):
        folder = self.folder_var.get()
        if os.path.exists(folder):
            os.startfile(folder)
        else:
            messagebox.showinfo("알림", "폴더가 없습니다.")

    def log(self, msg):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {msg}\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()

    def load_config(self):
        try:
            if os.path.exists("config.json"):
                with open("config.json", "r", encoding="utf-8") as f:
                    saved = json.load(f)
                    self.config.update(saved)
                    self.update_ui()
        except:
            pass

    def save_config(self):
        try:
            with open("config.json", "w", encoding="utf-8") as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
        except:
            pass

    def update_ui(self):
        self.folder_var.set(self.config["output_dir"])
        self.count_var.set(self.config["count"])
        for site_id, var in self.site_vars.items():
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
        
        if not (10 <= self.config["count"] <= 300):
            messagebox.showwarning("경고", "수집 개수는 10~300개 사이로 설정하세요.")
            return
        
        # 폴더 생성
        try:
            Path(self.config["output_dir"]).mkdir(parents=True, exist_ok=True)
        except Exception as e:
            messagebox.showerror("오류", f"폴더 생성 실패: {e}")
            return
        
        self.save_config()
        self.running = True
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        
        threading.Thread(target=self.crawl_worker, daemon=True).start()

    def stop_crawling(self):
        self.running = False
        self.status_var.set("중지 중...")
        self.log("중지 요청됨")

    def crawl_worker(self):
        enabled_sites = [(id, cfg) for id, cfg in self.config["sites"].items() if cfg["enabled"]]
        success = 0
        
        self.log(f"{len(enabled_sites)}개 사이트에서 각각 {self.config['count']}개씩 수집 시작")
        
        for i, (site_id, site_config) in enumerate(enabled_sites, 1):
            if not self.running:
                break
            
            self.status_var.set(f"{site_config['name']} 수집 중... ({i}/{len(enabled_sites)})")
            self.log(f"{site_config['name']} 시작")
            
            try:
                if self.run_crawler(site_id, site_config):
                    success += 1
                    self.log(f"{site_config['name']} 완료")
                else:
                    self.log(f"{site_config['name']} 실패")
            except Exception as e:
                self.log(f"{site_config['name']} 오류: {e}")
        
        if self.running:
            self.status_var.set(f"완료! ({success}/{len(enabled_sites)}개 성공)")
            self.log(f"전체 완료: {success}/{len(enabled_sites)}개 성공")
            if success > 0:
                messagebox.showinfo("완료", f"크롤링 완료!\n성공: {success}/{len(enabled_sites)}개")
        
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.running = False

    def load_module(self, name):
        try:
            # 현재 디렉토리를 Python 경로에 추가
            current_dir = os.path.dirname(os.path.abspath(__file__))
            if current_dir not in sys.path:
                sys.path.insert(0, current_dir)
            
            # 파일 존재 확인
            path = f"{name}.py"
            if not os.path.exists(path):
                self.log(f"{name}.py 파일이 존재하지 않습니다")
                return None
            
            # 이미 import된 모듈이 있다면 제거 (재로드를 위해)
            if name in sys.modules:
                del sys.modules[name]
            
            # 동적 import
            module = __import__(name)
            self.log(f"{name} 모듈 로드 성공")
            return module
            
        except ImportError as e:
            self.log(f"{name} import 오류: {e}")
            self.log("필요한 라이브러리가 설치되지 않았을 수 있습니다")
            return None
        except SyntaxError as e:
            self.log(f"{name} 문법 오류: {e}")
            return None
        except Exception as e:
            self.log(f"{name} 로드 중 오류: {e}")
            import traceback
            self.log(f"상세 오류:\n{traceback.format_exc()}")
            return None

    def run_crawler(self, site_id, site_config):
        output_dir = Path(self.config["output_dir"])
        
        if site_id == "cjonstyle":
            return self.run_cj(site_config, output_dir)
        elif site_id == "gmarket":
            return self.run_gm(site_config, output_dir)
        elif site_id == "gsshop":
            return self.run_gs(site_config, output_dir)
        elif site_id == "ssg":
            return self.run_ssg(site_config, output_dir)
        return False

    def run_cj(self, site_config, output_dir):
        module = self.load_module("cjonstyle")
        if not module:
            return False
        
        # 설정 백업
        orig = {
            'LIMIT': getattr(module, 'LIMIT', 100),
            'HEADLESS': getattr(module, 'HEADLESS', True),
            'OUT_DIR': getattr(module, 'OUT_DIR', Path("output")),
            'TARGET_URL': getattr(module, 'TARGET_URL', "")
        }
        
        # 설정 적용
        module.LIMIT = self.config["count"]
        module.HEADLESS = True
        module.OUT_DIR = output_dir
        module.TARGET_URL = site_config["url"]
        
        try:
            module.crawl()
            return True
        finally:
            # 복원
            for k, v in orig.items():
                setattr(module, k, v)

    def run_gm(self, site_config, output_dir):
        module = self.load_module("gmarket")
        if not module:
            return False
        
        module.crawl(
            list_url=site_config["url"],
            top_n=self.config["count"],
            headless=True,
            out_dir=str(output_dir)
        )
        return True

    def run_gs(self, site_config, output_dir):
        module = self.load_module("gsshop")
        if not module:
            return False
        
        module.crawl(
            url=site_config["url"],
            top=self.config["count"],
            headless=True,
            out_dir=str(output_dir),
            stall_max=5,
            snap_dir=None
        )
        return True

    def run_ssg(self, site_config, output_dir):
        module = self.load_module("ssg")
        if not module:
            return False
        
        module.crawl_ssg(
            list_url=site_config["url"],
            top_n=self.config["count"],
            headless=True,
            out_dir=str(output_dir)
        )
        return True

def main():
    required = ["cjonstyle.py", "gmarket.py", "gsshop.py", "ssg.py"]
    missing = [f for f in required if not os.path.exists(f)]
    
    if missing:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("파일 부족", f"필요한 파일: {', '.join(missing)}")
        return
    
    root = tk.Tk()
    app = CrawlerGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()