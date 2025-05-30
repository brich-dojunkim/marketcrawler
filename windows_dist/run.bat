@echo off
echo 쇼핑몰 크롤러를 시작합니다...
python unified_crawler.py
if errorlevel 1 (
    echo Python이 설치되지 않았거나 라이브러리가 없습니다.
    echo 다음 명령어를 실행하세요:
    echo pip install selenium undetected-chromedriver selenium-stealth fake-useragent beautifulsoup4 pandas openpyxl pytz tkinter
    pause
)
