@echo off
echo 실행파일을 생성합니다...
pip install pyinstaller
pyinstaller --onefile --windowed --name "쇼핑몰크롤러" unified_crawler.py
echo 완료! dist 폴더의 쇼핑몰크롤러.exe 파일을 사용하세요.
pause
