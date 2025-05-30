#!/bin/bash

echo "윈도우용 배포 패키지 생성 중..."

# 배포용 폴더 생성
mkdir -p windows_dist
cp *.py windows_dist/

# 윈도우용 실행 배치파일 생성
cat > windows_dist/run.bat << 'EOF'
@echo off
echo 쇼핑몰 크롤러를 시작합니다...
python unified_crawler.py
if errorlevel 1 (
    echo Python이 설치되지 않았거나 라이브러리가 없습니다.
    echo 다음 명령어를 실행하세요:
    echo pip install selenium undetected-chromedriver selenium-stealth fake-useragent beautifulsoup4 pandas openpyxl pytz tkinter
    pause
)
EOF

# 윈도우용 설치 스크립트 생성
cat > windows_dist/install.bat << 'EOF'
@echo off
echo 필요한 라이브러리를 설치합니다...
pip install selenium undetected-chromedriver selenium-stealth fake-useragent beautifulsoup4 pandas openpyxl pytz
echo 설치 완료!
echo run.bat 파일을 실행하세요.
pause
EOF

# 실행파일 생성 스크립트 (윈도우에서 실행)
cat > windows_dist/make_exe.bat << 'EOF'
@echo off
echo 실행파일을 생성합니다...
pip install pyinstaller
pyinstaller --onefile --windowed --name "쇼핑몰크롤러" unified_crawler.py
echo 완료! dist 폴더의 쇼핑몰크롤러.exe 파일을 사용하세요.
pause
EOF

# README 생성
cat > windows_dist/README.txt << 'EOF'
쇼핑몰 크롤러 - 윈도우용

1. 첫 실행시: install.bat 더블클릭 (라이브러리 설치)
2. 프로그램 실행: run.bat 더블클릭
3. 실행파일 만들기: make_exe.bat 더블클릭

문제가 있으면 Python 3.7 이상을 먼저 설치하세요.
EOF

echo "완료! windows_dist 폴더를 윈도우 사용자에게 전달하세요."
echo "윈도우에서 install.bat → run.bat 순서로 실행하면 됩니다."