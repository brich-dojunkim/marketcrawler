@echo off
chcp 65001
echo ====================================
echo    쇼핑몰 크롤러 빌드 스크립트
echo ====================================
echo.

echo [1/4] Python 설치 확인...
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python이 설치되지 않았습니다!
    echo Python 3.8+ 을 설치한 후 다시 실행하세요.
    pause
    exit /b 1
)
echo ✅ Python 설치 확인됨

echo.
echo [2/4] 필요한 패키지 설치...
pip install --upgrade pip
pip install pyinstaller
pip install -r requirements.txt
if errorlevel 1 (
    echo ❌ 패키지 설치 실패!
    pause
    exit /b 1
)
echo ✅ 패키지 설치 완료

echo.
echo [3/4] 실행 파일 빌드 시작...
if exist "shopping_crawler.spec" (
    echo spec 파일 사용하여 빌드...
    pyinstaller shopping_crawler.spec --clean
) else (
    echo 기본 설정으로 빌드...
    pyinstaller --onefile --windowed --name "쇼핑몰크롤러" ^
      --add-data "cjonstyle.py;." ^
      --add-data "gmarket.py;." ^
      --add-data "gsshop.py;." ^
      --add-data "ssg.py;." ^
      --hidden-import "selenium" ^
      --hidden-import "undetected_chromedriver" ^
      --hidden-import "selenium_stealth" ^
      --hidden-import "fake_useragent" ^
      --hidden-import "bs4" ^
      --hidden-import "pandas" ^
      --hidden-import "openpyxl" ^
      unified_crawler.py
)

if errorlevel 1 (
    echo ❌ 빌드 실패!
    pause
    exit /b 1
)
echo ✅ 빌드 완료

echo.
echo [4/4] 파일 확인...
if exist "dist\쇼핑몰크롤러.exe" (
    echo ✅ 실행 파일 생성 성공!
    echo 📂 위치: %CD%\dist\쇼핑몰크롤러.exe
    echo 📏 크기: 
    dir "dist\쇼핑몰크롤러.exe" | findstr "쇼핑몰크롤러.exe"
    echo.
    echo ====================================
    echo          빌드 완료!
    echo ====================================
    echo dist 폴더의 실행 파일을 배포하세요.
    echo.
    choice /c YN /m "dist 폴더를 여시겠습니까? (Y/N)"
    if errorlevel 2 goto :end
    if errorlevel 1 explorer dist
) else (
    echo ❌ 실행 파일을 찾을 수 없습니다!
)

:end
pause