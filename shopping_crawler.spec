# shopping_crawler.spec
# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

# 추가할 데이터 파일들
added_files = [
    ('cjonstyle.py', '.'),
    ('gmarket.py', '.'),
    ('gsshop.py', '.'),
    ('ssg.py', '.'),
]

# 숨겨진 imports (동적으로 로드되는 모듈들)
hidden_imports = [
    'selenium',
    'selenium.webdriver',
    'selenium.webdriver.common.by',
    'selenium.webdriver.support.ui',
    'selenium.webdriver.support.expected_conditions',
    'selenium.webdriver.chrome.service',
    'selenium.webdriver.chrome.options',
    'undetected_chromedriver',
    'selenium_stealth',
    'fake_useragent',
    'bs4',
    'pandas',
    'openpyxl',
    'pytz',
    'certifi',
    'urllib3',
    'requests',
    'json',
    'threading',
    'tkinter',
    'tkinter.ttk',
    'tkinter.filedialog',
    'tkinter.messagebox',
    'tkinter.scrolledtext',
    'importlib.util',
    'pathlib',
    'datetime',
    'traceback'
]

a = Analysis(
    ['unified_crawler.py'],
    pathex=[],
    binaries=[],
    datas=added_files,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='쇼핑몰크롤러',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # GUI 앱이므로 콘솔 창 숨김
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='icon.ico' if __import__('os').path.exists('icon.ico') else None,
)