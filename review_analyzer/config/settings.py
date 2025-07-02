# config/settings.py
"""설정 파일 - 하드코딩 최소화"""

import os
from datetime import datetime

# 기본 설정
DEFAULT_OUTPUT_DIR = '/Users/brich/Desktop/marketcrawler/output'
DEFAULT_DATA_FILE = 'coupang_reviews_20250701_180647.csv'

# 라이브러리 가용성 체크
def check_library_availability():
    """라이브러리 사용 가능 여부 확인"""
    availability = {
        'konlpy': False,
        'pykospacing': False,
        'transformers': False
    }
    
    # KoNLPy 체크
    try:
        from konlpy.tag import Okt
        availability['konlpy'] = True
        print("✅ KoNLPy 사용 가능")
    except Exception as e:
        print(f"❌ KoNLPy 사용 불가: {e}")
    
    # PyKoSpacing 체크
    try:
        from pykospacing import Spacing
        availability['pykospacing'] = True
        print("✅ PyKoSpacing 사용 가능")
    except ImportError as e:
        print(f"❌ PyKoSpacing 사용 불가: {e}")
    
    # Transformers 체크
    try:
        import torch
        from transformers import BertTokenizer
        availability['transformers'] = True
        print("✅ PyTorch/Transformers 사용 가능")
    except ImportError:
        print("❌ PyTorch/Transformers를 설치하지 못했습니다. 규칙 기반 감성 분석을 사용합니다.")
    
    return availability

# 한글 폰트 경로 (macOS 기준)
FONT_PATHS = [
    '/System/Library/Fonts/NanumGothic.ttc',
    '/Library/Fonts/NanumGothic.ttf',
    '/System/Library/Fonts/Arial Unicode MS.ttf',
    '/System/Library/Fonts/AppleGothic.ttf'
]

def get_font_path():
    """사용 가능한 한글 폰트 경로 반환"""
    for path in FONT_PATHS:
        if os.path.exists(path):
            return path
    return None

# 분석 매개변수 - 데이터 기반 접근을 위한 설정
ANALYSIS_PARAMS = {
    'max_tfidf_features': 100,
    'ngram_range': (1, 3),  # 확장: 3-gram까지
    'min_word_count': 3,    # 감소: 더 많은 키워드 포함
    'max_word_length': 15,  # 증가: 복합어 포함
    'n_topics': 5,
    'auto_stopword_threshold': 0.7,  # 새로운: 자동 불용어 임계값
    'min_phrase_freq': 3,            # 새로운: 최소 구문 빈도
    'cluster_method': 'kmeans'       # 새로운: 클러스터링 방법
}

def get_output_filename(prefix="analysis_report"):
    """출력 파일명 생성"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}.xlsx"

def ensure_output_dir(output_dir=None):
    """출력 디렉토리 확인 및 생성"""
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    return output_dir