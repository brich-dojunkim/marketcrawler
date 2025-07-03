# config/settings.py
"""설정 파일 - 최종 정리"""

import os
from datetime import datetime

# 기본 설정
DEFAULT_OUTPUT_DIR = '/Users/brich/Desktop/marketcrawler/output'
DEFAULT_DATA_FILE = 'coupang_reviews_20250701_184246.csv'

# 한글 폰트 경로 (시각화용)
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

# 분석 매개변수
ANALYSIS_PARAMS = {
    'max_tfidf_features': 100,
    'ngram_range': (1, 3),
    'min_word_count': 3,
    'max_word_length': 15,
    'n_topics': 5,
    'auto_stopword_threshold': 0.5,
    'min_phrase_freq': 3,
    'cluster_method': 'kmeans'
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