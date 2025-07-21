# config/settings.py
"""설정 파일 - 최종 정리"""

import os
from datetime import datetime

# 기본 설정
DEFAULT_OUTPUT_DIR = '/Users/brich/Desktop/marketcrawler/review_analyzer/csv_output'
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

def get_user_paths():
    """터미널에서 사용자 입력 받기"""
    print("\n" + "="*50)
    print("📁 파일 경로 설정")
    print("="*50)
    
    # 입력 파일 경로
    while True:
        input_path = input("분석할 CSV 파일 경로를 입력하세요: ").strip()
        
        if not input_path:
            print("❌ 경로를 입력해주세요.")
            continue
        
        # 상대 경로를 절대 경로로 변환
        input_path = os.path.abspath(input_path)
        
        if not os.path.exists(input_path):
            print(f"❌ 파일을 찾을 수 없습니다: {input_path}")
            continue
        
        if not input_path.lower().endswith('.csv'):
            print("❌ CSV 파일만 지원됩니다.")
            continue
        
        print(f"✅ 입력 파일: {input_path}")
        break
    
    # 출력 디렉토리 경로
    output_path = input(f"결과 저장 디렉토리 (엔터=기본값 '{DEFAULT_OUTPUT_DIR}'): ").strip()
    
    if not output_path:
        output_path = DEFAULT_OUTPUT_DIR
    else:
        output_path = os.path.abspath(output_path)
    
    # 출력 디렉토리 생성
    output_path = ensure_output_dir(output_path)
    print(f"✅ 출력 디렉토리: {output_path}")
    
    return input_path, output_path

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