# config/settings.py
"""설정 파일"""

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
        print("Java가 필요합니다. 설치 가이드를 확인하세요.")
    
    # PyKoSpacing 체크
    try:
        from pykospacing import Spacing
        availability['pykospacing'] = True
        print("✅ PyKoSpacing 사용 가능")
    except ImportError as e:
        print(f"❌ PyKoSpacing 사용 불가: {e}")
        print("TensorFlow가 필요합니다. 'pip install tensorflow' 설치 후 재시도하세요.")
    
    # Transformers 체크
    try:
        import torch
        from transformers import BertTokenizer
        availability['transformers'] = True
        print("✅ PyTorch/Transformers 사용 가능")
    except ImportError:
        print("❌ PyTorch/Transformers를 설치하지 못했습니다. 규칙 기반 감성 분석을 사용합니다.")
    
    return availability

# 불용어 리스트
STOPWORDS = [
    '은', '는', '이', '가', '을', '를', '의', '에', '에서', '로', '으로',
    '과', '와', '도', '만', '부터', '까지', '뿐', '아니라', '처럼', '같이',
    '그', '저', '이', '그것', '저것', '것', '수', '곳', '때', '점',
    '정말', '너무', '진짜', '완전', '매우', '아주', '좀', '조금'
]

# 감성사전
SENTIMENT_DICT = {
    'positive': [
        '좋다', '훌륭하다', '만족', '추천', '최고', '완벽', '우수', '탁월',
        '뛰어나다', '멋지다', '성공', '효과', '품질', '가성비', '깔끔하다',
        '예쁘다', '괜찮다', '마음에들다', '든든하다', '편리하다', '유용하다',
        '빠르다', '정확하다', '친절하다', '싸다', '저렴하다', '합리적',
        '효율적', '신속하다', '만족스럽다', '감사하다', '고맙다', '도움',
        '편하다', '쉽다', '간편하다', '실용적', '경제적', '알뜰하다'
    ],
    'negative': [
        '나쁘다', '별로', '최악', '실망', '불만', '문제', '부족', '아쉽다',
        '후회', '비추', '개선', '단점', '불편', '오류', '느리다', '비싸다',
        '어렵다', '복잡하다', '귀찮다', '짜증', '화나다', '속상하다',
        '걱정', '불안', '고장', '파손', '결함', '흠집', '더럽다',
        '냄새', '시끄럽다', '무겁다', '작다', '크다', '딱딱하다', '질기다'
    ]
}

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

# 분석 매개변수
ANALYSIS_PARAMS = {
    'max_tfidf_features': 100,
    'ngram_range': (1, 2),
    'min_word_count': 5,
    'max_word_length': 10,
    'n_topics': 5,
    'sentiment_threshold': 3
}

def get_output_filename(prefix="coupang_analysis_report"):
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