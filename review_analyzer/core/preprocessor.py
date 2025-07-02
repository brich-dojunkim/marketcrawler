# core/preprocessor.py
"""텍스트 전처리 모듈"""

import pandas as pd
import re
from typing import List, Optional
from config.settings import check_library_availability, STOPWORDS

class TextPreprocessor:
    """텍스트 전처리 클래스"""
    
    def __init__(self):
        self.library_availability = check_library_availability()
        self.stopwords = STOPWORDS
        
        # PyKoSpacing 초기화
        if self.library_availability['pykospacing']:
            try:
                from pykospacing import Spacing
                self.spacing = Spacing()
            except Exception as e:
                print(f"PyKoSpacing 초기화 실패: {e}")
                self.spacing = None
        else:
            self.spacing = None
    
    def simple_spacing_correction(self, text: str) -> str:
        """
        간단한 띄어쓰기 규칙 적용 (PyKoSpacing 대체)
        
        Args:
            text: 입력 텍스트
            
        Returns:
            띄어쓰기가 교정된 텍스트
        """
        spacing_rules = [
            # 조사 앞에 띄어쓰기
            (r'([가-힣])([은는이가을를])([가-힣])', r'\1 \2 \3'),
            # 어미 앞에 띄어쓰기  
            (r'([가-힣])([습니다해요])([가-힣])', r'\1\2 \3'),
            # 숫자와 단위 사이
            (r'([0-9])([가-힣])', r'\1 \2'),
            # 영어와 한글 사이
            (r'([a-zA-Z])([가-힣])', r'\1 \2'),
            (r'([가-힣])([a-zA-Z])', r'\1 \2'),
        ]
        
        for pattern, replacement in spacing_rules:
            text = re.sub(pattern, replacement, text)
        
        return text
    
    def preprocess_text(self, text: str) -> str:
        """
        텍스트 전처리
        - 띄어쓰기 교정
        - 특수문자 제거
        - 정규화
        
        Args:
            text: 입력 텍스트
            
        Returns:
            전처리된 텍스트
        """
        if pd.isna(text):
            return ""
        
        # 띄어쓰기 교정
        if self.spacing is not None:
            try:
                text = self.spacing(text)
            except Exception as e:
                print(f"띄어쓰기 교정 중 오류: {e}")
                text = self.simple_spacing_correction(text)
        else:
            text = self.simple_spacing_correction(text)
        
        # 특수문자 제거 (한글, 영문, 숫자, 공백만 유지)
        text = re.sub(r'[^가-힣a-zA-Z0-9\s]', '', text)
        
        # 연속된 공백 제거
        text = re.sub(r'\s+', ' ', text)
        
        # 앞뒤 공백 제거
        text = text.strip()
        
        return text
    
    def clean_dataframe(self, df: pd.DataFrame, text_column: str) -> pd.DataFrame:
        """
        DataFrame의 텍스트 컬럼 전처리
        
        Args:
            df: 입력 DataFrame
            text_column: 전처리할 텍스트 컬럼명
            
        Returns:
            전처리된 DataFrame
        """
        print("리뷰 데이터 전처리 중...")
        
        df = df.copy()
        df['cleaned_review'] = df[text_column].apply(self.preprocess_text)
        
        # 빈 리뷰 제거
        initial_count = len(df)
        df = df[df['cleaned_review'].str.len() > 0].reset_index(drop=True)
        final_count = len(df)
        
        print(f"전처리 완료: {initial_count:,}개 → {final_count:,}개 리뷰")
        
        return df