# core/analyzer.py
"""형태소 분석 모듈"""

import pandas as pd
from typing import List
from kiwipiepy import Kiwi
from konlpy.tag import Okt

class MorphologicalAnalyzer:
    """형태소 분석 클래스"""
    
    def __init__(self):
        """형태소 분석기 초기화"""
        # Kiwi 초기화 (기본)
        self.kiwi = Kiwi()
        
        # KoNLPy 초기화
        self.okt = Okt()
        print("✅ 형태소 분석기 초기화 완료 (Kiwi + KoNLPy)")
    
    def tokenize_with_kiwi(self, text: str, pos_filter: List[str] = ['NNG', 'NNP', 'VA', 'VV']) -> List[str]:
        """Kiwi를 이용한 형태소 분석"""
        if not text:
            return []
        
        tokens = self.kiwi.tokenize(text)
        filtered_tokens = []
        
        for token in tokens:
            if token.tag in pos_filter and len(token.form) > 1:
                filtered_tokens.append(token.form)
        
        return filtered_tokens
    
    def tokenize_with_okt(self, text: str, pos_filter: List[str] = ['Noun', 'Verb', 'Adjective']) -> List[str]:
        """OKT를 이용한 형태소 분석"""
        if not text:
            return []
        
        tokens = self.okt.pos(text, stem=True)
        filtered_tokens = []
        
        for word, pos in tokens:
            if pos in pos_filter and len(word) > 1:
                filtered_tokens.append(word)
                
        return filtered_tokens
    
    def tokenize_dataframe(self, df: pd.DataFrame, text_column: str, 
                          method: str = 'kiwi') -> pd.DataFrame:
        """DataFrame의 텍스트 컬럼 토큰화"""
        print(f"🔤 {method}로 리뷰 토큰화 중...")
        
        df = df.copy()
        
        if method == 'okt':
            df['tokens'] = df[text_column].apply(self.tokenize_with_okt)
        else:  # 기본값 kiwi
            df['tokens'] = df[text_column].apply(self.tokenize_with_kiwi)
        
        # 토큰화된 결과를 문자열로 변환
        df['tokens_str'] = df['tokens'].apply(lambda x: ' '.join(x))
        
        print("✅ 토큰화 완료")
        return df