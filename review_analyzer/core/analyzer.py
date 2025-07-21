# core/analyzer.py
"""형태소 분석 모듈 - NaN 값 처리 개선"""

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
    
    def tokenize_with_kiwi(self, text: str, pos_filter: List[str] = ['NNG', 'NNP']) -> List[str]:
        """Kiwi를 이용한 형태소 분석 - NaN 값 처리 개선"""
        # NaN 값이나 빈 값 처리
        if pd.isna(text) or not text or text == '':
            return []
        
        # 문자열로 변환
        text = str(text).strip()
        if not text:
            return []
        
        try:
            tokens = self.kiwi.tokenize(text)
            filtered_tokens = []
            
            for token in tokens:
                if token.tag in pos_filter and len(token.form) > 1:
                    filtered_tokens.append(token.form)
            
            return filtered_tokens
        except Exception as e:
            print(f"⚠️ Kiwi 토큰화 오류 (텍스트: '{text[:50]}...'): {e}")
            return []
    
    def tokenize_with_okt(self, text: str, pos_filter: List[str] = ['Noun', 'Verb', 'Adjective']) -> List[str]:
        """OKT를 이용한 형태소 분석 - NaN 값 처리 개선"""
        # NaN 값이나 빈 값 처리
        if pd.isna(text) or not text or text == '':
            return []
        
        # 문자열로 변환
        text = str(text).strip()
        if not text:
            return []
        
        try:
            tokens = self.okt.pos(text, stem=True)
            filtered_tokens = []
            
            for word, pos in tokens:
                if pos in pos_filter and len(word) > 1:
                    filtered_tokens.append(word)
                    
            return filtered_tokens
        except Exception as e:
            print(f"⚠️ OKT 토큰화 오류 (텍스트: '{text[:50]}...'): {e}")
            return []
    
    def tokenize_dataframe(self, df: pd.DataFrame, text_column: str, 
                          method: str = 'kiwi') -> pd.DataFrame:
        """DataFrame의 텍스트 컬럼 토큰화 - 개선된 버전"""
        print(f"🔤 {method}로 리뷰 토큰화 중...")
        
        df = df.copy()
        
        # NaN 값과 빈 값 사전 처리
        print(f"📊 토큰화 전 데이터 상태:")
        print(f"   - 전체 행 수: {len(df)}")
        print(f"   - {text_column} 컬럼 NaN 개수: {df[text_column].isna().sum()}")
        print(f"   - {text_column} 컬럼 빈 문자열 개수: {(df[text_column] == '').sum()}")
        
        # NaN 값을 빈 문자열로 변환
        df[text_column] = df[text_column].fillna('')
        
        # 유효한 텍스트가 있는 행만 필터링
        valid_mask = df[text_column].astype(str).str.strip().str.len() > 0
        original_count = len(df)
        df = df[valid_mask].copy()
        filtered_count = len(df)
        
        if filtered_count < original_count:
            print(f"⚠️ 빈 텍스트로 인해 {original_count - filtered_count}개 행이 제거되었습니다.")
        
        if len(df) == 0:
            print("❌ 유효한 텍스트 데이터가 없습니다.")
            return df
        
        # 토큰화 실행
        if method == 'okt':
            df['tokens'] = df[text_column].apply(self.tokenize_with_okt)
        else:  # 기본값 kiwi
            df['tokens'] = df[text_column].apply(self.tokenize_with_kiwi)
        
        # 토큰화된 결과를 문자열로 변환
        df['tokens_str'] = df['tokens'].apply(lambda x: ' '.join(x) if x else '')
        
        # 토큰이 없는 행 제거
        empty_tokens_mask = df['tokens'].apply(len) > 0
        before_token_filter = len(df)
        df = df[empty_tokens_mask].copy()
        after_token_filter = len(df)
        
        if after_token_filter < before_token_filter:
            print(f"⚠️ 토큰이 없는 {before_token_filter - after_token_filter}개 행이 제거되었습니다.")
        
        print(f"✅ 토큰화 완료: {len(df)}개 리뷰")
        print(f"📊 토큰화 결과:")
        print(f"   - 평균 토큰 수: {df['tokens'].apply(len).mean():.1f}개")
        print(f"   - 최대 토큰 수: {df['tokens'].apply(len).max()}개")
        
        return df