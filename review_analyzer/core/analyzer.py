# core/analyzer.py
"""형태소 분석 모듈"""

import pandas as pd
from typing import List
from kiwipiepy import Kiwi
from config.settings import check_library_availability, STOPWORDS

class MorphologicalAnalyzer:
    """형태소 분석 클래스"""
    
    def __init__(self):
        """형태소 분석기 초기화"""
        self.library_availability = check_library_availability()
        self.stopwords = STOPWORDS
        
        # Kiwi 초기화 (Java 불필요)
        self.kiwi = Kiwi()
        
        # KoNLPy 초기화 (Java 사용 가능한 경우)
        if self.library_availability['konlpy']:
            try:
                from konlpy.tag import Okt, Kkma
                self.okt = Okt()
                self.kkma = Kkma()
                print("KoNLPy 형태소 분석기 초기화 완료")
            except Exception as e:
                print(f"KoNLPy 초기화 실패: {e}")
                self.okt = None
                self.kkma = None
        else:
            self.okt = None
            self.kkma = None
    
    def tokenize_with_kiwi(self, text: str, pos_filter: List[str] = ['NNG', 'NNP', 'VA', 'VV']) -> List[str]:
        """
        Kiwi를 이용한 형태소 분석
        
        Args:
            text: 입력 텍스트
            pos_filter: 품사 필터 리스트
            
        Returns:
            필터링된 토큰 리스트
        """
        if not text:
            return []
        
        tokens = self.kiwi.tokenize(text)
        filtered_tokens = [
            token.form for token in tokens
            if token.tag in pos_filter 
            and token.form not in self.stopwords 
            and len(token.form) > 1
        ]
        
        return filtered_tokens
    
    def tokenize_with_okt(self, text: str, pos_filter: List[str] = ['Noun', 'Verb', 'Adjective']) -> List[str]:
        """
        OKT를 이용한 형태소 분석 (Java 필요)
        
        Args:
            text: 입력 텍스트
            pos_filter: 품사 필터 리스트
            
        Returns:
            필터링된 토큰 리스트
        """
        if not text:
            return []
        
        if not self.library_availability['konlpy'] or self.okt is None:
            print("KoNLPy를 사용할 수 없습니다. Kiwi를 사용하세요.")
            return self.tokenize_with_kiwi(text)
        
        try:
            tokens = self.okt.pos(text, stem=True)
            filtered_tokens = [
                word for word, pos in tokens 
                if pos in pos_filter 
                and word not in self.stopwords 
                and len(word) > 1
            ]
            return filtered_tokens
        except Exception as e:
            print(f"OKT 분석 오류: {e}")
            return self.tokenize_with_kiwi(text)
    
    def tokenize_dataframe(self, df: pd.DataFrame, text_column: str, 
                          method: str = 'kiwi') -> pd.DataFrame:
        """
        DataFrame의 텍스트 컬럼 토큰화
        
        Args:
            df: 입력 DataFrame
            text_column: 토큰화할 텍스트 컬럼명
            method: 사용할 형태소 분석기 ('kiwi' 또는 'okt')
            
        Returns:
            토큰이 추가된 DataFrame
        """
        print(f"{method}로 리뷰 토큰화 중...")
        
        df = df.copy()
        
        if method == 'okt':
            df['tokens'] = df[text_column].apply(self.tokenize_with_okt)
        elif method == 'kiwi':
            df['tokens'] = df[text_column].apply(self.tokenize_with_kiwi)
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        # 토큰화된 결과를 문자열로 변환
        df['tokens_str'] = df['tokens'].apply(lambda x: ' '.join(x))
        
        print("토큰화 완료")
        return df