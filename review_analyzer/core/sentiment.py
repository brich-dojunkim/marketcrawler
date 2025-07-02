# core/sentiment.py
"""감성 분석 모듈"""

import pandas as pd
from typing import List, Dict
from config.settings import SENTIMENT_DICT

class SentimentAnalyzer:
    """감성 분석 클래스"""
    
    def __init__(self, sentiment_dict: Dict = None):
        """
        감성 분석기 초기화
        
        Args:
            sentiment_dict: 사용자 정의 감성사전
        """
        self.sentiment_dict = sentiment_dict or SENTIMENT_DICT
    
    def analyze_sentiment_rule_based(self, tokens: List[str]) -> str:
        """
        규칙 기반 감성 분석
        
        Args:
            tokens: 토큰화된 단어 리스트
            
        Returns:
            감성 라벨 ('positive', 'negative', 'neutral')
        """
        pos_score = sum(1 for token in tokens if token in self.sentiment_dict['positive'])
        neg_score = sum(1 for token in tokens if token in self.sentiment_dict['negative'])
        
        # 점수 차이로 중립 판단
        if abs(pos_score - neg_score) <= 1 and (pos_score + neg_score) <= 2:
            return 'neutral'
        elif pos_score > neg_score:
            return 'positive'
        elif neg_score > pos_score:
            return 'negative'
        else:
            return 'neutral'
    
    def analyze_sentiment_rating_based(self, rating: float, threshold: float = 3.0) -> str:
        """
        평점 기반 감성 라벨링
        
        Args:
            rating: 평점 값
            threshold: 중립 기준점
            
        Returns:
            감성 라벨 ('positive', 'negative', 'neutral')
        """
        if rating > threshold:
            return 'positive'
        elif rating < threshold:
            return 'negative'
        else:
            return 'neutral'
    
    def create_sentiment_labels(self, df: pd.DataFrame, method: str = 'rating', 
                              tokens_column: str = 'tokens', 
                              rating_column: str = None) -> pd.DataFrame:
        """
        감성 라벨 생성
        
        Args:
            df: 입력 DataFrame
            method: 감성 분석 방법 ('rating' 또는 'rule')
            tokens_column: 토큰 컬럼명
            rating_column: 평점 컬럼명
            
        Returns:
            감성 라벨이 추가된 DataFrame
        """
        print(f"{method} 방식으로 감성 라벨 생성 중...")
        
        df = df.copy()
        
        if method == 'rating' and rating_column and rating_column in df.columns:
            df['sentiment'] = df[rating_column].apply(self.analyze_sentiment_rating_based)
        elif method == 'rule' and tokens_column in df.columns:
            df['sentiment'] = df[tokens_column].apply(self.analyze_sentiment_rule_based)
        else:
            raise ValueError(f"Invalid method '{method}' or missing required columns")
        
        print("감성 라벨 생성 완료")
        print(df['sentiment'].value_counts())
        
        return df
    
    def get_sentiment_statistics(self, df: pd.DataFrame, sentiment_column: str = 'sentiment') -> Dict:
        """
        감성 분석 통계 반환
        
        Args:
            df: 입력 DataFrame
            sentiment_column: 감성 컬럼명
            
        Returns:
            감성 통계 딕셔너리
        """
        sentiment_counts = df[sentiment_column].value_counts()
        total_count = len(df)
        
        stats = {}
        for sentiment in ['positive', 'negative', 'neutral']:
            count = sentiment_counts.get(sentiment, 0)
            ratio = count / total_count * 100
            stats[sentiment] = {
                'count': count,
                'ratio': ratio
            }
        
        return stats