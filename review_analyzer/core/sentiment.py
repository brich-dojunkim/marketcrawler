# core/sentiment.py
"""감성 분석 모듈 - 데이터 기반 접근"""

import pandas as pd
import numpy as np
from typing import List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer

class SentimentAnalyzer:
    """감성 분석 클래스 - 데이터 기반"""
    
    def __init__(self):
        """감성 분석기 초기화"""
        self.learned_patterns = {'positive': [], 'negative': []}
        self.is_trained = False
    
    def learn_patterns_from_ratings(self, df: pd.DataFrame, text_col: str = 'tokens_str', 
                                  rating_col: str = 'rating') -> Dict:
        """
        평점 데이터로부터 감성 패턴 학습
        
        Args:
            df: 입력 DataFrame
            text_col: 텍스트 컬럼명
            rating_col: 평점 컬럼명
            
        Returns:
            학습된 패턴 딕셔너리
        """
        if rating_col not in df.columns:
            print("⚠️ 평점 컬럼이 없어 패턴 학습을 건너뜁니다.")
            return {}
        
        print("📖 평점 데이터로부터 감성 패턴 학습 중...")
        
        # 평점 기준으로 긍정/부정 분류
        high_rating_texts = df[df[rating_col] >= 4][text_col].dropna()
        low_rating_texts = df[df[rating_col] <= 2][text_col].dropna()
        
        if len(high_rating_texts) == 0 or len(low_rating_texts) == 0:
            print("⚠️ 충분한 긍정/부정 데이터가 없습니다.")
            return {}
        
        try:
            # 긍정 패턴 추출
            positive_text = ' '.join(high_rating_texts)
            pos_vectorizer = TfidfVectorizer(max_features=30, token_pattern=r'\b\w+\b')
            pos_tfidf = pos_vectorizer.fit_transform([positive_text])
            self.learned_patterns['positive'] = list(pos_vectorizer.get_feature_names_out())
            
            # 부정 패턴 추출  
            negative_text = ' '.join(low_rating_texts)
            if len(negative_text.strip()) > 0:
                neg_vectorizer = TfidfVectorizer(max_features=20, token_pattern=r'\b\w+\b')
                neg_tfidf = neg_vectorizer.fit_transform([negative_text])
                self.learned_patterns['negative'] = list(neg_vectorizer.get_feature_names_out())
            
            self.is_trained = True
            
            print(f"✅ 감성 패턴 학습 완료:")
            print(f"   긍정 패턴: {len(self.learned_patterns['positive'])}개")
            print(f"   부정 패턴: {len(self.learned_patterns['negative'])}개")
            
            return self.learned_patterns
            
        except Exception as e:
            print(f"❌ 감성 패턴 학습 실패: {e}")
            return {}
    
    def analyze_sentiment_rating_based(self, rating: float, threshold: float = 3.0) -> str:
        """
        평점 기반 감성 분석 (가장 객관적)
        
        Args:
            rating: 평점 값
            threshold: 중립 기준점
            
        Returns:
            감성 라벨
        """
        if rating > threshold:
            return 'positive'
        elif rating < threshold:
            return 'negative'
        else:
            return 'neutral'
    
    def analyze_sentiment_learned_patterns(self, tokens: List[str]) -> str:
        """
        학습된 패턴 기반 감성 분석
        
        Args:
            tokens: 토큰화된 단어 리스트
            
        Returns:
            감성 라벨
        """
        if not self.is_trained:
            return 'neutral'
        
        pos_score = sum(1 for token in tokens if token in self.learned_patterns['positive'])
        neg_score = sum(1 for token in tokens if token in self.learned_patterns['negative'])
        
        if pos_score > neg_score:
            return 'positive'
        elif neg_score > pos_score:
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
            method: 감성 분석 방법 ('rating', 'learned', 'hybrid')
            tokens_column: 토큰 컬럼명
            rating_column: 평점 컬럼명
            
        Returns:
            감성 라벨이 추가된 DataFrame
        """
        print(f"🎯 {method} 방식으로 감성 분석 중...")
        
        df = df.copy()
        
        if method == 'rating' and rating_column and rating_column in df.columns:
            # 평점 기반 (가장 객관적)
            df['sentiment'] = df[rating_column].apply(self.analyze_sentiment_rating_based)
            
        elif method == 'learned' and tokens_column in df.columns:
            # 학습된 패턴 기반
            if not self.is_trained:
                # 패턴 학습 먼저 실행
                self.learn_patterns_from_ratings(df, 'tokens_str', rating_column)
            
            df['sentiment'] = df[tokens_column].apply(
                lambda x: self.analyze_sentiment_learned_patterns(x if isinstance(x, list) else x.split())
            )
            
        elif method == 'hybrid' and rating_column and tokens_column:
            # 하이브리드: 평점 우선, 패턴 보조
            df['sentiment_rating'] = df[rating_column].apply(self.analyze_sentiment_rating_based)
            
            if not self.is_trained:
                self.learn_patterns_from_ratings(df, 'tokens_str', rating_column)
            
            df['sentiment_learned'] = df[tokens_column].apply(
                lambda x: self.analyze_sentiment_learned_patterns(x if isinstance(x, list) else x.split())
            )
            
            # 평점 기반을 메인으로, 중립인 경우만 패턴 기반 사용
            df['sentiment'] = df.apply(
                lambda row: row['sentiment_learned'] if row['sentiment_rating'] == 'neutral' 
                           else row['sentiment_rating'], axis=1
            )
        else:
            # 기본값
            df['sentiment'] = 'neutral'
        
        print("✅ 감성 분석 완료")
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
    
    def get_learned_patterns(self) -> Dict:
        """학습된 패턴 반환"""
        return self.learned_patterns