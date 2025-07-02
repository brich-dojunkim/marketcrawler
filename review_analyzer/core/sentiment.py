# core/sentiment.py
"""감성 분석 모듈 - Transformers 통합"""

import pandas as pd
import numpy as np
from typing import List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer

class SentimentAnalyzer:
    """감성 분석 클래스 - 다중 방식 지원"""
    
    def __init__(self, use_transformers: bool = True):
        """
        감성 분석기 초기화
        
        Args:
            use_transformers: Transformers 모델 사용 여부
        """
        self.learned_patterns = {'positive': [], 'negative': []}
        self.is_trained = False
        self.transformers_pipeline = None
        
        # Transformers 초기화
        if use_transformers:
            self._initialize_transformers()
    
    def _initialize_transformers(self):
        """Transformers 모델 초기화"""
        try:
            import torch
            from transformers import pipeline
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"🧠 Transformers 감성 분석기 초기화 중... (Device: {device})")
            
            # 한국어 감성 분석 모델 시도
            model_options = [
                "beomi/KcELECTRA-base-v2022",
                "klue/roberta-base", 
                "cardiffnlp/twitter-xlm-roberta-base-sentiment"
            ]
            
            for model_name in model_options:
                try:
                    self.transformers_pipeline = pipeline(
                        "sentiment-analysis",
                        model=model_name,
                        device=0 if device == "cuda" else -1
                    )
                    print(f"✅ Transformers 모델 로드 완료: {model_name}")
                    break
                except Exception as e:
                    print(f"⚠️ {model_name} 로드 실패: {e}")
                    continue
            
            if not self.transformers_pipeline:
                print("❌ 모든 Transformers 모델 로드 실패")
                
        except ImportError:
            print("⚠️ transformers 라이브러리가 없습니다. 기본 감성 분석만 사용합니다.")
        except Exception as e:
            print(f"❌ Transformers 초기화 실패: {e}")
    
    def learn_patterns_from_ratings(self, df: pd.DataFrame, text_col: str = 'tokens_str', 
                                  rating_col: str = 'rating') -> Dict:
        """평점 데이터로부터 감성 패턴 학습"""
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
        """평점 기반 감성 분석"""
        if rating > threshold:
            return 'positive'
        elif rating < threshold:
            return 'negative'
        else:
            return 'neutral'
    
    def analyze_sentiment_learned_patterns(self, tokens: List[str]) -> str:
        """학습된 패턴 기반 감성 분석"""
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
    
    def analyze_sentiment_transformers(self, text: str) -> Dict:
        """Transformers 기반 감성 분석"""
        if not self.transformers_pipeline or not text.strip():
            return {'label': 'neutral', 'confidence': 0.0}
        
        try:
            # 텍스트 길이 제한
            text = text[:512]
            result = self.transformers_pipeline(text)
            
            # 결과 정규화
            label = result[0]['label'].lower()
            confidence = result[0]['score']
            
            # 라벨 매핑
            label_mapping = {
                'positive': 'positive',
                'negative': 'negative',
                'neutral': 'neutral',
                'label_0': 'negative',
                'label_1': 'neutral',
                'label_2': 'positive'
            }
            
            mapped_label = label_mapping.get(label, 'neutral')
            
            return {'label': mapped_label, 'confidence': confidence}
            
        except Exception as e:
            print(f"Transformers 감성 분석 오류: {e}")
            return {'label': 'neutral', 'confidence': 0.0}
    
    def create_sentiment_labels(self, df: pd.DataFrame, method: str = 'rating', 
                              tokens_column: str = 'tokens', 
                              rating_column: str = None,
                              text_column: str = None) -> pd.DataFrame:
        """
        감성 라벨 생성 (다중 방식 지원)
        
        Args:
            df: 입력 DataFrame
            method: 감성 분석 방법 ('rating', 'learned', 'transformers', 'ensemble')
            tokens_column: 토큰 컬럼명
            rating_column: 평점 컬럼명
            text_column: 원본 텍스트 컬럼명
            
        Returns:
            감성 라벨이 추가된 DataFrame
        """
        print(f"🎯 {method} 방식으로 감성 분석 중...")
        
        df = df.copy()
        
        if method == 'transformers' and self.transformers_pipeline and text_column:
            # Transformers 기반 감성 분석
            print("🧠 Transformers 기반 감성 분석 실행 중...")
            
            texts = df[text_column].fillna("").tolist()
            results = []
            
            # 배치 처리
            batch_size = 16
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                batch_results = [self.analyze_sentiment_transformers(text) for text in batch_texts]
                results.extend(batch_results)
            
            df['sentiment'] = [result['label'] for result in results]
            df['confidence_transformers'] = [result['confidence'] for result in results]
            
        elif method == 'ensemble' and self.transformers_pipeline and text_column and rating_column:
            # 앙상블: 평점 + Transformers + 학습된 패턴
            print("🎯 앙상블 감성 분석 실행 중...")
            
            # 1. 평점 기반
            df['sentiment_rating'] = df[rating_column].apply(self.analyze_sentiment_rating_based)
            
            # 2. Transformers 기반
            texts = df[text_column].fillna("").tolist()
            transformers_results = [self.analyze_sentiment_transformers(text) for text in texts]
            df['sentiment_transformers'] = [result['label'] for result in transformers_results]
            df['confidence_transformers'] = [result['confidence'] for result in transformers_results]
            
            # 3. 학습된 패턴
            if not self.is_trained:
                self.learn_patterns_from_ratings(df, 'tokens_str', rating_column)
            
            df['sentiment_learned'] = df[tokens_column].apply(
                lambda x: self.analyze_sentiment_learned_patterns(x if isinstance(x, list) else x.split())
            )
            
            # 4. 앙상블 결합 (가중 투표)
            df['sentiment'] = df.apply(self._ensemble_sentiment, axis=1)
            
        elif method == 'rating' and rating_column and rating_column in df.columns:
            # 평점 기반만
            df['sentiment'] = df[rating_column].apply(self.analyze_sentiment_rating_based)
            
        elif method == 'learned' and tokens_column in df.columns:
            # 학습된 패턴 기반만
            if not self.is_trained:
                self.learn_patterns_from_ratings(df, 'tokens_str', rating_column)
            
            df['sentiment'] = df[tokens_column].apply(
                lambda x: self.analyze_sentiment_learned_patterns(x if isinstance(x, list) else x.split())
            )
        else:
            # 기본값
            df['sentiment'] = 'neutral'
        
        print("✅ 감성 분석 완료")
        if 'sentiment' in df.columns:
            sentiment_counts = df['sentiment'].value_counts()
            print(sentiment_counts)
            
            # Transformers 결과가 있으면 평균 신뢰도도 출력
            if 'confidence_transformers' in df.columns:
                avg_confidence = df['confidence_transformers'].mean()
                print(f"평균 Transformers 신뢰도: {avg_confidence:.3f}")
        
        return df
    
    def _ensemble_sentiment(self, row) -> str:
        """앙상블 감성 결정 (가중 투표)"""
        votes = []
        weights = []
        
        # 평점 기반 (가중치: 0.4)
        if 'sentiment_rating' in row:
            votes.append(row['sentiment_rating'])
            weights.append(0.4)
        
        # Transformers (가중치: 신뢰도 기반)
        if 'sentiment_transformers' in row and 'confidence_transformers' in row:
            votes.append(row['sentiment_transformers'])
            weights.append(row['confidence_transformers'] * 0.4)
        
        # 학습된 패턴 (가중치: 0.2)
        if 'sentiment_learned' in row:
            votes.append(row['sentiment_learned'])
            weights.append(0.2)
        
        if not votes:
            return 'neutral'
        
        # 가중 투표
        sentiment_scores = {'positive': 0, 'negative': 0, 'neutral': 0}
        
        for vote, weight in zip(votes, weights):
            sentiment_scores[vote] += weight
        
        # 최고 점수 감성 반환
        return max(sentiment_scores, key=sentiment_scores.get)
    
    def compare_sentiment_methods(self, df: pd.DataFrame) -> Dict:
        """감성 분석 방법들 비교"""
        comparison = {}
        
        if 'sentiment_rating' in df.columns and 'sentiment_transformers' in df.columns:
            # 평점 vs Transformers 일치도
            agreement = (df['sentiment_rating'] == df['sentiment_transformers']).mean()
            comparison['rating_vs_transformers'] = agreement
            
            # 크로스탭
            crosstab = pd.crosstab(df['sentiment_rating'], df['sentiment_transformers'])
            comparison['crosstab'] = crosstab
            
            print(f"📊 평점 vs Transformers 일치도: {agreement:.3f}")
        
        return comparison
    
    def get_sentiment_statistics(self, df: pd.DataFrame, sentiment_column: str = 'sentiment') -> Dict:
        """감성 분석 통계 반환"""
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