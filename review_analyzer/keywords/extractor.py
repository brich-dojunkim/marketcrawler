# keywords/extractor.py
"""키워드 추출 모듈 - 의미 단위 구문 중심"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

class KeywordExtractor:
    """키워드 추출 클래스 - 구문 중심"""
    
    def __init__(self):
        """키워드 추출기 초기화"""
        self.auto_stopwords = set()
    
    def extract_auto_stopwords(self, texts: List[str], threshold: float = 0.7) -> set:
        """
        자동 불용어 추출 - 너무 흔한 단어들
        
        Args:
            texts: 텍스트 리스트
            threshold: 불용어 판단 임계값 (문서 빈도 비율)
            
        Returns:
            추출된 불용어 집합
        """
        try:
            vectorizer = TfidfVectorizer(
                tokenizer=lambda x: x.split(),
                lowercase=False,
                min_df=2
            )
            
            tfidf_matrix = vectorizer.fit_transform(texts)
            feature_names = vectorizer.get_feature_names_out()
            
            # 문서 빈도 계산
            doc_freq = np.array((tfidf_matrix > 0).sum(axis=0)).flatten()
            total_docs = len(texts)
            doc_freq_ratio = doc_freq / total_docs
            
            # 너무 흔한 단어들을 불용어로 판정
            too_common = set(feature_names[doc_freq_ratio > threshold])
            
            self.auto_stopwords = too_common
            print(f"📊 자동 불용어 추출: {len(self.auto_stopwords)}개")
            
            return self.auto_stopwords
            
        except Exception as e:
            print(f"자동 불용어 추출 실패: {e}")
            return set()
    
    def extract_meaningful_phrases(self, texts: List[str], min_freq: int = 3, 
                                 max_ngram: int = 3) -> List[Tuple[str, float]]:
        """
        의미있는 구문 추출 (N-gram 기반)
        
        Args:
            texts: 텍스트 리스트
            min_freq: 최소 출현 빈도
            max_ngram: 최대 n-gram 길이
            
        Returns:
            (구문, 점수) 튜플 리스트
        """
        print("🔍 의미있는 구문 추출 중...")
        
        all_ngrams = []
        
        for text in texts:
            words = text.split()
            # 1-gram부터 max_ngram까지 생성
            for n in range(1, max_ngram + 1):
                for i in range(len(words) - n + 1):
                    ngram = ' '.join(words[i:i+n])
                    
                    # 불용어 필터링
                    if not any(word in self.auto_stopwords for word in ngram.split()):
                        # 길이 필터링
                        if 2 <= len(ngram.replace(' ', '')) <= 20:
                            all_ngrams.append(ngram)
        
        # 빈도 계산
        ngram_counts = Counter(all_ngrams)
        
        # 최소 빈도 이상만 선택
        frequent_ngrams = {ngram: count for ngram, count in ngram_counts.items() 
                          if count >= min_freq}
        
        # 점수 계산: 빈도 × 길이 가중치
        scored_phrases = []
        for ngram, freq in frequent_ngrams.items():
            word_count = len(ngram.split())
            # 긴 구문에 가중치, 하지만 과도하지 않게
            score = freq * (1 + word_count * 0.3)
            scored_phrases.append((ngram, score))
        
        # 점수순 정렬
        scored_phrases.sort(key=lambda x: x[1], reverse=True)
        
        print(f"✅ {len(scored_phrases)}개 의미구문 추출 완료")
        return scored_phrases
    
    def cluster_based_topics(self, df: pd.DataFrame, text_col: str = 'tokens_str', 
                           n_clusters: int = 5) -> Tuple[Dict, pd.DataFrame]:
        """
        클러스터링 기반 토픽 추출
        
        Args:
            df: 입력 DataFrame
            text_col: 텍스트 컬럼명
            n_clusters: 클러스터 수
            
        Returns:
            (토픽 딕셔너리, 클러스터 라벨이 추가된 DataFrame)
        """
        print(f"🎯 클러스터링 기반 {n_clusters}개 토픽 추출 중...")
        
        texts = df[text_col].tolist()
        
        try:
            # TF-IDF 벡터화 (자동 불용어 제외)
            stop_words = list(self.auto_stopwords) if self.auto_stopwords else None
            
            vectorizer = TfidfVectorizer(
                max_features=200,
                token_pattern=r'\b\w+\b',
                min_df=2,
                max_df=0.8,
                stop_words=stop_words
            )
            
            tfidf_matrix = vectorizer.fit_transform(texts)
            feature_names = vectorizer.get_feature_names_out()
            
            # K-means 클러스터링
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(tfidf_matrix)
            
            # 각 클러스터의 중심 키워드 추출
            topics = {}
            for i in range(n_clusters):
                center = kmeans.cluster_centers_[i]
                top_indices = center.argsort()[-10:][::-1]
                top_keywords = [feature_names[idx] for idx in top_indices]
                
                cluster_size = sum(cluster_labels == i)
                ratio = cluster_size / len(texts)
                
                topics[f'토픽_{i+1}'] = {
                    'keywords': top_keywords,
                    'size': cluster_size,
                    'ratio': ratio
                }
            
            # DataFrame에 클러스터 정보 추가
            df_with_clusters = df.copy()
            df_with_clusters['topic_cluster'] = cluster_labels
            
            print(f"✅ 클러스터 기반 토픽 추출 완료")
            for topic, info in topics.items():
                print(f"   {topic}: {info['size']}개 ({info['ratio']:.1%})")
            
            return topics, df_with_clusters
            
        except Exception as e:
            print(f"❌ 클러스터 기반 토픽 추출 실패: {e}")
            return {}, df.copy()
    
    def extract_keywords_tfidf(self, df: pd.DataFrame, text_column: str = 'tokens_str',
                              max_features: int = 100) -> List[Tuple[str, float]]:
        """
        TF-IDF 기반 키워드 추출 (개선됨)
        
        Args:
            df: 입력 DataFrame
            text_column: 텍스트 컬럼명
            max_features: 최대 특성 수
            
        Returns:
            (키워드, 점수) 튜플 리스트
        """
        print("📊 TF-IDF 키워드 추출 중...")
        
        # 자동 불용어 제외
        stop_words = list(self.auto_stopwords) if self.auto_stopwords else None
        
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),  # 1-2 gram
            token_pattern=r'\b\w+\b',
            stop_words=stop_words
        )
        
        try:
            tfidf_matrix = vectorizer.fit_transform(df[text_column])
            feature_names = vectorizer.get_feature_names_out()
            
            # 전체 TF-IDF 점수 합계
            tfidf_scores = tfidf_matrix.sum(axis=0).A1
            keyword_scores = list(zip(feature_names, tfidf_scores))
            keyword_scores.sort(key=lambda x: x[1], reverse=True)
            
            print(f"✅ TF-IDF 키워드 {len(keyword_scores)}개 추출 완료")
            return keyword_scores
            
        except Exception as e:
            print(f"❌ TF-IDF 키워드 추출 실패: {e}")
            return []
    
    def extract_sentiment_keywords(self, df: pd.DataFrame, sentiment_column: str = 'sentiment',
                                  text_column: str = 'tokens_str', top_n: int = 15) -> Dict[str, List[Tuple[str, float]]]:
        """
        감성별 특징 키워드 추출
        
        Args:
            df: 입력 DataFrame
            sentiment_column: 감성 컬럼명
            text_column: 텍스트 컬럼명
            top_n: 상위 n개 키워드
            
        Returns:
            감성별 키워드 딕셔너리
        """
        print("💭 감성별 특징 키워드 추출 중...")
        
        sentiment_keywords = {}
        
        for sentiment in ['positive', 'negative', 'neutral']:
            sentiment_data = df[df[sentiment_column] == sentiment]
            
            if len(sentiment_data) > 0:
                sentiment_texts = sentiment_data[text_column].tolist()
                combined_text = ' '.join(sentiment_texts)
                
                # 해당 감성의 특징 키워드 추출
                vectorizer = TfidfVectorizer(
                    max_features=top_n,
                    ngram_range=(1, 2),
                    token_pattern=r'\b\w+\b',
                    stop_words=list(self.auto_stopwords) if self.auto_stopwords else None
                )
                
                try:
                    tfidf_matrix = vectorizer.fit_transform([combined_text])
                    feature_names = vectorizer.get_feature_names_out()
                    tfidf_scores = tfidf_matrix.toarray()[0]
                    
                    keywords = sorted(zip(feature_names, tfidf_scores), 
                                    key=lambda x: x[1], reverse=True)
                    sentiment_keywords[sentiment] = keywords
                    
                except Exception:
                    sentiment_keywords[sentiment] = []
            else:
                sentiment_keywords[sentiment] = []
        
        return sentiment_keywords