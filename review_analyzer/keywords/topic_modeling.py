# keywords/topic_modeling.py
"""토픽 모델링 모듈"""

import pandas as pd
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans

class TopicModeling:
    """토픽 모델링 클래스"""
    
    def __init__(self):
        """토픽 모델링 초기화"""
        pass
    
    def lda_topic_modeling(self, df: pd.DataFrame, text_column: str = 'tokens_str',
                          n_topics: int = 5, max_features: int = 100) -> List[List[str]]:
        """
        LDA 토픽 모델링
        
        Args:
            df: 입력 DataFrame
            text_column: 텍스트 컬럼명
            n_topics: 토픽 수
            max_features: 최대 특성 수
            
        Returns:
            토픽별 주요 단어 리스트
        """
        print(f"LDA로 {n_topics}개 토픽 추출 중...")
        
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            token_pattern=r'\b\w+\b'
        )
        
        tfidf_matrix = vectorizer.fit_transform(df[text_column])
        
        lda = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=42
        )
        
        lda.fit(tfidf_matrix)
        
        # 토픽별 주요 단어 추출
        feature_names = vectorizer.get_feature_names_out()
        topics = []
        
        for topic_idx, topic in enumerate(lda.components_):
            top_words_idx = topic.argsort()[-10:][::-1]
            top_words = [feature_names[i] for i in top_words_idx]
            topics.append(top_words)
        
        return topics
    
    def kmeans_clustering(self, df: pd.DataFrame, text_column: str = 'tokens_str',
                         n_clusters: int = 5, max_features: int = 100) -> pd.DataFrame:
        """
        K-Means 클러스터링 기반 토픽 모델링
        
        Args:
            df: 입력 DataFrame
            text_column: 텍스트 컬럼명
            n_clusters: 클러스터 수
            max_features: 최대 특성 수
            
        Returns:
            클러스터 라벨이 추가된 DataFrame
        """
        print(f"K-Means로 {n_clusters}개 클러스터 생성 중...")
        
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            token_pattern=r'\b\w+\b'
        )
        
        tfidf_matrix = vectorizer.fit_transform(df[text_column])
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(tfidf_matrix)
        
        df_with_clusters = df.copy()
        df_with_clusters['cluster'] = cluster_labels
        
        return df_with_clusters
    
    def get_cluster_keywords(self, df: pd.DataFrame, text_column: str = 'tokens_str',
                           cluster_column: str = 'cluster', top_n: int = 10) -> dict:
        """
        클러스터별 주요 키워드 추출
        
        Args:
            df: 클러스터 라벨이 있는 DataFrame
            text_column: 텍스트 컬럼명
            cluster_column: 클러스터 컬럼명
            top_n: 상위 n개 키워드
            
        Returns:
            클러스터별 키워드 딕셔너리
        """
        cluster_keywords = {}
        
        for cluster_id in df[cluster_column].unique():
            cluster_data = df[df[cluster_column] == cluster_id]
            cluster_text = ' '.join(cluster_data[text_column])
            
            vectorizer = TfidfVectorizer(max_features=top_n, token_pattern=r'\b\w+\b')
            try:
                tfidf_matrix = vectorizer.fit_transform([cluster_text])
                feature_names = vectorizer.get_feature_names_out()
                tfidf_scores = tfidf_matrix.toarray()[0]
                keywords = sorted(zip(feature_names, tfidf_scores), 
                                key=lambda x: x[1], reverse=True)
                cluster_keywords[cluster_id] = keywords
            except:
                cluster_keywords[cluster_id] = []
        
        return cluster_keywords