# keywords/extractor.py
"""키워드 추출 모듈"""

import pandas as pd
from typing import List, Tuple, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from soynlp.word import WordExtractor
from krwordrank.word import KRWordRank

class KeywordExtractor:
    """키워드 추출 클래스"""
    
    def __init__(self):
        """키워드 추출기 초기화"""
        pass
    
    def extract_keywords_tfidf(self, df: pd.DataFrame, text_column: str = 'tokens_str',
                              max_features: int = 100, ngram_range: Tuple[int, int] = (1, 2)) -> List[Tuple[str, float]]:
        """
        TF-IDF 기반 키워드 추출
        
        Args:
            df: 입력 DataFrame
            text_column: 텍스트 컬럼명
            max_features: 최대 특성 수
            ngram_range: n-gram 범위
            
        Returns:
            (키워드, 점수) 튜플 리스트
        """
        print("TF-IDF로 키워드 추출 중...")
        
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            token_pattern=r'\b\w+\b'
        )
        
        tfidf_matrix = vectorizer.fit_transform(df[text_column])
        feature_names = vectorizer.get_feature_names_out()
        
        # 단어별 TF-IDF 점수 계산
        tfidf_scores = tfidf_matrix.sum(axis=0).A1
        keyword_scores = list(zip(feature_names, tfidf_scores))
        keyword_scores.sort(key=lambda x: x[1], reverse=True)
        
        return keyword_scores
    
    def extract_keywords_krwordrank(self, df: pd.DataFrame, text_column: str = 'cleaned_review',
                                   min_count: int = 5, max_length: int = 10) -> List[Tuple[str, float]]:
        """
        KR-WordRank로 키워드 추출
        
        Args:
            df: 입력 DataFrame
            text_column: 텍스트 컬럼명
            min_count: 최소 출현 빈도
            max_length: 최대 단어 길이
            
        Returns:
            (키워드, 점수) 튜플 리스트
        """
        print("KR-WordRank로 키워드 추출 중...")
        
        texts = df[text_column].tolist()
        
        wordrank_extractor = KRWordRank(
            min_count=min_count,
            max_length=max_length
        )
        
        beta = 0.85
        max_iter = 10
        
        try:
            keywords, rank, graph = wordrank_extractor.extract(texts, beta, max_iter)
            # 키워드를 점수순으로 정렬
            sorted_keywords = sorted(keywords.items(), key=lambda x: x[1], reverse=True)
            return sorted_keywords
        except Exception as e:
            print(f"KR-WordRank 추출 중 오류: {e}")
            return []
    
    def extract_words_with_soynlp(self, df: pd.DataFrame, text_column: str = 'cleaned_review',
                                 min_count: int = 10, max_length: int = 10) -> Dict[str, float]:
        """
        soynlp를 이용한 단어 추출
        
        Args:
            df: 입력 DataFrame
            text_column: 텍스트 컬럼명
            min_count: 최소 출현 빈도
            max_length: 최대 단어 길이
            
        Returns:
            단어별 점수 딕셔너리
        """
        print("soynlp로 단어 추출 중...")
        
        texts = df[text_column].tolist()
        
        word_extractor = WordExtractor(
            min_count=min_count,
            min_length=2,
            max_length=max_length
        )
        
        word_extractor.train(texts)
        words = word_extractor.extract()
        
        # 단어 점수가 높은 순으로 정렬
        word_scores = {
            word: score.cohesion_forward * score.right_branching_entropy
            for word, score in words.items()
        }
        
        return word_scores
    
    def extract_sentiment_keywords(self, df: pd.DataFrame, sentiment_column: str = 'sentiment',
                                  text_column: str = 'tokens_str', top_n: int = 10) -> Dict[str, List[Tuple[str, float]]]:
        """
        감성별 키워드 추출
        
        Args:
            df: 입력 DataFrame
            sentiment_column: 감성 컬럼명
            text_column: 텍스트 컬럼명
            top_n: 상위 n개 키워드
            
        Returns:
            감성별 키워드 딕셔너리
        """
        sentiment_keywords = {}
        
        for sentiment in ['positive', 'negative', 'neutral']:
            if sentiment in df[sentiment_column].values:
                sentiment_data = df[df[sentiment_column] == sentiment]
                if len(sentiment_data) > 0:
                    sentiment_text = ' '.join(sentiment_data[text_column])
                    vectorizer = TfidfVectorizer(max_features=top_n, token_pattern=r'\b\w+\b')
                    try:
                        tfidf_matrix = vectorizer.fit_transform([sentiment_text])
                        feature_names = vectorizer.get_feature_names_out()
                        tfidf_scores = tfidf_matrix.toarray()[0]
                        keywords = sorted(zip(feature_names, tfidf_scores), 
                                        key=lambda x: x[1], reverse=True)
                        sentiment_keywords[sentiment] = keywords
                    except:
                        sentiment_keywords[sentiment] = []
        
        return sentiment_keywords