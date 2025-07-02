# visualization/charts.py
"""시각화 모듈"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from wordcloud import WordCloud
from typing import List, Tuple, Optional
from config.settings import get_font_path

class Visualizer:
    """시각화 클래스"""
    
    def __init__(self):
        """시각화 초기화"""
        # 한글 폰트 설정
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['axes.unicode_minus'] = False
        
        # macOS 한글 폰트
        font_path = get_font_path()
        if font_path:
            plt.rcParams['font.family'] = 'NanumGothic'
    
    def plot_sentiment_distribution(self, df: pd.DataFrame, sentiment_column: str = 'sentiment',
                                   rating_column: Optional[str] = None):
        """
        감성 분포 시각화
        
        Args:
            df: 입력 DataFrame
            sentiment_column: 감성 컬럼명
            rating_column: 평점 컬럼명 (선택사항)
        """
        if rating_column:
            plt.figure(figsize=(14, 6))
            
            # 감성별 분포
            plt.subplot(1, 2, 1)
            sentiment_counts = df[sentiment_column].value_counts()
            plt.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%')
            plt.title('감성 분포')
            
            # 평점별 분포
            plt.subplot(1, 2, 2)
            plt.hist(df[rating_column], bins=5, edgecolor='black')
            plt.xlabel('평점')
            plt.ylabel('리뷰 수')
            plt.title('평점 분포')
        else:
            plt.figure(figsize=(8, 6))
            sentiment_counts = df[sentiment_column].value_counts()
            plt.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%')
            plt.title('감성 분포')
        
        plt.tight_layout()
        plt.show()
    
    def create_wordcloud(self, df: pd.DataFrame, text_column: str = 'tokens_str',
                        sentiment: Optional[str] = None, sentiment_column: str = 'sentiment',
                        max_words: int = 100):
        """
        워드클라우드 생성
        
        Args:
            df: 입력 DataFrame
            text_column: 텍스트 컬럼명
            sentiment: 특정 감성 필터 (선택사항)
            sentiment_column: 감성 컬럼명
            max_words: 최대 단어 수
        """
        if sentiment:
            text_data = df[df[sentiment_column] == sentiment][text_column]
            title = f'{sentiment.title()} 리뷰 워드클라우드'
        else:
            text_data = df[text_column]
            title = '전체 리뷰 워드클라우드'
        
        text = ' '.join(text_data)
        
        wordcloud_params = {
            'width': 800,
            'height': 400,
            'background_color': 'white',
            'max_words': max_words
        }
        
        font_path = get_font_path()
        if font_path:
            wordcloud_params['font_path'] = font_path
        
        wordcloud = WordCloud(**wordcloud_params).generate(text)
        
        plt.figure(figsize=(12, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(title, fontsize=16)
        plt.show()
    
    def plot_keyword_ranking(self, keywords: List[Tuple[str, float]], 
                           title: str = 'Keywords Ranking', top_n: int = 20):
        """
        키워드 랭킹 시각화
        
        Args:
            keywords: (키워드, 점수) 튜플 리스트
            title: 차트 제목
            top_n: 상위 n개 키워드
        """
        if not keywords:
            print("키워드 데이터가 없습니다.")
            return
        
        top_keywords = keywords[:top_n]
        words = [item[0] for item in top_keywords]
        scores = [item[1] for item in top_keywords]
        
        plt.figure(figsize=(12, 8))
        plt.barh(range(len(words)), scores)
        plt.yticks(range(len(words)), words)
        plt.xlabel('점수')
        plt.title(title)
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()
    
    def plot_sentiment_by_rating(self, df: pd.DataFrame, rating_column: str,
                                sentiment_column: str = 'sentiment'):
        """
        평점별 감성 분포 시각화
        
        Args:
            df: 입력 DataFrame
            rating_column: 평점 컬럼명
            sentiment_column: 감성 컬럼명
        """
        plt.figure(figsize=(12, 6))
        
        # 평점별 감성 분포
        sentiment_by_rating = pd.crosstab(df[rating_column], df[sentiment_column], normalize='index') * 100
        
        sentiment_by_rating.plot(kind='bar', stacked=True)
        plt.title('평점별 감성 분포')
        plt.xlabel('평점')
        plt.ylabel('비율 (%)')
        plt.legend(title='감성')
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.show()
    
    def plot_topic_distribution(self, topics: List[List[str]]):
        """
        토픽별 주요 키워드 시각화
        
        Args:
            topics: 토픽별 키워드 리스트
        """
        fig, axes = plt.subplots(1, len(topics), figsize=(20, 6))
        
        if len(topics) == 1:
            axes = [axes]
        
        for i, topic_words in enumerate(topics):
            # 단어 빈도를 가상으로 생성 (실제로는 토픽 확률을 사용)
            word_counts = list(range(len(topic_words), 0, -1))
            
            axes[i].barh(topic_words[::-1], word_counts[::-1])
            axes[i].set_title(f'Topic {i+1}')
            axes[i].set_xlabel('Score')
        
        plt.tight_layout()
        plt.show()
    
    def plot_text_length_distribution(self, df: pd.DataFrame, text_column: str = 'cleaned_review'):
        """
        텍스트 길이 분포 시각화
        
        Args:
            df: 입력 DataFrame
            text_column: 텍스트 컬럼명
        """
        text_lengths = df[text_column].str.len()
        
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.hist(text_lengths, bins=50, edgecolor='black')
        plt.xlabel('텍스트 길이')
        plt.ylabel('빈도')
        plt.title('텍스트 길이 분포')
        
        plt.subplot(1, 2, 2)
        plt.boxplot(text_lengths)
        plt.ylabel('텍스트 길이')
        plt.title('텍스트 길이 박스플롯')
        
        plt.tight_layout()
        plt.show()