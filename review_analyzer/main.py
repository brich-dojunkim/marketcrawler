# main.py
"""메인 실행 모듈"""

import warnings
import sys
import os

# 현재 디렉토리를 Python 경로에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

warnings.filterwarnings('ignore')

from utils.data_loader import load_csv_data, identify_columns, select_columns, validate_data
from core.preprocessor import TextPreprocessor
from core.analyzer import MorphologicalAnalyzer
from core.sentiment import SentimentAnalyzer
from keywords.extractor import KeywordExtractor
from keywords.topic_modeling import TopicModeling
from visualization.charts import Visualizer
from output.excel_exporter import ExcelExporter
from config.settings import DEFAULT_DATA_FILE, DEFAULT_OUTPUT_DIR, ANALYSIS_PARAMS

class ReviewAnalysisSystem:
    """리뷰 분석 시스템 통합 클래스"""
    
    def __init__(self, output_dir=None):
        """
        분석 시스템 초기화
        
        Args:
            output_dir: 출력 디렉토리 경로
        """
        self.preprocessor = TextPreprocessor()
        self.analyzer = MorphologicalAnalyzer()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.keyword_extractor = KeywordExtractor()
        self.topic_modeling = TopicModeling()
        self.visualizer = Visualizer()
        self.excel_exporter = ExcelExporter(output_dir)
        
        # 분석 결과 저장
        self.df = None
        self.text_col = None
        self.rating_col = None
        self.keywords_tfidf = None
        self.keywords_krwordrank = None
        self.topics = None
        self.sentiment_keywords = None
        self.initial_count = 0
        self.final_count = 0
    
    def load_and_prepare_data(self, file_path):
        """
        데이터 로딩 및 준비
        
        Args:
            file_path: CSV 파일 경로
            
        Returns:
            성공 여부
        """
        print("🚀 쿠팡 리뷰 분석 시스템 시작")
        print("=" * 50)
        
        # 데이터 로딩
        self.df = load_csv_data(file_path)
        if self.df is None:
            return False
        
        self.initial_count = len(self.df)
        
        # 컬럼 분석 및 선택
        text_columns, rating_columns = identify_columns(self.df)
        self.text_col, self.rating_col = select_columns(text_columns, rating_columns)
        
        # 데이터 유효성 검증
        if not validate_data(self.df, self.text_col, self.rating_col):
            return False
        
        print("\n🔄 분석 시작...")
        print("=" * 50)
        
        return True
    
    def preprocess_data(self):
        """데이터 전처리"""
        print("1. 텍스트 전처리 중...")
        self.df = self.preprocessor.clean_dataframe(self.df, self.text_col)
        self.final_count = len(self.df)
    
    def tokenize_data(self, method='kiwi'):
        """형태소 분석"""
        print("2. 형태소 분석 중...")
        self.df = self.analyzer.tokenize_dataframe(self.df, 'cleaned_review', method)
    
    def analyze_sentiment(self):
        """감성 분석"""
        print("3. 감성 분석 중...")
        
        # 규칙 기반 감성 분석
        self.df = self.sentiment_analyzer.create_sentiment_labels(
            self.df, method='rule', tokens_column='tokens'
        )
        self.df = self.df.rename(columns={'sentiment': 'sentiment_rule'})
        
        # 평점 기반 감성 분석 (평점 컬럼이 있는 경우)
        if self.rating_col:
            sentiment_rating_df = self.sentiment_analyzer.create_sentiment_labels(
                self.df, method='rating', rating_column=self.rating_col
            )
            self.df['sentiment_rating'] = sentiment_rating_df['sentiment']
    
    def extract_keywords(self):
        """키워드 추출"""
        print("4. 키워드 추출 중...")
        
        # TF-IDF 키워드 추출
        self.keywords_tfidf = self.keyword_extractor.extract_keywords_tfidf(
            self.df, max_features=ANALYSIS_PARAMS['max_tfidf_features'],
            ngram_range=ANALYSIS_PARAMS['ngram_range']
        )
        
        # KR-WordRank 키워드 추출
        try:
            self.keywords_krwordrank = self.keyword_extractor.extract_keywords_krwordrank(
                self.df, min_count=ANALYSIS_PARAMS['min_word_count']
            )
        except Exception as e:
            print(f"KR-WordRank 키워드 추출 실패: {e}")
            self.keywords_krwordrank = None
        
        # 감성별 키워드 추출
        self.sentiment_keywords = self.keyword_extractor.extract_sentiment_keywords(
            self.df, sentiment_column='sentiment_rule'
        )
    
    def perform_topic_modeling(self):
        """토픽 모델링"""
        print("5. 토픽 모델링 중...")
        
        try:
            self.topics = self.topic_modeling.lda_topic_modeling(
                self.df, n_topics=ANALYSIS_PARAMS['n_topics'],
                max_features=ANALYSIS_PARAMS['max_tfidf_features']
            )
        except Exception as e:
            print(f"토픽 모델링 실패: {e}")
            self.topics = None
    
    def generate_reports(self, report_type='comprehensive'):
        """
        분석 결과 리포트 생성
        
        Args:
            report_type: 리포트 타입 ('comprehensive' 또는 'detailed')
            
        Returns:
            출력 파일 경로
        """
        print("6. 분석 결과 출력 중...")
        
        if report_type == 'comprehensive':
            output_file = self.excel_exporter.create_comprehensive_report(
                self.df, self.keywords_tfidf, self.keywords_krwordrank,
                self.topics, self.sentiment_keywords,
                self.initial_count, self.final_count,
                self.text_col, self.rating_col
            )
        elif report_type == 'detailed':
            output_file = self.excel_exporter.create_detailed_report(
                self.df, self.keywords_tfidf, self.keywords_krwordrank,
                self.topics, self.sentiment_keywords,
                self.initial_count, self.final_count,
                self.text_col, self.rating_col
            )
        else:
            raise ValueError(f"Unsupported report type: {report_type}")
        
        return output_file
    
    def print_summary(self):
        """분석 결과 요약 출력"""
        print("\n" + "=" * 50)
        print("📊 분석 결과 요약")
        print("=" * 50)
        
        print(f"총 리뷰 수: {self.initial_count:,}개 → {self.final_count:,}개")
        print(f"평균 리뷰 길이: {self.df['cleaned_review'].str.len().mean():.1f}자")
        
        if self.rating_col:
            print(f"평균 평점: {self.df[self.rating_col].mean():.2f}")
        
        # 감성 분석 결과
        sentiment_counts = self.df['sentiment_rule'].value_counts()
        print("\n감성 분석 결과:")
        for sentiment, count in sentiment_counts.items():
            ratio = count / len(self.df) * 100
            sentiment_kr = {'positive': '긍정', 'negative': '부정', 'neutral': '중립'}[sentiment]
            print(f"  {sentiment_kr}: {count:,}개 ({ratio:.1f}%)")
        
        # 주요 키워드
        print("\n주요 키워드 (TF-IDF TOP 10):")
        for i, (word, score) in enumerate(self.keywords_tfidf[:10], 1):
            print(f"  {i:2d}. {word} ({score:.3f})")
        
        print("=" * 50)
    
    def create_visualizations(self):
        """시각화 생성"""
        print("7. 시각화 생성 중...")
        
        # 감성 분포
        self.visualizer.plot_sentiment_distribution(self.df, 'sentiment_rule', self.rating_col)
        
        # 워드클라우드
        self.visualizer.create_wordcloud(self.df)
        
        # 키워드 랭킹
        if self.keywords_tfidf:
            self.visualizer.plot_keyword_ranking(self.keywords_tfidf, 'TF-IDF 키워드 랭킹')
        
        # 평점별 감성 분포 (평점이 있는 경우)
        if self.rating_col:
            self.visualizer.plot_sentiment_by_rating(self.df, self.rating_col, 'sentiment_rule')
        
        # 토픽 분포 (토픽이 있는 경우)
        if self.topics:
            self.visualizer.plot_topic_distribution(self.topics)
    
    def run_full_analysis(self, file_path, report_type='comprehensive', create_viz=False):
        """
        전체 분석 파이프라인 실행
        
        Args:
            file_path: 입력 CSV 파일 경로
            report_type: 리포트 타입 ('comprehensive' 또는 'detailed')
            create_viz: 시각화 생성 여부
            
        Returns:
            출력 파일 경로
        """
        try:
            # 1. 데이터 준비
            if not self.load_and_prepare_data(file_path):
                return None
            
            # 2. 전처리
            self.preprocess_data()
            
            # 3. 형태소 분석
            self.tokenize_data()
            
            # 4. 감성 분석
            self.analyze_sentiment()
            
            # 5. 키워드 추출
            self.extract_keywords()
            
            # 6. 토픽 모델링
            self.perform_topic_modeling()
            
            # 7. 리포트 생성
            output_file = self.generate_reports(report_type)
            
            # 8. 시각화 (선택사항)
            if create_viz:
                self.create_visualizations()
            
            # 9. 요약 출력
            self.print_summary()
            
            print(f"\n🎉 분석 완료!")
            print(f"📄 결과 파일: {output_file}")
            
            return output_file
            
        except Exception as e:
            print(f"❌ 분석 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()
            return None

def main():
    """메인 실행 함수"""
    # 설정
    file_path = f'{DEFAULT_OUTPUT_DIR}/{DEFAULT_DATA_FILE}'
    output_dir = DEFAULT_OUTPUT_DIR
    
    # 분석 시스템 초기화
    system = ReviewAnalysisSystem(output_dir)
    
    # 전체 분석 실행
    output_file = system.run_full_analysis(
        file_path=file_path,
        report_type='comprehensive',  # 'comprehensive' 또는 'detailed'
        create_viz=False  # 시각화 생성 여부
    )
    
    return output_file

def run_with_custom_file(file_path, output_dir=None, report_type='comprehensive'):
    """
    사용자 정의 파일로 분석 실행
    
    Args:
        file_path: 입력 CSV 파일 경로
        output_dir: 출력 디렉토리 (None인 경우 기본값 사용)
        report_type: 리포트 타입 ('comprehensive' 또는 'detailed')
        
    Returns:
        출력 파일 경로
    """
    system = ReviewAnalysisSystem(output_dir)
    return system.run_full_analysis(file_path, report_type)

def analyze_with_visualization(file_path, output_dir=None):
    """
    시각화 포함 분석 실행
    
    Args:
        file_path: 입력 CSV 파일 경로
        output_dir: 출력 디렉토리
        
    Returns:
        출력 파일 경로
    """
    system = ReviewAnalysisSystem(output_dir)
    return system.run_full_analysis(
        file_path=file_path,
        report_type='comprehensive',
        create_viz=True
    )

if __name__ == "__main__":
    main()