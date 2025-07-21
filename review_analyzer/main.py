# main.py
"""메인 실행 모듈 - 데이터 기반 분석"""

import warnings
import sys
import os

# 현재 디렉토리를 Python 경로에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

warnings.filterwarnings('ignore')

from utils.data_loader import load_csv_data, identify_columns, validate_data
from core.analyzer import MorphologicalAnalyzer
from core.sentiment import SentimentAnalyzer
from keywords.extractor import KeywordExtractor
from output.excel_exporter import ExcelExporter
from config.settings import DEFAULT_DATA_FILE, DEFAULT_OUTPUT_DIR, ANALYSIS_PARAMS

class DataDrivenReviewAnalysisSystem:
    """데이터 기반 리뷰 분석 시스템"""
    
    def __init__(self, output_dir=None):
        """
        분석 시스템 초기화
        
        Args:
            output_dir: 출력 디렉토리 경로
        """
        self.analyzer = MorphologicalAnalyzer()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.keyword_extractor = KeywordExtractor()
        self.excel_exporter = ExcelExporter(output_dir)
        
        # 분석 결과 저장
        self.df = None
        self.text_col = None
        self.rating_col = None
        self.meaningful_phrases = None
        self.cluster_topics = None
        self.sentiment_keywords = None
        self.learned_patterns = None
        self.transformers_comparison = None
        self.initial_count = 0
        self.final_count = 0
    
    def load_and_prepare_data(self, file_path):
        """데이터 로딩 및 준비"""
        print("🚀 데이터 기반 리뷰 분석 시스템")
        print("=" * 50)
        
        # 데이터 로딩
        self.df = load_csv_data(file_path)
        if self.df is None:
            return False
        
        self.initial_count = len(self.df)
        
        # 컬럼 분석 및 선택
        text_columns, rating_columns = identify_columns(self.df)
        
        # body 컬럼 우선 선택 (더 풍부한 내용)
        if 'body' in text_columns:
            self.text_col = 'body'
            print("📝 텍스트 컬럼: 'body' (상세 내용 우선 선택)")
        elif text_columns:
            self.text_col = text_columns[0]
            print(f"📝 텍스트 컬럼: '{self.text_col}'")
        else:
            print("❌ 텍스트 컬럼을 찾을 수 없습니다.")
            return False
        
        self.rating_col = rating_columns[0] if rating_columns else None
        if self.rating_col:
            print(f"⭐ 평점 컬럼: '{self.rating_col}'")
        else:
            print("⚠️ 평점 컬럼 없음 (텍스트 기반 분석만 수행)")
        
        # 데이터 유효성 검증
        if not validate_data(self.df, self.text_col, self.rating_col):
            return False
        
        self.final_count = len(self.df)
        
        print("\n🔄 데이터 기반 분석 시작...")
        print("=" * 50)
        
        return True
    
    def preprocess_data(self):
        """전처리 생략 - 원본 텍스트 그대로 사용"""
        print("1. 원본 텍스트 그대로 사용 (전처리 생략)")
        
        # 단순히 원본 텍스트를 그대로 복사
        self.df['cleaned_review'] = self.df[self.text_col]
        self.final_count = len(self.df)
        
        print(f"✅ 원본 텍스트 유지: {self.final_count:,}개 리뷰")
        print(f"📏 평균 텍스트 길이: {self.df['cleaned_review'].str.len().mean():.1f}자")
    
    def tokenize_data(self, method='kiwi'):
        """형태소 분석 - 원본 텍스트 직접 사용"""
        print("2. 형태소 분석 중... (원본 텍스트 직접 사용)")
        self.df = self.analyzer.tokenize_dataframe(self.df, self.text_col, method)
    
    def extract_auto_stopwords(self):
        """자동 불용어 추출"""
        print("3. 자동 불용어 추출 중...")
        texts = self.df['tokens_str'].tolist()
        self.keyword_extractor.extract_auto_stopwords(
            texts, ANALYSIS_PARAMS['auto_stopword_threshold']
        )
    
    def analyze_sentiment(self):
        """데이터 기반 감성 분석 (Transformers 포함)"""
        print("4. 고급 감성 분석 중...")
        
        if self.rating_col:
            # 앙상블 방식: 평점 + Transformers + 학습된 패턴
            self.df = self.sentiment_analyzer.create_sentiment_labels(
                self.df, 
                method='ensemble',  # 🆕 앙상블 방식
                tokens_column='tokens',
                rating_column=self.rating_col,
                text_column=self.text_col  # 원본 텍스트 (Transformers용)
            )
            
            # 학습된 패턴도 저장
            self.learned_patterns = self.sentiment_analyzer.get_learned_patterns()
            
            # Transformers vs 평점 기반 비교 분석
            if hasattr(self.sentiment_analyzer, 'transformers_pipeline') and self.sentiment_analyzer.transformers_pipeline:
                comparison = self.sentiment_analyzer.compare_sentiment_methods(self.df)
                self.transformers_comparison = comparison
            
        else:
            # 평점이 없는 경우 Transformers만 사용
            self.df = self.sentiment_analyzer.create_sentiment_labels(
                self.df,
                method='transformers',
                text_column=self.text_col
            )
            print("⚠️ 평점이 없어 Transformers 기반 감성 분석만 수행")
    
    def extract_meaningful_content(self):
        """의미있는 콘텐츠 추출"""
        print("5. 의미있는 콘텐츠 추출 중...")
        
        texts = self.df['tokens_str'].tolist()
        
        # 의미있는 구문 추출 (N-gram 기반)
        self.meaningful_phrases = self.keyword_extractor.extract_meaningful_phrases(
            texts, 
            min_freq=ANALYSIS_PARAMS['min_phrase_freq'],
            max_ngram=ANALYSIS_PARAMS['ngram_range'][1]
        )
        
        # 클러스터 기반 토픽 추출
        self.cluster_topics, self.df = self.keyword_extractor.cluster_based_topics(
            self.df, 'tokens_str', ANALYSIS_PARAMS['n_topics']
        )
        
        # 감성별 특징 키워드
        self.sentiment_keywords = self.keyword_extractor.extract_sentiment_keywords(
            self.df, 'sentiment', 'tokens_str'
        )
    
    def generate_data_driven_report(self):
        """데이터 기반 리포트 생성"""
        print("6. 데이터 기반 결과 출력 중...")
        
        # 엑셀 리포트 생성
        output_file = self.create_comprehensive_excel_report()
        
        return output_file
    
    def print_analysis_summary(self):
        """분석 결과 요약 출력"""
        print("\n" + "=" * 50)
        print("📊 데이터 기반 분석 결과 요약")
        print("=" * 50)
        
        # 기본 통계
        print(f"📈 데이터 현황:")
        print(f"   총 리뷰 수: {self.initial_count:,}개 → {self.final_count:,}개")
        print(f"   평균 리뷰 길이: {self.df[self.text_col].str.len().mean():.1f}자")
        print(f"   텍스트 컬럼: {self.text_col}")
        
        if self.rating_col:
            print(f"   평균 평점: {self.df[self.rating_col].mean():.2f}")
        
        # 자동 추출 결과
        print(f"\n🤖 자동 추출 결과:")
        print(f"   자동 불용어: {len(self.keyword_extractor.auto_stopwords)}개")
        if self.learned_patterns:
            print(f"   학습된 감성 패턴: 긍정 {len(self.learned_patterns.get('positive', []))}개, "
                  f"부정 {len(self.learned_patterns.get('negative', []))}개")
        print(f"   의미구문: {len(self.meaningful_phrases)}개")
        print(f"   토픽 클러스터: {len(self.cluster_topics)}개")
        
        # 감성 분포
        if 'sentiment' in self.df.columns:
            sentiment_counts = self.df['sentiment'].value_counts()
            print(f"\n💭 감성 분포 (평점 기반):")
            for sentiment, count in sentiment_counts.items():
                ratio = count / len(self.df) * 100
                sentiment_kr = {'positive': '긍정', 'negative': '부정', 'neutral': '중립'}[sentiment]
                print(f"   {sentiment_kr}: {count:,}개 ({ratio:.1f}%)")
        
        # 주요 의미구문
        print(f"\n🎯 주요 의미구문 TOP 10:")
        for i, (phrase, score) in enumerate(self.meaningful_phrases[:10], 1):
            print(f"   {i:2d}. {phrase} ({score:.1f})")
        
        # 토픽 클러스터
        print(f"\n📋 토픽 클러스터:")
        for topic_name, topic_info in self.cluster_topics.items():
            keywords = ', '.join(topic_info['keywords'][:5])
            print(f"   {topic_name} ({topic_info['ratio']:.1%}): {keywords}")
        
        print("=" * 50)
    
    def create_comprehensive_excel_report(self):
        """종합 엑셀 리포트 생성"""
        from datetime import datetime
        import pandas as pd
        
        output_file = os.path.join(
            self.excel_exporter.output_dir, 
            f"data_driven_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        )
        
        # 리포트 데이터 구성
        report_data = []
        
        # 분석 개요
        report_data.extend([
            ['=== 데이터 기반 리뷰 분석 리포트 ===', '', ''],
            ['분석 방식', '데이터 기반 (하드코딩 최소화)', '평점 기반 감성분석, 자동 불용어, 의미구문 추출'],
            ['분석 일시', datetime.now().strftime("%Y-%m-%d %H:%M:%S"), ''],
            ['', '', '']
        ])
        
        # 기본 통계
        report_data.extend([
            ['[기본 통계]', '', ''],
            ['총 리뷰 수 (원본)', f"{self.initial_count:,}개", ''],
            ['분석된 리뷰 수', f"{self.final_count:,}개", ''],
            ['텍스트 컬럼', self.text_col, ''],
            ['평점 컬럼', self.rating_col or '없음', ''],
            ['평균 리뷰 길이', f"{self.df[self.text_col].str.len().mean():.1f}자", '']
        ])
        
        if self.rating_col and self.rating_col in self.df.columns:
            report_data.append(['평균 평점', f"{self.df[self.rating_col].mean():.2f}", ''])
        
        report_data.append(['', '', ''])
        
        # 자동 추출 현황
        report_data.extend([
            ['[자동 추출 현황]', '', ''],
            ['자동 불용어 수', f"{len(self.keyword_extractor.auto_stopwords)}개", '문서 빈도 50% 이상'],
            ['의미구문 수', f"{len(self.meaningful_phrases)}개", 'N-gram 기반 자동 추출'],
            ['토픽 클러스터 수', f"{len(self.cluster_topics)}개", 'K-means 클러스터링'],
        ])
        
        if self.learned_patterns:
            pos_count = len(self.learned_patterns.get('positive', []))
            neg_count = len(self.learned_patterns.get('negative', []))
            report_data.append(['학습된 감성 패턴', f"긍정 {pos_count}개, 부정 {neg_count}개", '평점 4+ vs 2- 기반 학습'])
        
        report_data.append(['', '', ''])
        
        # 감성 분석 결과
        if 'sentiment' in self.df.columns:
            sentiment_counts = self.df['sentiment'].value_counts()
            report_data.extend([
                ['[감성 분석 결과]', '', ''],
                ['긍정 리뷰', f"{sentiment_counts.get('positive', 0):,}개 ({sentiment_counts.get('positive', 0)/len(self.df)*100:.1f}%)", '평점 기반'],
                ['부정 리뷰', f"{sentiment_counts.get('negative', 0):,}개 ({sentiment_counts.get('negative', 0)/len(self.df)*100:.1f}%)", '평점 기반'],
                ['중립 리뷰', f"{sentiment_counts.get('neutral', 0):,}개 ({sentiment_counts.get('neutral', 0)/len(self.df)*100:.1f}%)", '평점 기반'],
                ['', '', '']
            ])
        
        # 주요 의미구문
        report_data.extend([
            ['[주요 의미구문 TOP 20]', '', ''],
            ['순위', '의미구문', '점수']
        ])
        
        for i, (phrase, score) in enumerate(self.meaningful_phrases[:20], 1):
            report_data.append([i, phrase, round(score, 2)])
        
        report_data.append(['', '', ''])
        
        # 토픽 클러스터
        report_data.extend([
            ['[토픽 클러스터 분석]', '', ''],
            ['토픽', '주요 키워드', '문서 수 (비율)']
        ])
        
        for topic_name, topic_info in self.cluster_topics.items():
            keywords = ', '.join(topic_info['keywords'][:8])
            size_info = f"{topic_info['size']}개 ({topic_info['ratio']:.1%})"
            report_data.append([topic_name, keywords, size_info])
        
        report_data.append(['', '', ''])
        
        # 감성별 특징 키워드
        if self.sentiment_keywords:
            report_data.extend([
                ['[감성별 특징 키워드]', '', ''],
                ['감성', '특징 키워드', '설명']
            ])
            
            for sentiment in ['positive', 'negative', 'neutral']:
                if sentiment in self.sentiment_keywords and self.sentiment_keywords[sentiment]:
                    keywords = ', '.join([word for word, score in self.sentiment_keywords[sentiment][:10]])
                    sentiment_kr = {'positive': '긍정', 'negative': '부정', 'neutral': '중립'}[sentiment]
                    report_data.append([sentiment_kr, keywords, f'{sentiment_kr} 리뷰의 특징적 표현'])
        
        # Excel 파일 생성
        df_report = pd.DataFrame(report_data, columns=['항목', '값', '설명'])
        
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            df_report.to_excel(writer, sheet_name='데이터기반_분석_리포트', index=False)
            self.excel_exporter._style_worksheet(writer.sheets['데이터기반_분석_리포트'], report_data)
        
        print(f"✅ 데이터 기반 분석 리포트 저장:")
        print(f"📁 파일 경로: {output_file}")
        
        return output_file
    
    def run_full_analysis(self, file_path):
        """전체 데이터 기반 분석 파이프라인 실행"""
        try:
            # 1. 데이터 준비
            if not self.load_and_prepare_data(file_path):
                return None
            
            # 2. 형태소 분석 (전처리 없이 바로)
            self.tokenize_data()
            
            # 3. 자동 불용어 추출
            self.extract_auto_stopwords()
            
            # 4. 감성 분석
            self.analyze_sentiment()
            
            # 5. 의미있는 콘텐츠 추출
            self.extract_meaningful_content()
            
            # 6. 리포트 생성
            output_file = self.generate_data_driven_report()
            
            # 7. 요약 출력
            self.print_analysis_summary()
            
            print(f"\n🎉 데이터 기반 분석 완료!")
            print(f"📄 결과 파일: {output_file}")
            
            return output_file
            
        except Exception as e:
            print(f"❌ 분석 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()
            return None

def main():
    """메인 실행 함수 - 터미널 입력 방식"""
    
    # 사용자로부터 경로 입력받기
    from config.settings import get_user_paths
    
    input_file, output_dir = get_user_paths()
    
    if not input_file:
        print("❌ 올바른 파일을 선택해주세요.")
        return None
    
    print(f"\n🚀 분석을 시작합니다...")
    print(f"📁 입력: {input_file}")
    print(f"📁 출력: {output_dir}")
    
    # 데이터 기반 분석 시스템 초기화
    system = DataDrivenReviewAnalysisSystem(output_dir)
    
    # 전체 분석 실행
    output_file = system.run_full_analysis(input_file)
    
    return output_file

if __name__ == "__main__":
    main()