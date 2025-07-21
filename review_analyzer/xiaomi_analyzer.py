# batch_xiaomi_review_analyzer.py
"""
샤오미 상품 리뷰 일괄 분석기
개별 리뷰 파일들을 분석하여 상품 정보와 함께 통합 결과표 생성
"""

import os
import sys
import pandas as pd
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

# review_analyzer 모듈 import
sys.path.append('/Users/brich/Desktop/marketcrawler/review_analyzer')
from utils.data_loader import load_csv_data, identify_columns, validate_data
from core.analyzer import MorphologicalAnalyzer
from core.sentiment import SentimentAnalyzer
from keywords.extractor import KeywordExtractor

class BatchXiaomiReviewAnalyzer:
    """샤오미 상품 리뷰 일괄 분석 클래스"""
    
    def __init__(self, reviews_dir: str, product_info_path: str, output_dir: str):
        """
        Args:
            reviews_dir: 개별 리뷰 CSV 파일들이 있는 디렉토리
            product_info_path: 상품 정보 CSV 파일 경로
            output_dir: 결과 저장 디렉토리
        """
        self.reviews_dir = Path(reviews_dir)
        self.product_info_path = product_info_path
        self.output_dir = Path(output_dir)
        
        # 분석기 초기화
        self.morphological_analyzer = MorphologicalAnalyzer()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.keyword_extractor = KeywordExtractor()
        
        # 결과 저장용
        self.product_info_df = None
        self.analysis_results = []
        
        print("🚀 샤오미 상품 리뷰 일괄 분석기 초기화 완료")
    
    def load_product_info(self) -> bool:
        """상품 정보 CSV 로드"""
        try:
            self.product_info_df = pd.read_csv(self.product_info_path)
            print(f"✅ 상품 정보 로드 완료: {len(self.product_info_df)}개 상품")
            print(f"📊 상품 정보 컬럼: {list(self.product_info_df.columns)}")
            return True
        except Exception as e:
            print(f"❌ 상품 정보 로드 실패: {e}")
            return False
    
    def extract_product_name_from_filename(self, filename: str) -> str:
        """파일명에서 상품명 추출"""
        # reviews_{product_name}_{timestamp}.csv 형태에서 product_name 추출
        pattern = r'reviews_(.+?)_\d{8}_\d{6}\.csv'
        match = re.match(pattern, filename)
        if match:
            product_name = match.group(1).replace('_', ' ').replace('-', ' ')
            return product_name
        return filename.replace('.csv', '')
    
    def find_matching_product_info(self, extracted_name: str) -> Optional[Dict]:
        """추출된 상품명으로 상품 정보 매칭"""
        if self.product_info_df is None:
            return None
        
        # 1. 정확한 매칭 시도
        for idx, row in self.product_info_df.iterrows():
            product_name = str(row['name']).lower()
            if extracted_name.lower() in product_name or product_name in extracted_name.lower():
                return {
                    'category': row['category'],
                    'subcategory': row['subcategory'],
                    'full_name': row['name'],
                    'sale_price': row.get('sale_price', 0),
                    'rating': row.get('rating', 0),
                    'rating_count': row.get('rating_count', 0)
                }
        
        # 2. 부분 매칭 시도 (키워드 기반)
        extracted_keywords = extracted_name.lower().split()
        best_match = None
        max_matches = 0
        
        for idx, row in self.product_info_df.iterrows():
            product_name = str(row['name']).lower()
            matches = sum(1 for keyword in extracted_keywords if keyword in product_name)
            if matches > max_matches and matches >= 2:  # 최소 2개 키워드 매칭
                max_matches = matches
                best_match = {
                    'category': row['category'],
                    'subcategory': row['subcategory'],
                    'full_name': row['name'],
                    'sale_price': row.get('sale_price', 0),
                    'rating': row.get('rating', 0),
                    'rating_count': row.get('rating_count', 0)
                }
        
        return best_match
    
    def analyze_single_review_file(self, file_path: Path) -> Optional[Dict]:
        """개별 리뷰 파일 분석"""
        try:
            print(f"\n📝 분석 중: {file_path.name}")
            
            # 1. 데이터 로드
            df = load_csv_data(str(file_path))
            if df is None or len(df) == 0:
                print(f"⚠️ 데이터 없음: {file_path.name}")
                return None
            
            # 2. 컬럼 식별
            text_columns, rating_columns = identify_columns(df)
            if not text_columns:
                print(f"⚠️ 텍스트 컬럼 없음: {file_path.name}")
                return None
            
            text_col = text_columns[0]
            rating_col = rating_columns[0] if rating_columns else None
            
            # 3. 데이터 유효성 검증
            if not validate_data(df, text_col, rating_col):
                return None
            
            # 4. 기본 통계
            total_reviews = len(df)
            avg_review_length = df[text_col].str.len().mean()
            avg_rating = df[rating_col].mean() if rating_col else None
            
            # 5. 텍스트 전처리 (간단히)
            df['cleaned_review'] = df[text_col].fillna('')
            df = df[df['cleaned_review'].str.len() > 0]  # 빈 리뷰 제거
            
            if len(df) == 0:
                return None
            
            # 6. 형태소 분석
            df = self.morphological_analyzer.tokenize_dataframe(df, text_col)
            
            # 7. 감성 분석
            if rating_col:
                df = self.sentiment_analyzer.create_sentiment_labels(
                    df, method='rating', rating_column=rating_col
                )
            else:
                df['sentiment'] = 'neutral'
            
            # 8. 키워드 추출
            texts = df['tokens_str'].tolist()
            self.keyword_extractor.extract_auto_stopwords(texts)
            
            # TF-IDF 키워드
            tfidf_keywords = self.keyword_extractor.extract_keywords_tfidf(df)
            top_keywords = [word for word, score in tfidf_keywords[:10]]
            
            # 감성별 키워드
            sentiment_keywords = self.keyword_extractor.extract_sentiment_keywords(df)
            
            # 9. 감성 분포 계산
            sentiment_counts = df['sentiment'].value_counts()
            total_sentiment = len(df)
            
            positive_ratio = sentiment_counts.get('positive', 0) / total_sentiment * 100
            negative_ratio = sentiment_counts.get('negative', 0) / total_sentiment * 100
            neutral_ratio = sentiment_counts.get('neutral', 0) / total_sentiment * 100
            
            # 10. 상품 정보 매칭
            extracted_name = self.extract_product_name_from_filename(file_path.name)
            product_info = self.find_matching_product_info(extracted_name)
            
            # 11. 결과 구성
            result = {
                'file_name': file_path.name,
                'extracted_name': extracted_name,
                'category': product_info['category'] if product_info else '미분류',
                'subcategory': product_info['subcategory'] if product_info else '미분류',
                'full_product_name': product_info['full_name'] if product_info else extracted_name,
                'sale_price': product_info['sale_price'] if product_info else 0,
                'store_rating': product_info['rating'] if product_info else 0,
                'store_rating_count': product_info['rating_count'] if product_info else 0,
                'total_reviews': total_reviews,
                'valid_reviews': len(df),
                'avg_review_length': round(avg_review_length, 1),
                'avg_rating': round(avg_rating, 2) if avg_rating else None,
                'positive_count': sentiment_counts.get('positive', 0),
                'negative_count': sentiment_counts.get('negative', 0),
                'neutral_count': sentiment_counts.get('neutral', 0),
                'positive_ratio': round(positive_ratio, 1),
                'negative_ratio': round(negative_ratio, 1),
                'neutral_ratio': round(neutral_ratio, 1),
                'top_keywords': ', '.join(top_keywords[:5]),
                'positive_keywords': ', '.join([word for word, score in sentiment_keywords.get('positive', [])[:3]]),
                'negative_keywords': ', '.join([word for word, score in sentiment_keywords.get('negative', [])[:3]]),
                'all_keywords': ', '.join(top_keywords)
            }
            
            print(f"✅ 분석 완료: {extracted_name} ({len(df)}개 리뷰)")
            return result
            
        except Exception as e:
            print(f"❌ 분석 실패 {file_path.name}: {e}")
            return None
    
    def run_batch_analysis(self) -> List[Dict]:
        """일괄 분석 실행"""
        print("\n" + "="*60)
        print("🔄 샤오미 상품 리뷰 일괄 분석 시작")
        print("="*60)
        
        # 1. 상품 정보 로드
        if not self.load_product_info():
            return []
        
        # 2. 리뷰 파일 목록 수집
        review_files = list(self.reviews_dir.glob("*.csv"))
        print(f"📁 발견된 리뷰 파일: {len(review_files)}개")
        
        if not review_files:
            print("❌ 리뷰 파일이 없습니다.")
            return []
        
        # 3. 각 파일 분석
        self.analysis_results = []
        for i, file_path in enumerate(review_files, 1):
            print(f"\n[{i}/{len(review_files)}] 처리 중...")
            result = self.analyze_single_review_file(file_path)
            if result:
                self.analysis_results.append(result)
        
        print(f"\n✅ 일괄 분석 완료: {len(self.analysis_results)}개 상품")
        return self.analysis_results
    
    def create_comprehensive_report(self) -> str:
        """종합 분석 리포트 생성"""
        if not self.analysis_results:
            print("❌ 분석 결과가 없습니다.")
            return ""
        
        # 출력 디렉토리 생성
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"xiaomi_batch_analysis_{timestamp}.xlsx"
        
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            
            # 1. 메인 요약 테이블
            self._create_main_summary_sheet(writer)
            
            # 2. 카테고리별 분석
            self._create_category_analysis_sheet(writer)
            
            # 3. 상세 분석 결과
            self._create_detailed_analysis_sheet(writer)
            
            # 4. 키워드 분석
            self._create_keyword_analysis_sheet(writer)
            
            # 5. 통계 요약
            self._create_statistics_summary_sheet(writer)
        
        print(f"\n🎉 종합 분석 리포트 생성 완료!")
        print(f"📄 파일 경로: {output_file}")
        
        return str(output_file)
    
    def _create_main_summary_sheet(self, writer):
        """메인 요약 시트 생성"""
        df_results = pd.DataFrame(self.analysis_results)
        
        # 컬럼 순서 정리
        summary_columns = [
            'full_product_name', 'category', 'subcategory', 'total_reviews', 
            'avg_rating', 'positive_ratio', 'negative_ratio', 'neutral_ratio',
            'top_keywords', 'positive_keywords', 'negative_keywords', 'sale_price'
        ]
        
        summary_df = df_results[summary_columns].copy()
        summary_df.columns = [
            '상품명', '카테고리', '서브카테고리', '총 리뷰수', '평균평점', 
            '긍정비율(%)', '부정비율(%)', '중립비율(%)', '주요키워드', 
            '긍정키워드', '부정키워드', '판매가격'
        ]
        
        # 정렬 (카테고리 -> 긍정비율 순)
        summary_df = summary_df.sort_values(['카테고리', '긍정비율(%)'], ascending=[True, False])
        
        summary_df.to_excel(writer, sheet_name='1_전체요약', index=False)
        print("✅ 메인 요약 시트 생성 완료")
    
    def _create_category_analysis_sheet(self, writer):
        """카테고리별 분석 시트 생성"""
        df_results = pd.DataFrame(self.analysis_results)
        
        # 카테고리별 통계
        category_stats = df_results.groupby('category').agg({
            'total_reviews': 'sum',
            'avg_rating': 'mean',
            'positive_ratio': 'mean',
            'negative_ratio': 'mean',
            'neutral_ratio': 'mean',
            'sale_price': 'mean',
            'full_product_name': 'count'
        }).round(2)
        
        category_stats.columns = [
            '총 리뷰수', '평균평점', '평균 긍정비율', '평균 부정비율', 
            '평균 중립비율', '평균 판매가격', '상품수'
        ]
        
        category_stats = category_stats.sort_values('평균 긍정비율', ascending=False)
        category_stats.to_excel(writer, sheet_name='2_카테고리별분석')
        print("✅ 카테고리별 분석 시트 생성 완료")
    
    def _create_detailed_analysis_sheet(self, writer):
        """상세 분석 시트 생성"""
        df_results = pd.DataFrame(self.analysis_results)
        
        detailed_columns = [
            'full_product_name', 'category', 'subcategory', 'total_reviews', 
            'valid_reviews', 'avg_review_length', 'avg_rating', 'store_rating',
            'positive_count', 'negative_count', 'neutral_count', 
            'positive_ratio', 'negative_ratio', 'neutral_ratio',
            'sale_price', 'all_keywords'
        ]
        
        detailed_df = df_results[detailed_columns].copy()
        detailed_df.columns = [
            '상품명', '카테고리', '서브카테고리', '총 리뷰수', '유효 리뷰수',
            '평균 리뷰길이', '리뷰 평균평점', '스토어 평점', '긍정수', '부정수', 
            '중립수', '긍정비율', '부정비율', '중립비율', '판매가격', '전체키워드'
        ]
        
        detailed_df.to_excel(writer, sheet_name='3_상세분석', index=False)
        print("✅ 상세 분석 시트 생성 완료")
    
    def _create_keyword_analysis_sheet(self, writer):
        """키워드 분석 시트 생성"""
        df_results = pd.DataFrame(self.analysis_results)
        
        # 카테고리별 키워드 분석
        keyword_data = []
        
        for category in df_results['category'].unique():
            category_data = df_results[df_results['category'] == category]
            
            # 긍정/부정 키워드 수집
            all_positive = ' '.join(category_data['positive_keywords'].fillna(''))
            all_negative = ' '.join(category_data['negative_keywords'].fillna(''))
            
            keyword_data.append({
                '카테고리': category,
                '상품수': len(category_data),
                '평균 긍정비율': category_data['positive_ratio'].mean(),
                '주요 긍정키워드': all_positive[:100],
                '주요 부정키워드': all_negative[:100]
            })
        
        keyword_df = pd.DataFrame(keyword_data)
        keyword_df.to_excel(writer, sheet_name='4_키워드분석', index=False)
        print("✅ 키워드 분석 시트 생성 완료")
    
    def _create_statistics_summary_sheet(self, writer):
        """통계 요약 시트 생성"""
        df_results = pd.DataFrame(self.analysis_results)
        
        stats_data = [
            ['전체 통계', '', ''],
            ['분석된 상품 수', len(df_results), '개'],
            ['총 리뷰 수', df_results['total_reviews'].sum(), '개'],
            ['평균 리뷰 수 (상품당)', df_results['total_reviews'].mean(), '개'],
            ['전체 평균 긍정비율', df_results['positive_ratio'].mean(), '%'],
            ['전체 평균 부정비율', df_results['negative_ratio'].mean(), '%'],
            ['', '', ''],
            ['카테고리별 상품 수', '', ''],
        ]
        
        # 카테고리별 상품 수
        category_counts = df_results['category'].value_counts()
        for category, count in category_counts.items():
            stats_data.append([category, count, '개'])
        
        stats_df = pd.DataFrame(stats_data, columns=['항목', '값', '단위'])
        stats_df.to_excel(writer, sheet_name='5_통계요약', index=False)
        print("✅ 통계 요약 시트 생성 완료")
    
    def print_summary(self):
        """분석 결과 요약 출력"""
        if not self.analysis_results:
            return
        
        df_results = pd.DataFrame(self.analysis_results)
        
        print("\n" + "="*60)
        print("📊 샤오미 상품 리뷰 분석 결과 요약")
        print("="*60)
        
        print(f"🔢 분석된 상품 수: {len(df_results)}개")
        print(f"📝 총 리뷰 수: {df_results['total_reviews'].sum():,}개")
        print(f"⭐ 전체 평균 긍정비율: {df_results['positive_ratio'].mean():.1f}%")
        
        print(f"\n📋 카테고리별 현황:")
        category_summary = df_results.groupby('category').agg({
            'full_product_name': 'count',
            'total_reviews': 'sum',
            'positive_ratio': 'mean'
        }).round(1)
        
        for category, row in category_summary.iterrows():
            print(f"   {category}: {row['full_product_name']}개 상품, "
                  f"{row['total_reviews']:,}개 리뷰, 긍정비율 {row['positive_ratio']}%")
        
        print(f"\n🏆 긍정비율 TOP 5 상품:")
        top_products = df_results.nlargest(5, 'positive_ratio')
        for idx, row in top_products.iterrows():
            print(f"   {row['full_product_name'][:30]:30} "
                  f"({row['category']:10}) 긍정비율: {row['positive_ratio']}%")
        
        print("="*60)


def main():
    """메인 실행 함수"""
    # 경로 설정
    reviews_dir = "/Users/brich/Desktop/marketcrawler/crawler/output/reviews_unique"
    product_info_path = "/Users/brich/Desktop/marketcrawler/crawler/output/xiaomi_store_20250704_103745.csv"
    output_dir = "/Users/brich/Desktop/marketcrawler/output"
    
    # 분석기 초기화 및 실행
    analyzer = BatchXiaomiReviewAnalyzer(reviews_dir, product_info_path, output_dir)
    
    # 일괄 분석 실행
    results = analyzer.run_batch_analysis()
    
    if results:
        # 종합 리포트 생성
        output_file = analyzer.create_comprehensive_report()
        
        # 결과 요약 출력
        analyzer.print_summary()
        
        print(f"\n🎉 분석 완료! 결과 파일: {output_file}")
    else:
        print("❌ 분석할 데이터가 없습니다.")


if __name__ == "__main__":
    main()