# naver_review_analyzer.py
"""
네이버 상품 리뷰 일괄 분석기
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
from utils.data_loader import load_csv_data, validate_data
from core.analyzer import MorphologicalAnalyzer
from core.sentiment import SentimentAnalyzer
from keywords.extractor import KeywordExtractor

class NaverReviewAnalyzer:
    """네이버 상품 리뷰 일괄 분석 클래스"""
    
    def __init__(self, reviews_dir: str, output_dir: str):
        """
        Args:
            reviews_dir: 개별 리뷰 CSV 파일들이 있는 디렉토리
            output_dir: 결과 저장 디렉토리
        """
        self.reviews_dir = Path(reviews_dir)
        self.output_dir = Path(output_dir)
        
        # 분석기 초기화
        self.morphological_analyzer = MorphologicalAnalyzer()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.keyword_extractor = KeywordExtractor()
        
        # 결과 저장용
        self.analysis_results = []
        
        print("🚀 네이버 상품 리뷰 일괄 분석기 초기화 완료")
    
    def extract_product_name_from_filename(self, filename: str) -> str:
        """파일명에서 상품명 추출"""
        # .csv 확장자 제거
        product_name = filename.replace('.csv', '')
        # 언더스코어를 공백으로 변경
        product_name = product_name.replace('_', ' ')
        return product_name
    
    def detect_naver_columns(self, df: pd.DataFrame) -> Tuple[str, str, str]:
        """네이버 리뷰 컬럼 감지"""
        # 네이버 리뷰 표준 컬럼명
        text_col = 'content'  # 리뷰 내용
        rating_col = 'rating'  # 평점
        product_col = 'product_name'  # 상품명
        
        # 컬럼명 확인 및 대안 검색
        available_cols = df.columns.tolist()
        
        # 텍스트 컬럼 찾기
        if 'content' not in available_cols:
            text_candidates = ['review', 'text', 'comment', '내용', '리뷰']
            for candidate in text_candidates:
                if candidate in available_cols:
                    text_col = candidate
                    break
            else:
                # 가장 긴 텍스트를 가진 컬럼 찾기
                text_col = self._find_longest_text_column(df)
        
        # 평점 컬럼 찾기
        if 'rating' not in available_cols:
            rating_candidates = ['score', 'star', 'rate', '평점', '별점']
            for candidate in rating_candidates:
                if candidate in available_cols:
                    rating_col = candidate
                    break
            else:
                rating_col = None
        
        # 상품명 컬럼 찾기
        if 'product_name' not in available_cols:
            product_candidates = ['name', 'title', 'product', '상품명', '제품명']
            for candidate in product_candidates:
                if candidate in available_cols:
                    product_col = candidate
                    break
            else:
                product_col = None
        
        return text_col, rating_col, product_col
    
    def _find_longest_text_column(self, df: pd.DataFrame) -> str:
        """가장 긴 평균 텍스트 길이를 가진 컬럼 찾기"""
        text_columns = []
        for col in df.columns:
            if df[col].dtype == 'object':
                avg_length = df[col].astype(str).str.len().mean()
                if avg_length > 10:  # 평균 10자 이상
                    text_columns.append((col, avg_length))
        
        if text_columns:
            # 가장 긴 평균 길이를 가진 컬럼 반환
            return max(text_columns, key=lambda x: x[1])[0]
        
        # 기본값으로 첫 번째 object 컬럼 반환
        for col in df.columns:
            if df[col].dtype == 'object':
                return col
        
        return df.columns[0]  # 최후의 수단
    
    def analyze_single_review_file(self, file_path: Path) -> Optional[Dict]:
        """개별 리뷰 파일 분석"""
        try:
            print(f"\n📝 분석 중: {file_path.name}")
            
            # 1. 데이터 로드
            df = load_csv_data(str(file_path))
            if df is None or len(df) == 0:
                print(f"⚠️ 데이터 없음: {file_path.name}")
                return None
            
            # 2. 네이버 리뷰 컬럼 감지
            text_col, rating_col, product_col = self.detect_naver_columns(df)
            print(f"📋 감지된 컬럼: 텍스트='{text_col}', 평점='{rating_col}', 상품='{product_col}'")
            
            # 3. 데이터 유효성 검증
            if not validate_data(df, text_col, rating_col):
                return None
            
            # 4. 기본 통계
            total_reviews = len(df)
            avg_review_length = df[text_col].astype(str).str.len().mean()
            avg_rating = df[rating_col].mean() if rating_col and rating_col in df.columns else None
            
            # 5. 텍스트 전처리
            df['cleaned_review'] = df[text_col].fillna('').astype(str)
            df = df[df['cleaned_review'].str.len() > 0]  # 빈 리뷰 제거
            
            if len(df) == 0:
                print("❌ 유효한 리뷰가 없습니다.")
                return None
            
            # 6. 형태소 분석
            df = self.morphological_analyzer.tokenize_dataframe(df, 'cleaned_review')
            
            if len(df) == 0:
                print("❌ 토큰화 후 유효한 데이터가 없습니다.")
                return None
            
            # 7. 감성 분석
            if rating_col and rating_col in df.columns:
                df = self.sentiment_analyzer.create_sentiment_labels(
                    df, method='rating', rating_column=rating_col
                )
            else:
                # 평점이 없으면 텍스트 기반 감성 분석 시도
                df = self.sentiment_analyzer.create_sentiment_labels(
                    df, method='learned', tokens_column='tokens'
                )
            
            # 8. 키워드 추출
            texts = df['tokens_str'].tolist()
            if texts:
                self.keyword_extractor.extract_auto_stopwords(texts)
                
                # TF-IDF 키워드
                tfidf_keywords = self.keyword_extractor.extract_keywords_tfidf(df)
                top_keywords = [word for word, score in tfidf_keywords[:10]]
                
                # 감성별 키워드
                sentiment_keywords = self.keyword_extractor.extract_sentiment_keywords(df)
            else:
                top_keywords = []
                sentiment_keywords = {}
            
            # 9. 감성 분포 계산
            sentiment_counts = df['sentiment'].value_counts()
            total_sentiment = len(df)
            
            positive_ratio = sentiment_counts.get('positive', 0) / total_sentiment * 100
            negative_ratio = sentiment_counts.get('negative', 0) / total_sentiment * 100
            neutral_ratio = sentiment_counts.get('neutral', 0) / total_sentiment * 100
            
            # 10. 상품명 추출
            extracted_name = self.extract_product_name_from_filename(file_path.name)
            
            # 파일에서 상품명 가져오기 시도
            if product_col and product_col in df.columns:
                product_names = df[product_col].dropna().unique()
                if len(product_names) > 0:
                    file_product_name = product_names[0]
                else:
                    file_product_name = extracted_name
            else:
                file_product_name = extracted_name
            
            # 11. 결과 구성
            result = {
                'file_name': file_path.name,
                'extracted_name': extracted_name,
                'product_name': file_product_name,
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
                'all_keywords': ', '.join(top_keywords),
                'text_column': text_col,
                'rating_column': rating_col,
                'product_column': product_col
            }
            
            print(f"✅ 분석 완료: {file_product_name} ({len(df)}개 리뷰)")
            return result
            
        except Exception as e:
            print(f"❌ 분석 실패 {file_path.name}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def run_batch_analysis(self) -> List[Dict]:
        """일괄 분석 실행"""
        print("\n" + "="*60)
        print("🔄 네이버 상품 리뷰 일괄 분석 시작")
        print("="*60)
        
        # 리뷰 파일 목록 수집
        review_files = list(self.reviews_dir.glob("*.csv"))
        print(f"📁 발견된 리뷰 파일: {len(review_files)}개")
        
        if not review_files:
            print("❌ 리뷰 파일이 없습니다.")
            return []
        
        # 각 파일 분석
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
        output_file = self.output_dir / f"naver_batch_analysis_{timestamp}.xlsx"
        
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            
            # 1. 분석 개요 및 메인 요약 테이블 (통합)
            self._create_comprehensive_summary_sheet(writer)
            
            # 2. 상세 분석 결과
            self._create_detailed_analysis_sheet(writer)
            
            # 3. 키워드 분석
            self._create_keyword_analysis_sheet(writer)
            
            # 4. 통계 요약
            self._create_statistics_summary_sheet(writer)
        
        print(f"\n🎉 종합 분석 리포트 생성 완료!")
        print(f"📄 파일 경로: {output_file}")
        
        return str(output_file)
    
    def _create_comprehensive_summary_sheet(self, writer):
        """종합 요약 시트 생성 (분석 개요 + 상품 요약)"""
        df_results = pd.DataFrame(self.analysis_results)
        
        # 분석 대상 자료 개요 및 상품 요약을 하나의 시트에 통합
        all_data = []
        
        # === 1. 분석 개요 ===
        all_data.extend([
            ['=== 네이버 리뷰 종합 분석 리포트 ===', '', '', '', '', ''],
            ['분석 일시', datetime.now().strftime("%Y-%m-%d %H:%M:%S"), '', '', '', ''],
            ['분석 디렉토리', str(self.reviews_dir), '', '', '', ''],
            ['', '', '', '', '', ''],
            ['[분석 대상 자료 개요]', '', '', '', '', ''],
            ['총 분석 파일 수', len(df_results), '개', '', '', ''],
            ['총 리뷰 수', df_results['total_reviews'].sum(), '개', '', '', ''],
            ['유효 리뷰 수', df_results['valid_reviews'].sum(), '개', '', '', ''],
            ['제거된 리뷰 수', df_results['total_reviews'].sum() - df_results['valid_reviews'].sum(), '개', '', '', ''],
            ['평균 리뷰 수 (상품당)', f"{df_results['total_reviews'].mean():.1f}", '개', '', '', ''],
            ['평균 유효 리뷰 수 (상품당)', f"{df_results['valid_reviews'].mean():.1f}", '개', '', '', ''],
            ['', '', '', '', '', ''],
            ['[전체 감성 분포]', '', '', '', '', ''],
            ['전체 긍정 리뷰', df_results['positive_count'].sum(), '개', f"({df_results['positive_ratio'].mean():.1f}%)", '', ''],
            ['전체 부정 리뷰', df_results['negative_count'].sum(), '개', f"({df_results['negative_ratio'].mean():.1f}%)", '', ''],
            ['전체 중립 리뷰', df_results['neutral_count'].sum(), '개', f"({df_results['neutral_ratio'].mean():.1f}%)", '', ''],
            ['', '', '', '', '', ''],
            ['[평점 정보]', '', '', '', '', '']
        ])
        
        # 평점 정보가 있는 상품들의 통계
        rated_products = df_results[df_results['avg_rating'].notna()]
        if len(rated_products) > 0:
            all_data.extend([
                ['평점 정보 있는 상품 수', len(rated_products), '개', '', '', ''],
                ['전체 평균 평점', f"{rated_products['avg_rating'].mean():.2f}", '점', '5점 만점', '', ''],
                ['최고 평점', f"{rated_products['avg_rating'].max():.2f}", '점', '', '', ''],
                ['최저 평점', f"{rated_products['avg_rating'].min():.2f}", '점', '', '', '']
            ])
        else:
            all_data.append(['평점 정보', '없음', '', '', '', ''])
        
        all_data.extend([
            ['', '', '', '', '', ''],
            ['[상품별 요약 - 긍정비율 순]', '', '', '', '', ''],
            ['순위', '상품명', '총리뷰수', '유효리뷰수', '평균평점', '긍정비율(%)']
        ])
        
        # 상품 요약 (긍정비율 순으로 정렬)
        summary_df = df_results.sort_values('positive_ratio', ascending=False)
        
        for i, (idx, row) in enumerate(summary_df.iterrows(), 1):
            product_name = row['product_name'][:40] + "..." if len(row['product_name']) > 40 else row['product_name']
            avg_rating = f"{row['avg_rating']:.2f}" if pd.notna(row['avg_rating']) else '없음'
            all_data.append([
                i,
                product_name,
                row['total_reviews'],
                row['valid_reviews'],
                avg_rating,
                f"{row['positive_ratio']:.1f}%"
            ])
        
        # DataFrame으로 변환하여 저장
        comprehensive_df = pd.DataFrame(all_data, columns=['구분', '항목', '값', '추가정보1', '추가정보2', '추가정보3'])
        comprehensive_df.to_excel(writer, sheet_name='1_종합요약', index=False)
        
        # 워크시트 스타일링
        worksheet = writer.sheets['1_종합요약']
        worksheet.column_dimensions['A'].width = 15
        worksheet.column_dimensions['B'].width = 45
        worksheet.column_dimensions['C'].width = 12
        worksheet.column_dimensions['D'].width = 15
        worksheet.column_dimensions['E'].width = 15
        worksheet.column_dimensions['F'].width = 15
        
        print("✅ 종합 요약 시트 생성 완료")
    
    def _create_detailed_analysis_sheet(self, writer):
        """상세 분석 시트 생성"""
        df_results = pd.DataFrame(self.analysis_results)
        
        detailed_columns = [
            'product_name', 'file_name', 'total_reviews', 'valid_reviews', 
            'avg_review_length', 'avg_rating', 'positive_count', 'negative_count', 
            'neutral_count', 'positive_ratio', 'negative_ratio', 'neutral_ratio',
            'all_keywords', 'text_column', 'rating_column'
        ]
        
        detailed_df = df_results[detailed_columns].copy()
        detailed_df.columns = [
            '상품명', '파일명', '총 리뷰수', '유효 리뷰수', '평균 리뷰길이',
            '평균평점', '긍정수', '부정수', '중립수', '긍정비율(%)', 
            '부정비율(%)', '중립비율(%)', '전체키워드', '텍스트컬럼', '평점컬럼'
        ]
        
        detailed_df.to_excel(writer, sheet_name='2_상세분석', index=False)
        print("✅ 상세 분석 시트 생성 완료")
    
    def _create_keyword_analysis_sheet(self, writer):
        """키워드 분석 시트 생성"""
        df_results = pd.DataFrame(self.analysis_results)
        
        # 상품별 키워드 분석
        keyword_data = []
        
        for idx, row in df_results.iterrows():
            keyword_data.append({
                '상품명': row['product_name'],
                '총 리뷰수': row['total_reviews'],
                '긍정비율': row['positive_ratio'],
                '주요 키워드': row['top_keywords'],
                '긍정 키워드': row['positive_keywords'],
                '부정 키워드': row['negative_keywords']
            })
        
        keyword_df = pd.DataFrame(keyword_data)
        keyword_df = keyword_df.sort_values('긍정비율', ascending=False)
        keyword_df.to_excel(writer, sheet_name='3_키워드분석', index=False)
        print("✅ 키워드 분석 시트 생성 완료")
    
    def _create_statistics_summary_sheet(self, writer):
        """통계 요약 시트 생성"""
        df_results = pd.DataFrame(self.analysis_results)
        
        stats_data = [
            ['전체 통계', '', ''],
            ['분석된 상품 수', len(df_results), '개'],
            ['총 리뷰 수', df_results['total_reviews'].sum(), '개'],
            ['총 유효 리뷰 수', df_results['valid_reviews'].sum(), '개'],
            ['평균 리뷰 수 (상품당)', df_results['total_reviews'].mean(), '개'],
            ['전체 평균 긍정비율', df_results['positive_ratio'].mean(), '%'],
            ['전체 평균 부정비율', df_results['negative_ratio'].mean(), '%'],
            ['', '', ''],
            ['TOP 5 긍정 상품', '', ''],
        ]
        
        # TOP 5 긍정 상품
        top_positive = df_results.nlargest(5, 'positive_ratio')
        for idx, row in top_positive.iterrows():
            stats_data.append([
                row['product_name'][:30], 
                f"{row['positive_ratio']}%", 
                f"{row['total_reviews']}개 리뷰"
            ])
        
        stats_data.extend([
            ['', '', ''],
            ['TOP 5 부정 상품', '', ''],
        ])
        
        # TOP 5 부정 상품
        top_negative = df_results.nlargest(5, 'negative_ratio')
        for idx, row in top_negative.iterrows():
            stats_data.append([
                row['product_name'][:30], 
                f"{row['negative_ratio']}%", 
                f"{row['total_reviews']}개 리뷰"
            ])
        
        stats_df = pd.DataFrame(stats_data, columns=['항목', '값', '단위'])
        stats_df.to_excel(writer, sheet_name='4_통계요약', index=False)
        print("✅ 통계 요약 시트 생성 완료")
    
    def print_summary(self):
        """분석 결과 요약 출력"""
        if not self.analysis_results:
            return
        
        df_results = pd.DataFrame(self.analysis_results)
        
        print("\n" + "="*60)
        print("📊 네이버 상품 리뷰 분석 결과 요약")
        print("="*60)
        
        print(f"📁 분석 디렉토리: {self.reviews_dir}")
        print(f"🔢 분석된 상품 수: {len(df_results)}개")
        print(f"📝 총 리뷰 수: {df_results['total_reviews'].sum():,}개")
        print(f"📝 유효 리뷰 수: {df_results['valid_reviews'].sum():,}개")
        print(f"🗑️ 제거된 리뷰 수: {df_results['total_reviews'].sum() - df_results['valid_reviews'].sum():,}개")
        print(f"⭐ 전체 평균 긍정비율: {df_results['positive_ratio'].mean():.1f}%")
        print(f"⭐ 전체 평균 부정비율: {df_results['negative_ratio'].mean():.1f}%")
        
        # 평점 정보가 있는 상품들의 평균
        rated_products = df_results[df_results['avg_rating'].notna()]
        if len(rated_products) > 0:
            print(f"⭐ 평균 평점: {rated_products['avg_rating'].mean():.2f} (평점 정보 있는 {len(rated_products)}개 상품)")
        else:
            print(f"⭐ 평점 정보: 없음")
        
        # 컬럼 정보 요약
        unique_text_cols = df_results['text_column'].unique()
        unique_rating_cols = df_results['rating_column'].unique()
        print(f"\n📋 데이터 구조:")
        print(f"   텍스트 컬럼: {list(unique_text_cols)}")
        print(f"   평점 컬럼: {list(unique_rating_cols)}")
        
        print(f"\n🏆 긍정비율 TOP 5 상품:")
        top_products = df_results.nlargest(5, 'positive_ratio')
        for idx, row in top_products.iterrows():
            print(f"   {row['product_name'][:40]:40} "
                  f"긍정: {row['positive_ratio']}% ({row['total_reviews']}개 리뷰)")
        
        print(f"\n📉 부정비율 TOP 5 상품:")
        bottom_products = df_results.nlargest(5, 'negative_ratio')
        for idx, row in bottom_products.iterrows():
            print(f"   {row['product_name'][:40]:40} "
                  f"부정: {row['negative_ratio']}% ({row['total_reviews']}개 리뷰)")
        
        print("="*60)


def main():
    """메인 실행 함수"""
    # 경로 설정
    reviews_dir = "/Users/brich/Desktop/marketcrawler/crawler/operators/output/naver_reviews1"
    output_dir = "/Users/brich/Desktop/marketcrawler/output"
    
    # 분석기 초기화 및 실행
    analyzer = NaverReviewAnalyzer(reviews_dir, output_dir)
    
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