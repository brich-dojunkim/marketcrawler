# product_info_analyzer.py
"""
네이버 상품 정보 키워드 및 감성 분석기
상품명, 가격, 평점 등을 분석하여 키워드 추출 및 감성 분석 수행
"""

import pandas as pd
import numpy as np
import re
from typing import Dict, List, Tuple, Optional
from collections import Counter
from datetime import datetime
from pathlib import Path
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

warnings.filterwarnings('ignore')

class ProductInfoAnalyzer:
    """상품 정보 분석 클래스"""
    
    def __init__(self, output_dir: str = "/Users/brich/Desktop/marketcrawler/output"):
        """
        Args:
            output_dir: 결과 저장 디렉토리
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 분석 결과 저장용
        self.analysis_results = []
        
        # 동적으로 추출될 데이터
        self.extracted_brands = set()
        self.extracted_keywords = []
        self.category_clusters = {}
        
        print("🚀 상품 정보 분석기 초기화 완료")
    
    def preprocess_product_name(self, product_name: str) -> str:
        """상품명 전처리"""
        # 특수문자 제거하고 공백으로 변경
        cleaned = re.sub(r'[^\w\s가-힣]', ' ', product_name)
        # 연속된 공백을 하나로
        cleaned = re.sub(r'\s+', ' ', cleaned)
        return cleaned.strip()
    
    def extract_brands_from_data(self, df: pd.DataFrame) -> List[str]:
        """데이터에서 브랜드명 자동 추출"""
        print("🏷️ 브랜드명 자동 추출 중...")
        
        all_words = []
        for name in df['name']:
            # 괄호 안의 내용 추출 (브랜드일 가능성 높음)
            bracket_matches = re.findall(r'\(([^)]+)\)', name)
            all_words.extend(bracket_matches)
            
            # 첫 번째 단어들 (브랜드일 가능성)
            words = self.preprocess_product_name(name).split()
            if words:
                all_words.append(words[0])
        
        # 빈도 계산해서 자주 나오는 브랜드들 추출
        word_counts = Counter(all_words)
        
        # 최소 2번 이상 나오고, 한글이 포함된 단어들을 브랜드로 인정
        brands = []
        for word, count in word_counts.items():
            if count >= 2 and len(word) >= 2 and re.search(r'[가-힣]', word):
                brands.append(word)
        
        self.extracted_brands = set(brands)
        print(f"✅ 추출된 브랜드: {len(brands)}개 - {brands[:10]}")
        return brands
    
    def extract_keywords_tfidf(self, df: pd.DataFrame, max_features: int = 50) -> List[Tuple[str, float]]:
        """TF-IDF 기반 키워드 자동 추출"""
        print("📊 TF-IDF 키워드 자동 추출 중...")
        
        # 상품명 전처리
        processed_names = [self.preprocess_product_name(name) for name in df['name']]
        
        # 한글 단어만 추출하는 커스텀 토크나이저
        def korean_tokenizer(text):
            # 한글 2글자 이상 단어만 추출
            words = re.findall(r'[가-힣]{2,}', text)
            return words
        
        try:
            vectorizer = TfidfVectorizer(
                tokenizer=korean_tokenizer,
                max_features=max_features,
                min_df=2,  # 최소 2번 이상 나타나는 단어만
                lowercase=False
            )
            
            tfidf_matrix = vectorizer.fit_transform(processed_names)
            feature_names = vectorizer.get_feature_names_out()
            
            # 전체 TF-IDF 점수 합계로 중요도 계산
            scores = tfidf_matrix.sum(axis=0).A1
            keyword_scores = list(zip(feature_names, scores))
            keyword_scores.sort(key=lambda x: x[1], reverse=True)
            
            self.extracted_keywords = keyword_scores
            print(f"✅ 추출된 키워드: {len(keyword_scores)}개")
            
            return keyword_scores
            
        except Exception as e:
            print(f"❌ TF-IDF 키워드 추출 실패: {e}")
            return []
    
    def cluster_products_by_similarity(self, df: pd.DataFrame, n_clusters: int = 5) -> Dict[int, List[str]]:
        """상품 유사도 기반 클러스터링으로 카테고리 자동 분류"""
        print(f"🎯 상품 클러스터링 ({n_clusters}개 클러스터)...")
        
        try:
            # 상품명 전처리
            processed_names = [self.preprocess_product_name(name) for name in df['name']]
            
            def korean_tokenizer(text):
                return re.findall(r'[가-힣]{2,}', text)
            
            vectorizer = TfidfVectorizer(
                tokenizer=korean_tokenizer,
                max_features=100,
                min_df=1,
                lowercase=False
            )
            
            tfidf_matrix = vectorizer.fit_transform(processed_names)
            
            # K-means 클러스터링
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(tfidf_matrix)
            
            # 각 클러스터별 대표 키워드 추출
            feature_names = vectorizer.get_feature_names_out()
            clusters = {}
            
            for i in range(n_clusters):
                cluster_center = kmeans.cluster_centers_[i]
                top_indices = cluster_center.argsort()[-5:][::-1]  # 상위 5개 키워드
                cluster_keywords = [feature_names[idx] for idx in top_indices]
                
                # 해당 클러스터의 상품들
                cluster_products = df[cluster_labels == i]['name'].tolist()
                
                clusters[i] = {
                    'keywords': cluster_keywords,
                    'products': cluster_products,
                    'count': len(cluster_products)
                }
            
            self.category_clusters = clusters
            
            print(f"✅ 클러스터링 완료:")
            for i, cluster_info in clusters.items():
                print(f"   클러스터 {i}: {cluster_info['keywords'][:3]} ({cluster_info['count']}개 상품)")
            
            return clusters
            
        except Exception as e:
            print(f"❌ 클러스터링 실패: {e}")
            return {}
    
    def calculate_sentiment_score(self, row: pd.Series) -> Dict[str, float]:
        """다차원 감성 점수 계산"""
        scores = {}
        
        # 1. 평점 기반 점수 (0-100)
        rating = row.get('rating', 0)
        scores['rating_score'] = (rating / 5.0) * 100 if rating > 0 else 0
        
        # 2. 리뷰수 기반 인기도 점수 (로그 스케일)
        review_count = row.get('review_count', 0)
        if review_count > 0:
            # 리뷰수를 로그 변환하여 0-100 스케일로 정규화
            log_reviews = np.log1p(review_count)
            max_log_reviews = np.log1p(20000)  # 가정: 최대 리뷰수 20,000
            scores['popularity_score'] = min((log_reviews / max_log_reviews) * 100, 100)
        else:
            scores['popularity_score'] = 0
        
        # 3. 할인율 기반 가성비 점수
        discount = row.get('discount_percentage', 0)
        scores['discount_score'] = min(discount * 2, 100)  # 50% 할인이면 100점
        
        # 4. 가격 경쟁력 점수 (낮은 가격이 더 좋음)
        price = row.get('price', 0)
        if price > 0:
            # 가격대별 상대적 점수 (추후 전체 데이터 기반으로 정규화)
            scores['price_competitiveness'] = 50  # 기본값, 추후 조정
        else:
            scores['price_competitiveness'] = 0
        
        # 5. 베스트 상품 보너스
        is_best = row.get('is_best', False)
        scores['best_bonus'] = 20 if is_best else 0
        
        # 6. 종합 점수 (가중 평균)
        weights = {
            'rating_score': 0.3,
            'popularity_score': 0.25,
            'discount_score': 0.2,
            'price_competitiveness': 0.15,
            'best_bonus': 0.1
        }
        
        total_score = sum(scores[key] * weights[key] for key in weights.keys())
        scores['total_score'] = round(total_score, 1)
        
        return scores
    
    def extract_product_features(self, product_name: str) -> Dict[str, List[str]]:
        """상품명에서 특징 추출"""
        features = {
            'volume_info': [],  # 용량/수량 정보
            'descriptive_words': [],  # 설명 단어들
            'numbers': []  # 숫자 정보
        }
        
        # 용량/수량 정보 추출
        volume_patterns = [
            r'(\d+(?:\.\d+)?)(ml|L|g|kg|개|입)',
            r'(\d+)(팩|세트|개입)'
        ]
        
        for pattern in volume_patterns:
            matches = re.findall(pattern, product_name)
            for match in matches:
                if isinstance(match, tuple):
                    features['volume_info'].append(''.join(match))
                else:
                    features['volume_info'].append(match)
        
        # 숫자 정보 추출
        numbers = re.findall(r'\d+', product_name)
        features['numbers'] = numbers
        
        # 설명 단어들 (한글 2글자 이상)
        korean_words = re.findall(r'[가-힣]{2,}', product_name)
        features['descriptive_words'] = korean_words
        
        return features
    
    def analyze_single_product(self, row: pd.Series, cluster_id: Optional[int] = None) -> Dict:
        """개별 상품 분석"""
        product_name = row['name']
        
        # 기본 정보 추출
        brand = self.extract_brand_from_name(product_name)
        features = self.extract_product_features(product_name)
        sentiment_scores = self.calculate_sentiment_score(row)
        
        # 클러스터 정보
        cluster_info = None
        if cluster_id is not None and cluster_id in self.category_clusters:
            cluster_info = self.category_clusters[cluster_id]
        
        result = {
            'product_name': product_name,
            'brand': brand,
            'cluster_id': cluster_id,
            'cluster_keywords': cluster_info['keywords'] if cluster_info else [],
            'extracted_features': features,
            'volume_info': ', '.join(features['volume_info']),
            'descriptive_words': ', '.join(features['descriptive_words'][:10]),
            'price': row.get('price', 0),
            'original_price': row.get('original_price', 0),
            'discount_percentage': row.get('discount_percentage', 0),
            'rating': row.get('rating', 0),
            'review_count': row.get('review_count', 0),
            'is_best': row.get('is_best', False),
            **sentiment_scores
        }
        
        return result
    
    def extract_brand_from_name(self, product_name: str) -> str:
        """상품명에서 브랜드 추출 (동적)"""
        # 괄호 안의 브랜드명 우선
        bracket_match = re.search(r'\(([^)]+)\)', product_name)
        if bracket_match:
            brand = bracket_match.group(1)
            if brand in self.extracted_brands:
                return brand
        
        # 추출된 브랜드 목록에서 찾기
        for brand in self.extracted_brands:
            if brand in product_name:
                return brand
        
        # 첫 번째 한글 단어를 브랜드로 간주
        korean_words = re.findall(r'[가-힣]+', product_name)
        if korean_words:
            return korean_words[0]
        
        return 'Unknown'
    
    def run_comprehensive_analysis(self, csv_file_path: str) -> List[Dict]:
        """종합 분석 실행"""
        print("\n" + "="*60)
        print("🔄 상품 정보 종합 분석 시작")
        print("="*60)
        
        # 1. 데이터 로드
        try:
            df = pd.read_csv(csv_file_path)
            print(f"✅ 데이터 로드 완료: {len(df)}개 상품")
        except Exception as e:
            print(f"❌ 데이터 로드 실패: {e}")
            return []
        
        # 2. 키워드 자동 추출 (브랜드도 포함)
        self.extract_keywords_tfidf(df)
        
        # 3. 상품 클러스터링
        clusters = self.cluster_products_by_similarity(df)
        
        # 4. 클러스터 라벨 할당
        if clusters:
            processed_names = [self.preprocess_product_name(name) for name in df['name']]
            
            def korean_tokenizer(text):
                return re.findall(r'[가-힣]{2,}', text)
            
            vectorizer = TfidfVectorizer(
                tokenizer=korean_tokenizer,
                max_features=100,
                min_df=1,
                lowercase=False
            )
            
            tfidf_matrix = vectorizer.fit_transform(processed_names)
            kmeans = KMeans(n_clusters=len(clusters), random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(tfidf_matrix)
            
            df['cluster_id'] = cluster_labels
        else:
            df['cluster_id'] = 0
        
        # 5. 가격 경쟁력 점수 정규화
        if 'price' in df.columns:
            price_percentiles = df['price'].quantile([0.25, 0.75])
            df['price_competitiveness'] = df['price'].apply(
                lambda x: 75 if x <= price_percentiles[0.25] else 
                         25 if x >= price_percentiles[0.75] else 50
            )
        
        # 6. 개별 상품 분석
        print("🔍 개별 상품 분석 중...")
        self.analysis_results = []
        
        for idx, row in df.iterrows():
            cluster_id = row.get('cluster_id', 0)
            result = self.analyze_single_product(row, cluster_id)
            self.analysis_results.append(result)
        
        print(f"✅ 분석 완료: {len(self.analysis_results)}개 상품")
        return self.analysis_results
    
    def create_comprehensive_report(self) -> str:
        """종합 분석 리포트 생성 (단일 시트)"""
        if not self.analysis_results:
            print("❌ 분석 결과가 없습니다.")
            return ""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"product_analysis_{timestamp}.xlsx"
        
        # 모든 분석 결과를 하나의 시트로 통합
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            self._create_unified_analysis_sheet(writer)
        
        print(f"\n🎉 종합 분석 리포트 생성 완료!")
        print(f"📄 파일 경로: {output_file}")
        
        return str(output_file)
    
    def _create_unified_analysis_sheet(self, writer):
        """통합 분석 시트 생성 (모든 결과를 하나의 시트에)"""
        try:
            df_results = pd.DataFrame(self.analysis_results)
            
            # 통합 리포트 데이터 구성 - 모든 행이 4개 컬럼을 가지도록 수정
            all_data = []
            
            # === 1. 분석 개요 ===
            all_data.extend([
                ['=== 상품 정보 종합 분석 리포트 ===', '', '', ''],
                ['분석 일시', datetime.now().strftime("%Y-%m-%d %H:%M:%S"), '', ''],
                ['총 상품 수', str(len(df_results)), '개', ''],
                ['추출된 키워드 수', str(len(self.extracted_keywords)), '개', ''],
                ['클러스터 수', str(len(self.category_clusters)), '개', ''],
                ['', '', '', '']
            ])
            
            # === 2. 종합 통계 ===
            all_data.extend([
                ['[종합 통계]', '', '', ''],
                ['평균 종합점수', f"{df_results['total_score'].mean():.1f}", '점', '100점 만점'],
                ['평균 평점', f"{df_results['rating'].mean():.2f}", '점', '5점 만점'],
                ['평균 가격', f"{df_results['price'].mean():,.0f}", '원', ''],
                ['평균 할인율', f"{df_results['discount_percentage'].mean():.1f}", '%', ''],
                ['총 리뷰수', f"{df_results['review_count'].sum():,}", '개', ''],
                ['베스트 상품수', str(df_results['is_best'].sum()), '개', ''],
                ['', '', '', '']
            ])
            
            # === 3. TOP 상품 (종합점수 기준) ===
            all_data.extend([
                ['[TOP 10 상품 - 종합점수 기준]', '', '', ''],
                ['순위', '상품명', '종합점수', '평점']
            ])
            
            top_products = df_results.nlargest(10, 'total_score')
            for i, (idx, row) in enumerate(top_products.iterrows(), 1):
                product_name = row['product_name'][:50] + "..." if len(row['product_name']) > 50 else row['product_name']
                all_data.append([str(i), product_name, f"{row['total_score']:.1f}", f"{row['rating']:.2f}"])
            
            all_data.append(['', '', '', ''])
            
            # === 4. 키워드 분석 (브랜드 포함) ===
            all_data.extend([
                ['[키워드 분석]', '', '', ''],
                ['순위', '키워드', 'TF-IDF점수', '중요도']
            ])
            
            # 키워드 분석 (브랜드도 키워드로 포함)
            if self.extracted_keywords:
                for i, (keyword, score) in enumerate(self.extracted_keywords[:20], 1):
                    importance = "매우중요" if score > np.percentile([s for _, s in self.extracted_keywords], 90) else \
                               "중요" if score > np.percentile([s for _, s in self.extracted_keywords], 70) else "보통"
                    all_data.append([
                        str(i),
                        keyword,
                        f"{score:.3f}",
                        importance
                    ])
            
            all_data.append(['', '', '', ''])
            
            # === 5. 클러스터 분석 ===
            if self.category_clusters:
                all_data.extend([
                    ['[상품 클러스터 분석]', '', '', ''],
                    ['클러스터ID', '대표키워드', '상품수', '평균점수']
                ])
                
                for cluster_id, cluster_info in self.category_clusters.items():
                    cluster_products = df_results[df_results['cluster_id'] == cluster_id]
                    if len(cluster_products) > 0:
                        keywords_str = ', '.join(cluster_info['keywords'][:3])
                        avg_score = cluster_products['total_score'].mean()
                        all_data.append([
                            f"클러스터 {cluster_id}",
                            keywords_str,
                            f"{len(cluster_products)}개",
                            f"{avg_score:.1f}점"
                        ])
            
            all_data.append(['', '', '', ''])
            
            # === 6. 가격대별 분석 ===
            all_data.extend([
                ['[가격대별 분석]', '', '', ''],
                ['가격대', '상품수', '평균점수', '평균평점']
            ])
            
            # 가격대 구간 설정
            price_ranges = [
                (0, 10000, '1만원 미만'),
                (10000, 20000, '1-2만원'),
                (20000, 30000, '2-3만원'),
                (30000, float('inf'), '3만원 이상')
            ]
            
            for min_price, max_price, range_name in price_ranges:
                range_products = df_results[
                    (df_results['price'] >= min_price) & 
                    (df_results['price'] < max_price)
                ]
                
                if len(range_products) > 0:
                    all_data.append([
                        range_name,
                        f"{len(range_products)}개",
                        f"{range_products['total_score'].mean():.1f}점",
                        f"{range_products['rating'].mean():.2f}점"
                    ])
            
            all_data.append(['', '', '', ''])
            
            # === 7. 할인율 분석 ===
            all_data.extend([
                ['[할인율별 분석]', '', '', ''],
                ['할인구간', '상품수', '평균점수', '평균평점']
            ])
            
            discount_ranges = [
                (0, 20, '20% 미만'),
                (20, 40, '20-40%'),
                (40, 60, '40-60%'),
                (60, 100, '60% 이상')
            ]
            
            for min_disc, max_disc, range_name in discount_ranges:
                range_products = df_results[
                    (df_results['discount_percentage'] >= min_disc) & 
                    (df_results['discount_percentage'] < max_disc)
                ]
                
                if len(range_products) > 0:
                    all_data.append([
                        range_name,
                        f"{len(range_products)}개",
                        f"{range_products['total_score'].mean():.1f}점",
                        f"{range_products['rating'].mean():.2f}점"
                    ])
            
            all_data.append(['', '', '', ''])
            
            # === 8. 상세 상품 리스트 (상위 20개) - 컬럼 수 조정 ===
            all_data.extend([
                ['[상세 상품 정보 - TOP 20]', '', '', ''],
                ['순위 | 상품명', '브랜드 | 가격', '평점 | 할인율', '종합점수']
            ])
            
            top_20_products = df_results.nlargest(20, 'total_score')
            for i, (idx, row) in enumerate(top_20_products.iterrows(), 1):
                product_name = row['product_name'][:30] + "..." if len(row['product_name']) > 30 else row['product_name']
                all_data.append([
                    f"{i}. {product_name}",
                    f"{row['brand']} | {row['price']:,}원",
                    f"{row['rating']:.2f} | {row['discount_percentage']:.0f}%",
                    f"{row['total_score']:.1f}점"
                ])
            
            # DataFrame으로 변환하여 Excel에 저장
            unified_df = pd.DataFrame(all_data, columns=['구분', '항목', '값', '세부정보'])
            unified_df.to_excel(writer, sheet_name='종합분석리포트', index=False)
            
            # 워크시트 스타일링
            worksheet = writer.sheets['종합분석리포트']
            
            # 컬럼 폭 조정
            worksheet.column_dimensions['A'].width = 35
            worksheet.column_dimensions['B'].width = 45
            worksheet.column_dimensions['C'].width = 25
            worksheet.column_dimensions['D'].width = 35
            
            print("✅ 통합 분석 시트 생성 완료")
            
        except Exception as e:
            print(f"❌ 시트 생성 중 오류: {e}")
            # 오류 발생 시 간단한 시트라도 생성
            simple_df = pd.DataFrame([['오류 발생', str(e), '', '']], 
                                   columns=['구분', '항목', '값', '세부정보'])
            simple_df.to_excel(writer, sheet_name='종합분석리포트', index=False)
    
    def print_analysis_summary(self):
        """분석 결과 요약 출력"""
        if not self.analysis_results:
            return
        
        df_results = pd.DataFrame(self.analysis_results)
        
        print("\n" + "="*60)
        print("📊 상품 정보 분석 결과 요약")
        print("="*60)
        
        print(f"🔢 총 상품 수: {len(df_results)}개")
        print(f"🏷️ 추출된 브랜드 수: {len(self.extracted_brands)}개")
        print(f"🔤 추출된 키워드 수: {len(self.extracted_keywords)}개")
        print(f"🎯 클러스터 수: {len(self.category_clusters)}개")
        
        print(f"\n📈 종합 점수 통계:")
        print(f"   평균 점수: {df_results['total_score'].mean():.1f}")
        print(f"   최고 점수: {df_results['total_score'].max():.1f}")
        print(f"   최저 점수: {df_results['total_score'].min():.1f}")
        
        print(f"\n🏆 TOP 5 상품 (종합점수 기준):")
        top_products = df_results.nlargest(5, 'total_score')
        for idx, row in top_products.iterrows():
            print(f"   {row['product_name'][:50]:50} 점수: {row['total_score']}")
        
        print(f"\n🏷️ 브랜드별 상품 수:")
        brand_counts = df_results['brand'].value_counts()
        for brand, count in brand_counts.head(5).items():
            print(f"   {brand}: {count}개")
        
        print("="*60)


def main():
    """메인 실행 함수"""
    # CSV 파일 경로 설정
    csv_file_path = "/Users/brich/Desktop/marketcrawler/crawler/operators/output/naver_brandstore_20250721_123052.csv"  # 실제 파일 경로로 변경
    output_dir = "/Users/brich/Desktop/marketcrawler/output"
    
    # 분석기 초기화
    analyzer = ProductInfoAnalyzer(output_dir)
    
    # 종합 분석 실행
    results = analyzer.run_comprehensive_analysis(csv_file_path)
    
    if results:
        # 리포트 생성
        output_file = analyzer.create_comprehensive_report()
        
        # 결과 요약 출력
        analyzer.print_analysis_summary()
        
        print(f"\n🎉 분석 완료! 결과 파일: {output_file}")
    else:
        print("❌ 분석할 데이터가 없습니다.")


if __name__ == "__main__":
    main()