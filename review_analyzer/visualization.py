import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import numpy as np
from wordcloud import WordCloud
import matplotlib.font_manager as fm
import platform
import os
import networkx as nx
from matplotlib.patches import Rectangle

# 한글 폰트 설정 함수
def setup_korean_font():
    """운영체제별 한글 폰트 설정"""
    system = platform.system()
    
    if system == 'Windows':
        font_name = 'Malgun Gothic'
        try:
            plt.rcParams['font.family'] = font_name
        except:
            fonts = ['Malgun Gothic', 'Microsoft YaHei', 'SimHei']
            for font in fonts:
                try:
                    plt.rcParams['font.family'] = font
                    break
                except:
                    continue
    
    elif system == 'Darwin':  # macOS
        font_name = 'AppleGothic'
        try:
            plt.rcParams['font.family'] = font_name
        except:
            fonts = ['AppleGothic', 'Arial Unicode MS']
            for font in fonts:
                try:
                    plt.rcParams['font.family'] = font
                    break
                except:
                    continue
    
    else:  # Linux
        fonts = ['Noto Sans CJK KR', 'DejaVu Sans', 'NanumGothic']
        for font in fonts:
            try:
                plt.rcParams['font.family'] = font
                break
            except:
                continue
    
    plt.rcParams['axes.unicode_minus'] = False
    print(f"한글 폰트 설정 완료: {plt.rcParams['font.family']}")

def find_korean_font():
    """시스템에서 한글 폰트 경로 찾기"""
    system = platform.system()
    font_path = None
    
    if system == 'Windows':
        font_paths = [
            'C:/Windows/Fonts/malgun.ttf',
            'C:/Windows/Fonts/gulim.ttc',
            'C:/Windows/Fonts/batang.ttc'
        ]
    elif system == 'Darwin':  # macOS
        font_paths = [
            '/System/Library/Fonts/AppleGothic.ttf',
            '/Library/Fonts/Arial Unicode.ttf'
        ]
    else:  # Linux
        font_paths = [
            '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc',
            '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf'
        ]
    
    for path in font_paths:
        if os.path.exists(path):
            font_path = path
            break
    
    return font_path

def load_keyword_data(file_path):
    """Excel 파일에서 키워드 데이터 로드"""
    all_sheets = pd.read_excel(file_path, sheet_name=None)
    
    brand_data = {}
    
    for sheet_name, df in all_sheets.items():
        keywords = {
            'positive': [],
            'negative': [],
            'neutral': []
        }
        
        for _, row in df.iterrows():
            if row['항목'] == '긍정':
                keywords['positive'] = row['값'].split(', ')
            elif row['항목'] == '부정':
                keywords['negative'] = row['값'].split(', ')
            elif row['항목'] == '중립':
                keywords['neutral'] = row['값'].split(', ')
        
        brand_data[sheet_name] = keywords
    
    return brand_data

def create_product_network_graph(brand_name, keywords, save_path=None):
    """제품별 키워드 네트워크 그래프 생성"""
    G = nx.Graph()
    
    # 키워드와 감성 정보를 노드로 추가
    all_keywords = []
    keyword_sentiment = {}
    
    for sentiment, word_list in keywords.items():
        for word in word_list[:8]:  # 상위 8개만
            all_keywords.append(word)
            keyword_sentiment[word] = sentiment
            G.add_node(word, sentiment=sentiment)
    
    # 같은 감성의 키워드들을 연결 (간단한 연결 규칙)
    for sentiment, word_list in keywords.items():
        words = word_list[:5]  # 상위 5개만
        for i in range(len(words)):
            for j in range(i+1, len(words)):
                G.add_edge(words[i], words[j])
    
    # 시각화
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, k=3, iterations=50)
    
    # 감성별 색상 설정
    colors = {'positive': '#4CAF50', 'negative': '#F44336', 'neutral': '#9E9E9E'}
    node_colors = [colors[keyword_sentiment.get(node, 'neutral')] for node in G.nodes()]
    
    # 네트워크 그리기
    nx.draw_networkx_edges(G, pos, alpha=0.3, width=1, edge_color='gray')
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                          node_size=1000, alpha=0.8)
    nx.draw_networkx_labels(G, pos, font_size=9, font_weight='bold')
    
    plt.title(f'{brand_name} 키워드 네트워크', fontsize=16, fontweight='bold', pad=20)
    
    # 범례 추가
    legend_elements = [plt.scatter([], [], c=color, s=100, label=sentiment) 
                      for sentiment, color in [('긍정', '#4CAF50'), ('부정', '#F44336'), ('중립', '#9E9E9E')]]
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def create_product_sentiment_analysis(brand_name, keywords, save_path=None):
    """제품별 감성 분석 시각화"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. 감성 비율 도넛차트
    sentiment_counts = [len(keywords['positive']), len(keywords['negative']), len(keywords['neutral'])]
    colors = ['#4CAF50', '#F44336', '#9E9E9E']
    labels = ['긍정', '부정', '중립']
    
    wedges, texts, autotexts = ax1.pie(sentiment_counts, labels=labels, colors=colors, 
                                      autopct='%1.1f%%', startangle=90,
                                      wedgeprops=dict(width=0.5))
    ax1.set_title(f'{brand_name}\n감성 분포', fontweight='bold', fontsize=12)
    
    # 2. 키워드 길이 분포
    pos_lengths = [len(word) for word in keywords['positive']]
    neg_lengths = [len(word) for word in keywords['negative']]
    neu_lengths = [len(word) for word in keywords['neutral']]
    
    ax2.hist([pos_lengths, neg_lengths, neu_lengths], bins=range(1, 8), 
             color=colors, alpha=0.7, label=labels)
    ax2.set_title('키워드 길이 분포', fontweight='bold')
    ax2.set_xlabel('키워드 길이')
    ax2.set_ylabel('개수')
    ax2.legend()
    
    # 3. 상위 키워드 막대차트
    top_pos = keywords['positive'][:5]
    top_neg = keywords['negative'][:5]
    
    y_pos = np.arange(len(top_pos))
    y_neg = np.arange(len(top_neg))
    
    ax3.barh(y_pos, [1]*len(top_pos), color='#4CAF50', alpha=0.7)
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(top_pos)
    ax3.set_title('상위 긍정 키워드', fontweight='bold')
    ax3.set_xlim(0, 1)
    
    # 4. 부정 키워드
    ax4.barh(y_neg, [1]*len(top_neg), color='#F44336', alpha=0.7)
    ax4.set_yticks(y_neg)
    ax4.set_yticklabels(top_neg)
    ax4.set_title('상위 부정 키워드', fontweight='bold')
    ax4.set_xlim(0, 1)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def create_product_keyword_heatmap(brand_name, keywords, save_path=None):
    """제품별 키워드 히트맵"""
    # 키워드 카테고리 정의
    categories = {
        '기능': ['기능', '운동', '측정', '알림', '통화', '음악', 'GPS'],
        '디자인': ['디자인', '화면', '색상', '크기', '무게', '스타일'],
        '성능': ['배터리', '충전', '속도', '정확도', '반응'],
        '가격': ['가격', '가성비', '비용', '할인', '저렴'],
        '사용성': ['사용', '편리', '간편', '설치', '설정', '연결'],
        '품질': ['품질', '내구성', '견고', '재질', '마감']
    }
    
    # 각 카테고리별 감성 점수 계산
    heatmap_data = []
    
    for category, cat_keywords in categories.items():
        pos_score = sum(1 for word in keywords['positive'] if any(ck in word for ck in cat_keywords))
        neg_score = sum(1 for word in keywords['negative'] if any(ck in word for ck in cat_keywords))
        neu_score = sum(1 for word in keywords['neutral'] if any(ck in word for ck in cat_keywords))
        
        total = pos_score + neg_score + neu_score
        if total > 0:
            sentiment_score = (pos_score - neg_score) / total
        else:
            sentiment_score = 0
            
        heatmap_data.append([pos_score, neg_score, neu_score, sentiment_score])
    
    # 히트맵 생성
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 카테고리별 키워드 수
    counts_data = np.array(heatmap_data)[:, :3]
    sns.heatmap(counts_data, annot=True, fmt='.0f', 
                xticklabels=['긍정', '부정', '중립'],
                yticklabels=list(categories.keys()),
                cmap='Blues', ax=ax1)
    ax1.set_title(f'{brand_name} - 카테고리별 키워드 수', fontweight='bold')
    
    # 감성 점수
    sentiment_data = counts_data[:, 0:1] - counts_data[:, 1:2]  # 긍정 - 부정
    sns.heatmap(sentiment_data, annot=True, fmt='.0f',
                xticklabels=['감성점수'],
                yticklabels=list(categories.keys()),
                cmap='RdYlGn', center=0, ax=ax2)
    ax2.set_title(f'{brand_name} - 카테고리별 감성 점수', fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def create_all_product_visualizations(brand_data, output_dir="product_analysis"):
    """모든 제품에 대한 시각화 생성"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for brand_name, keywords in brand_data.items():
        print(f"🎨 {brand_name} 시각화 생성 중...")
        
        # 1. 네트워크 그래프
        try:
            create_product_network_graph(
                brand_name, keywords, 
                save_path=f"{output_dir}/{brand_name}_network.png"
            )
        except:
            print(f"네트워크 그래프 생성 실패: {brand_name}")
        
        # 2. 감성 분석
        create_product_sentiment_analysis(
            brand_name, keywords,
            save_path=f"{output_dir}/{brand_name}_sentiment.png"
        )
        
        # 3. 키워드 히트맵
        create_product_keyword_heatmap(
            brand_name, keywords,
            save_path=f"{output_dir}/{brand_name}_heatmap.png"
        )

def main():
    """메인 실행 함수"""
    setup_korean_font()
    
    file_path = "asdf.xlsx"
    
    try:
        print("📁 데이터 로딩 중...")
        brand_data = load_keyword_data(file_path)
        
        print("🎨 제품별 시각화 생성 중...")
        create_all_product_visualizations(brand_data)
        
        print(f"\n✅ 완료! {len(brand_data)}개 제품의 시각화가 'product_analysis' 폴더에 저장되었습니다.")
        print("각 제품마다 3개의 차트가 생성됩니다:")
        print("  - *_network.png: 키워드 네트워크")
        print("  - *_sentiment.png: 감성 분석")
        print("  - *_heatmap.png: 카테고리별 히트맵")
        
        return brand_data
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        return None

if __name__ == "__main__":
    print("필요한 라이브러리:")
    print("pip install pandas matplotlib seaborn wordcloud openpyxl networkx")
    print("-" * 50)
    
    results = main()