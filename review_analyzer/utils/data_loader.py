# utils/data_loader.py
"""데이터 로딩 및 컬럼 분석"""

import pandas as pd
import os
from typing import Tuple, List, Optional

def load_csv_data(file_path: str) -> Optional[pd.DataFrame]:
    """
    CSV 파일 로딩
    
    Args:
        file_path: CSV 파일 경로
        
    Returns:
        DataFrame 또는 None (오류 시)
    """
    try:
        df = pd.read_csv(file_path)
        print(f"✅ 데이터 로딩 성공: {len(df):,}개 리뷰")
        print(f"📊 컬럼 정보: {list(df.columns)}")
        return df
        
    except FileNotFoundError:
        print("❌ 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
        return None
    except Exception as e:
        print(f"❌ 데이터 로딩 중 오류: {e}")
        return None

def identify_text_columns(df: pd.DataFrame, min_avg_length: int = 10) -> List[str]:
    """
    텍스트 컬럼 식별
    
    Args:
        df: 분석할 DataFrame
        min_avg_length: 텍스트 컬럼으로 인정할 최소 평균 길이
        
    Returns:
        텍스트 컬럼 리스트
    """
    text_columns = []
    
    for col in df.columns:
        if df[col].dtype == 'object':
            avg_length = df[col].dropna().astype(str).str.len().mean()
            if avg_length > min_avg_length:
                text_columns.append(col)
    
    return text_columns

def identify_rating_columns(df: pd.DataFrame, min_val: int = 1, max_val: int = 10) -> List[str]:
    """
    평점 컬럼 식별
    
    Args:
        df: 분석할 DataFrame
        min_val: 평점의 최솟값
        max_val: 평점의 최댓값
        
    Returns:
        평점 컬럼 리스트
    """
    rating_columns = []
    
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            col_min = df[col].min()
            col_max = df[col].max()
            if min_val <= col_min and col_max <= max_val:
                rating_columns.append(col)
    
    return rating_columns

def identify_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    데이터 컬럼 분석 및 리뷰/평점 컬럼 식별
    
    Args:
        df: 분석할 DataFrame
        
    Returns:
        (텍스트 컬럼 리스트, 평점 컬럼 리스트)
    """
    text_columns = identify_text_columns(df)
    rating_columns = identify_rating_columns(df)
    
    print(f"🔍 텍스트 컬럼 후보: {text_columns}")
    print(f"🔍 평점 컬럼 후보: {rating_columns}")
    
    return text_columns, rating_columns

def select_columns(text_columns: List[str], rating_columns: List[str]) -> Tuple[str, Optional[str]]:
    """
    분석에 사용할 컬럼 선택
    
    Args:
        text_columns: 텍스트 컬럼 후보 리스트
        rating_columns: 평점 컬럼 후보 리스트
        
    Returns:
        (선택된 텍스트 컬럼, 선택된 평점 컬럼)
    """
    if not text_columns:
        raise ValueError("❌ 리뷰 텍스트 컬럼을 찾을 수 없습니다.")
    
    selected_text_col = text_columns[0]
    print(f"📝 선택된 리뷰 텍스트 컬럼: '{selected_text_col}'")
    
    if rating_columns:
        selected_rating_col = rating_columns[0]
        print(f"⭐ 선택된 평점 컬럼: '{selected_rating_col}'")
    else:
        selected_rating_col = None
        print("⚠️ 평점 컬럼 없음 (텍스트 기반 분석만 수행)")
    
    return selected_text_col, selected_rating_col

def validate_data(df: pd.DataFrame, text_col: str, rating_col: Optional[str] = None) -> bool:
    """
    데이터 유효성 검증
    
    Args:
        df: DataFrame
        text_col: 텍스트 컬럼명
        rating_col: 평점 컬럼명
        
    Returns:
        유효성 검증 결과
    """
    # 텍스트 컬럼 존재 확인
    if text_col not in df.columns:
        print(f"❌ 텍스트 컬럼 '{text_col}'이 존재하지 않습니다.")
        return False
    
    # 평점 컬럼 존재 확인
    if rating_col and rating_col not in df.columns:
        print(f"❌ 평점 컬럼 '{rating_col}'이 존재하지 않습니다.")
        return False
    
    # 빈 데이터 확인
    text_data = df[text_col].dropna()
    if len(text_data) == 0:
        print(f"❌ 텍스트 컬럼 '{text_col}'에 유효한 데이터가 없습니다.")
        return False
    
    print(f"✅ 데이터 유효성 검증 완료")
    return True