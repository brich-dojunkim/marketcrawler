# __init__.py (프로젝트 루트)
"""
한국어 상품 리뷰 분석 시스템
쿠팡 리뷰 데이터 분석 및 엑셀 결과 출력
"""

__version__ = "1.0.0"
__author__ = "Review Analysis Team"

# 순환 import 방지를 위해 lazy import 사용
def get_main_functions():
    from .main import ReviewAnalysisSystem, main, run_with_custom_file, analyze_with_visualization
    return {
        'ReviewAnalysisSystem': ReviewAnalysisSystem,
        'main': main,
        'run_with_custom_file': run_with_custom_file,
        'analyze_with_visualization': analyze_with_visualization
    }

__all__ = [
    'ReviewAnalysisSystem',
    'main', 
    'run_with_custom_file',
    'analyze_with_visualization',
    'get_main_functions'
]