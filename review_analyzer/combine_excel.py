#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
csv_output 디렉터리의 .xlsx · .csv 파일을 하나의 Excel 통합 파일로 합치는 스크립트
사용법:
    python combine_excel.py --input-dir csv_output --output-file combined.xlsx
"""
import argparse
import glob
import os
import pandas as pd

def sanitize_sheet_name(name: str, used: set[str]) -> str:
    """엑셀 시트 이름은 31자 제한·중복 금지 → 안전하게 변환"""
    base = name[:31]
    candidate = base
    counter = 1
    while candidate in used:
        # 뒤에서부터 잘라내고 ~1, ~2 … 덧붙이기
        suffix = f"~{counter}"
        candidate = f"{base[:31-len(suffix)]}{suffix}"
        counter += 1
    used.add(candidate)
    return candidate

def main(in_dir: str, out_file: str) -> None:
    files = sorted(glob.glob(os.path.join(in_dir, "*")))
    if not files:
        raise FileNotFoundError(f"❌ '{in_dir}' 폴더에 대상 파일이 없습니다.")
    used_sheet_names: set[str] = set()

    with pd.ExcelWriter(out_file, engine="openpyxl") as writer:
        for file_path in files:
            ext = os.path.splitext(file_path)[1].lower()
            sheet_name = sanitize_sheet_name(
                os.path.splitext(os.path.basename(file_path))[0],
                used_sheet_names,
            )

            # 파일 형식별 로더
            if ext == ".csv":
                df = pd.read_csv(file_path)
            elif ext in {".xlsx", ".xls"}:
                # 여러 시트를 가진 원본이라도 첫 시트만 읽도록 함
                df = pd.read_excel(file_path)
            else:
                print(f"⚠️  건너뜀(지원 안 함): {file_path}")
                continue

            df.to_excel(writer, sheet_name=sheet_name, index=False)
            print(f"▶ 병합 완료 → {sheet_name} ({len(df):,} rows)")

    print(f"✅ 통합 파일 저장 → {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="폴더 내 .csv/.xlsx 파일을 하나의 Excel 통합 파일로 합치기",
    )
    parser.add_argument(
        "--input-dir", "-i",
        default="csv_output",
        help="원본 파일들이 들어 있는 폴더 (기본: csv_output)",
    )
    parser.add_argument(
        "--output-file", "-o",
        default="combined.xlsx",
        help="출력할 통합 Excel 파일 이름 (기본: combined.xlsx)",
    )
    args = parser.parse_args()
    main(args.input_dir, args.output_file)
