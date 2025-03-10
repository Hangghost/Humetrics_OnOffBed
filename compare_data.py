#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib as mpl

# 設置中文字體支持
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']  # 優先使用這些字體
plt.rcParams['axes.unicode_minus'] = False  # 解決負號顯示問題

def load_processed_data(file_path):
    """
    載入processed_data.csv文件，該文件有特殊格式
    """
    with open(file_path, 'r') as f:
        header = f.readline().strip()
        data_lines = [line.strip() for line in f if line.strip()]
    
    columns = header.split(',')
    data_dict = {col: [] for col in columns}
    
    for line in data_lines:
        parts = line.split('],[')
        for i, col in enumerate(columns):
            if i < len(parts):
                # 清理數據並轉換為數字列表
                clean_part = parts[i].replace('[', '').replace(']', '')
                if clean_part:
                    try:
                        # 將字符串轉換為數字列表
                        numbers = np.array([int(x) for x in clean_part.split() if x.strip()])
                        data_dict[col].append(numbers)
                    except ValueError:
                        data_dict[col].append(np.array([]))
                else:
                    data_dict[col].append(np.array([]))
    
    return data_dict

def load_sps_data(file_path):
    """
    載入SPS2025PA000146 CSV文件
    """
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"載入文件 {file_path} 時出錯: {e}")
        return None

def compare_processed_data(pre_data, post_data):
    """
    比較pre和post目錄中的processed_data.csv文件
    """
    results = {}
    
    # 檢查兩個數據集是否有相同的列
    all_columns = set(pre_data.keys()) | set(post_data.keys())
    
    for col in all_columns:
        if col in pre_data and col in post_data:
            # 檢查兩個數據集中該列的數組數量是否相同
            pre_len = len(pre_data[col])
            post_len = len(post_data[col])
            
            if pre_len != post_len:
                results[col] = f"數組數量不同: pre={pre_len}, post={post_len}"
                continue
            
            # 比較每個數組
            differences = []
            for i in range(pre_len):
                pre_array = pre_data[col][i]
                post_array = post_data[col][i]
                
                # 檢查數組長度
                if len(pre_array) != len(post_array):
                    differences.append(f"第{i+1}個數組長度不同: pre={len(pre_array)}, post={len(post_array)}")
                    continue
                
                # 檢查數組內容
                if not np.array_equal(pre_array, post_array):
                    # 找出不同的元素
                    diff_indices = np.where(pre_array != post_array)[0]
                    if len(diff_indices) > 0:
                        sample_diffs = [(idx, pre_array[idx], post_array[idx]) 
                                       for idx in diff_indices[:min(5, len(diff_indices))]]
                        differences.append(f"第{i+1}個數組內容不同，樣本差異: {sample_diffs}")
            
            if differences:
                results[col] = differences
            else:
                results[col] = "完全相同"
        else:
            if col in pre_data:
                results[col] = "僅存在於pre數據中"
            else:
                results[col] = "僅存在於post數據中"
    
    return results

def compare_bed_status(pre_df, post_df):
    """
    比較SPS2025PA000146文件中的Bed_Status欄位
    """
    if pre_df is None or post_df is None:
        return "無法比較，至少有一個文件無法載入"
    
    if 'Bed_Status' not in pre_df.columns or 'Bed_Status' not in post_df.columns:
        return "至少有一個文件中沒有Bed_Status欄位"
    
    # 確保兩個DataFrame具有相同的時間戳
    pre_df['created_at'] = pd.to_datetime(pre_df['created_at'])
    post_df['created_at'] = pd.to_datetime(post_df['created_at'])
    
    # 合併數據集以進行比較
    merged_df = pd.merge(
        pre_df[['created_at', 'Bed_Status']], 
        post_df[['created_at', 'Bed_Status']], 
        on='created_at', 
        how='outer',
        suffixes=('_pre', '_post')
    )
    
    # 檢查是否有缺失值
    pre_missing = merged_df['Bed_Status_pre'].isna().sum()
    post_missing = merged_df['Bed_Status_post'].isna().sum()
    
    # 只比較兩個數據集都有的行
    common_rows = merged_df.dropna()
    
    # 檢查值是否相同
    differences = common_rows[common_rows['Bed_Status_pre'] != common_rows['Bed_Status_post']]
    
    # 計算統計信息
    total_rows = len(merged_df)
    common_rows_count = len(common_rows)
    different_values_count = len(differences)
    
    # 創建結果報告
    results = {
        "總行數": total_rows,
        "共同行數": common_rows_count,
        "僅在pre中存在的行數": pre_missing,
        "僅在post中存在的行數": post_missing,
        "值不同的行數": different_values_count,
        "差異百分比": f"{different_values_count / common_rows_count * 100:.2f}%" if common_rows_count > 0 else "N/A"
    }
    
    # 分析Bed_Status的分佈情況
    pre_status_counts = pre_df['Bed_Status'].value_counts().to_dict()
    post_status_counts = post_df['Bed_Status'].value_counts().to_dict()
    
    results["Pre Bed_Status分佈"] = pre_status_counts
    results["Post Bed_Status分佈"] = post_status_counts
    
    # 如果有差異，顯示一些樣本
    if different_values_count > 0:
        sample_diff = differences.head(10)  # 增加到10個樣本
        results["差異樣本"] = sample_diff.to_dict('records')
        
        # 分析差異的模式
        pre_to_post_transitions = differences.groupby(['Bed_Status_pre', 'Bed_Status_post']).size().reset_index(name='count')
        results["狀態轉換模式"] = pre_to_post_transitions.to_dict('records')
        
        # 繪製Bed_Status隨時間變化的圖表 - 修改為並排顯示
        plt.figure(figsize=(20, 12))
        
        # 繪製整體趨勢 - Pre數據
        plt.subplot(2, 2, 1)
        plt.plot(common_rows['created_at'], common_rows['Bed_Status_pre'], 'b-', label='Pre', alpha=0.8)
        plt.xlabel('時間')
        plt.ylabel('Bed_Status')
        plt.title('Pre數據 Bed_Status隨時間變化')
        plt.grid(True)
        plt.ylim(-0.1, 1.1)  # 設置y軸範圍，使0和1值更清晰
        
        # 繪製整體趨勢 - Post數據
        plt.subplot(2, 2, 2)
        plt.plot(common_rows['created_at'], common_rows['Bed_Status_post'], 'r-', label='Post', alpha=0.8)
        plt.xlabel('時間')
        plt.ylabel('Bed_Status')
        plt.title('Post數據 Bed_Status隨時間變化')
        plt.grid(True)
        plt.ylim(-0.1, 1.1)  # 設置y軸範圍，使0和1值更清晰
        
        # 繪製差異部分的放大視圖
        if len(differences) > 0:
            # 獲取差異時間點
            diff_times = differences['created_at'].unique()
            
            # 對於每個差異時間點，找出前後一小段時間的數據
            window_size = pd.Timedelta(minutes=5)
            highlight_data = pd.DataFrame()
            
            for diff_time in diff_times[:min(5, len(diff_times))]:  # 最多顯示5個差異區域
                window_start = diff_time - window_size
                window_end = diff_time + window_size
                window_data = common_rows[(common_rows['created_at'] >= window_start) & 
                                         (common_rows['created_at'] <= window_end)]
                highlight_data = pd.concat([highlight_data, window_data])
            
            if not highlight_data.empty:
                # 差異區域 - Pre數據
                plt.subplot(2, 2, 3)
                plt.plot(highlight_data['created_at'], highlight_data['Bed_Status_pre'], 'bo-', markersize=4)
                
                # 標記差異點
                diff_points = highlight_data.merge(differences[['created_at']], on='created_at', how='inner')
                plt.plot(diff_points['created_at'], diff_points['Bed_Status_pre'], 'bx', markersize=10, label='差異點')
                
                plt.xlabel('時間')
                plt.ylabel('Bed_Status')
                plt.title('Pre數據 差異區域放大視圖')
                plt.legend()
                plt.grid(True)
                plt.ylim(-0.1, 1.1)  # 設置y軸範圍，使0和1值更清晰
                
                # 差異區域 - Post數據
                plt.subplot(2, 2, 4)
                plt.plot(highlight_data['created_at'], highlight_data['Bed_Status_post'], 'ro-', markersize=4)
                
                # 標記差異點
                plt.plot(diff_points['created_at'], diff_points['Bed_Status_post'], 'rx', markersize=10, label='差異點')
                
                plt.xlabel('時間')
                plt.ylabel('Bed_Status')
                plt.title('Post數據 差異區域放大視圖')
                plt.legend()
                plt.grid(True)
                plt.ylim(-0.1, 1.1)  # 設置y軸範圍，使0和1值更清晰
        
        plt.tight_layout()
        plt.savefig('bed_status_comparison.png', dpi=300)
        
        # 另外創建一個圖表，專門顯示差異點
        if len(differences) > 0:
            plt.figure(figsize=(15, 10))
            
            # 按時間排序差異
            sorted_diffs = differences.sort_values('created_at')
            
            # 最多顯示10個差異點
            max_diffs = min(10, len(sorted_diffs))
            
            for i in range(max_diffs):
                diff_row = sorted_diffs.iloc[i]
                diff_time = diff_row['created_at']
                
                # 獲取差異點前後的數據
                window_start = diff_time - pd.Timedelta(minutes=2)
                window_end = diff_time + pd.Timedelta(minutes=2)
                window_data = common_rows[(common_rows['created_at'] >= window_start) & 
                                         (common_rows['created_at'] <= window_end)]
                
                # 繪製子圖
                plt.subplot(5, 2, i+1)
                
                # 繪製Pre和Post數據
                plt.plot(window_data['created_at'], window_data['Bed_Status_pre'], 'b-', label='Pre')
                plt.plot(window_data['created_at'], window_data['Bed_Status_post'], 'r-', label='Post')
                
                # 標記差異點
                plt.axvline(x=diff_time, color='green', linestyle='--', alpha=0.7)
                plt.plot(diff_time, diff_row['Bed_Status_pre'], 'bx', markersize=10)
                plt.plot(diff_time, diff_row['Bed_Status_post'], 'rx', markersize=10)
                
                plt.title(f'差異點 {i+1}: {diff_time.strftime("%Y-%m-%d %H:%M:%S")}')
                plt.ylim(-0.1, 1.1)
                
                # 只在第一個子圖顯示圖例
                if i == 0:
                    plt.legend()
                
                plt.grid(True)
            
            plt.tight_layout()
            plt.savefig('bed_status_diff_points.png', dpi=300)
            results["差異點圖表"] = "已保存為bed_status_diff_points.png"
        
        results["圖表"] = "已保存為bed_status_comparison.png"
        
        # 保存差異數據到CSV文件
        differences.to_csv('bed_status_differences.csv', index=False)
        results["差異數據CSV"] = "已保存為bed_status_differences.csv"
    
    return results

def main():
    # 定義文件路徑
    base_dir = Path("_data/00_validate")
    pre_processed_path = base_dir / "pre" / "processed_data.csv"
    post_processed_path = base_dir / "post" / "processed_data.csv"
    pre_sps_path = base_dir / "pre" / "SPS2025PA000146_2025-03-08 12:00:00_2025-03-09 12:00:00.csv"
    post_sps_path = base_dir / "post" / "SPS2025PA000146_2025-03-08 12:00:00_2025-03-09 12:00:00.csv"
    
    # 載入數據
    print("正在載入processed_data.csv文件...")
    pre_processed = load_processed_data(pre_processed_path)
    post_processed = load_processed_data(post_processed_path)
    
    print("正在載入SPS2025PA000146文件...")
    pre_sps = load_sps_data(pre_sps_path)
    post_sps = load_sps_data(post_sps_path)
    
    # 比較processed_data.csv
    print("\n比較processed_data.csv文件...")
    processed_results = compare_processed_data(pre_processed, post_processed)
    
    # 輸出結果
    print("\nprocessed_data.csv比較結果:")
    for col, result in processed_results.items():
        print(f"\n列 '{col}':")
        if isinstance(result, list):
            for diff in result:
                print(f"  - {diff}")
        else:
            print(f"  {result}")
    
    # 比較Bed_Status
    print("\n比較SPS2025PA000146文件中的Bed_Status欄位...")
    bed_status_results = compare_bed_status(pre_sps, post_sps)
    
    # 輸出結果
    print("\nBed_Status比較結果:")
    if isinstance(bed_status_results, dict):
        for key, value in bed_status_results.items():
            if key == "差異樣本" and isinstance(value, list):
                print(f"\n{key}:")
                for i, sample in enumerate(value):
                    print(f"  樣本 {i+1}:")
                    for k, v in sample.items():
                        print(f"    {k}: {v}")
            elif key == "狀態轉換模式" and isinstance(value, list):
                print(f"\n{key}:")
                for i, pattern in enumerate(value):
                    print(f"  模式 {i+1}: {pattern['Bed_Status_pre']} -> {pattern['Bed_Status_post']}, 出現次數: {pattern['count']}")
            elif isinstance(value, dict):
                print(f"\n{key}:")
                for k, v in value.items():
                    print(f"  {k}: {v}")
            else:
                print(f"{key}: {value}")
    else:
        print(bed_status_results)
    
    print("\n比較完成！")
    print("詳細結果已保存到圖表和CSV文件中。")

if __name__ == "__main__":
    main() 