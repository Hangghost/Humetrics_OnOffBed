#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
合併預測結果到原始檔案的腳本
"""

import pandas as pd
import numpy as np
import os
import glob

def merge_predictions_to_original_files():
    """
    將 _logs 目錄中的預測結果合併到原始檔案中
    """
    
    # 設定路徑
    logs_dir = "./_logs/bed_monitor_test_sum"
    original_data_dir = "./_data/pyqt_viewer"
    
    # 尋找所有預測結果檔案
    prediction_files = glob.glob(os.path.join(logs_dir, "predictions_*.csv"))
    
    if not prediction_files:
        print("❌ 在 logs 目錄中未找到預測結果檔案")
        print(f"請確認 {logs_dir} 目錄中是否有 predictions_*.csv 檔案")
        return
    
    print(f"📁 找到 {len(prediction_files)} 個預測結果檔案:")
    for file in prediction_files:
        print(f"  - {os.path.basename(file)}")
    
    successful_merges = 0
    
    for pred_file in prediction_files:
        try:
            # 從預測檔案名稱中提取原始檔案名稱
            # predictions_檔案名稱.csv -> 檔案名稱.csv
            base_name = os.path.basename(pred_file)
            original_filename = base_name.replace("predictions_", "")
            original_file_path = os.path.join(original_data_dir, original_filename)
            
            print(f"\n🔄 處理: {original_filename}")
            
            # 檢查原始檔案是否存在
            if not os.path.exists(original_file_path):
                print(f"❌ 原始檔案不存在: {original_file_path}")
                continue
            
            # 讀取預測結果
            pred_df = pd.read_csv(pred_file)
            print(f"📊 預測結果: {pred_df.shape[0]} 行, 欄位: {list(pred_df.columns)}")
            
            # 讀取原始檔案
            original_df = pd.read_csv(original_file_path)
            print(f"📊 原始檔案: {original_df.shape[0]} 行, {original_df.shape[1]} 欄")
            
            # 檢查預測結果是否包含需要的欄位
            if 'Predicted' not in pred_df.columns or 'Predicted_Prob' not in pred_df.columns:
                print("❌ 預測結果檔案中缺少 'Predicted' 或 'Predicted_Prob' 欄位")
                continue
            
            # 確保長度一致
            min_length = min(len(original_df), len(pred_df))
            
            # 如果原始檔案已經有這些欄位，先刪除
            if 'Predicted' in original_df.columns:
                original_df = original_df.drop(columns=['Predicted'])
            if 'Predicted_Prob' in original_df.columns:
                original_df = original_df.drop(columns=['Predicted_Prob'])
            
            # 直接添加預測結果到原始檔案
            original_df.loc[:min_length-1, 'Predicted'] = pred_df['Predicted'].values[:min_length]
            original_df.loc[:min_length-1, 'Predicted_Prob'] = pred_df['Predicted_Prob'].values[:min_length]
            
            # 為沒有預測結果的行填入預設值
            original_df['Predicted'] = original_df['Predicted'].fillna(0).astype(int)
            original_df['Predicted_Prob'] = original_df['Predicted_Prob'].fillna(0.0)
            
            # 保存合併後的檔案
            original_df.to_csv(original_file_path, index=False)
            print(f"✅ 成功合併預測結果到: {original_file_path}")
            print(f"📊 最終檔案: {original_df.shape[0]} 行, {original_df.shape[1]} 欄")
            
            successful_merges += 1
            
        except Exception as e:
            print(f"❌ 處理 {os.path.basename(pred_file)} 時發生錯誤: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n🎉 完成！成功合併 {successful_merges} 個檔案")
    
    if successful_merges > 0:
        print("\n📝 注意事項:")
        print("已添加兩個欄位:")
        print("   - Predicted: 預測結果")
        print("   - Predicted_Prob: 預測機率值")

if __name__ == "__main__":
    merge_predictions_to_original_files() 