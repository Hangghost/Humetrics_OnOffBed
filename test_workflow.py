#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
測試工作流程腳本
用於驗證第一步和第二步是否能正常運行
"""

import os
import sys
import subprocess
import glob

def check_requirements():
    """檢查必要的檔案和目錄是否存在"""
    print("=== 檢查必要檔案和目錄 ===")
    
    # 檢查 serial_ids_20250519.csv
    csv_file = "./serial_ids_20250519.csv"
    if os.path.exists(csv_file):
        print(f"✓ 找到 CSV 檔案: {csv_file}")
        with open(csv_file, 'r') as f:
            lines = f.readlines()
            print(f"  - 檔案包含 {len(lines)} 行")
            for i, line in enumerate(lines[:5]):  # 顯示前5行
                print(f"  - 第{i+1}行: {line.strip()}")
    else:
        print(f"✗ 找不到 CSV 檔案: {csv_file}")
        return False
    
    # 檢查模型檔案
    model_file = "./_logs/bed_monitor_test_sum/final_model_test_sum.keras"
    if os.path.exists(model_file):
        print(f"✓ 找到模型檔案: {model_file}")
        file_size = os.path.getsize(model_file) / (1024*1024)  # MB
        print(f"  - 檔案大小: {file_size:.2f} MB")
    else:
        print(f"✗ 找不到模型檔案: {model_file}")
        return False
    
    # 檢查必要目錄
    directories = [
        "./_data/pyqt_viewer",
        "./_data/training/prediction", 
        "./_logs/pyqt-viewer"
    ]
    
    for directory in directories:
        if os.path.exists(directory):
            print(f"✓ 目錄存在: {directory}")
        else:
            print(f"✗ 目錄不存在: {directory}")
            os.makedirs(directory, exist_ok=True)
            print(f"  - 已創建目錄: {directory}")
    
    return True

def test_step1_simulation():
    """模擬第一步：檢查下載功能的相關檔案"""
    print("\n=== 第一步測試：檢查批量下載功能 ===")
    
    # 檢查是否有現有的 CMB 檔案
    cmb_files = glob.glob("./_logs/pyqt-viewer/*.cmb")
    print(f"✓ 找到 {len(cmb_files)} 個現有的 CMB 檔案")
    
    if cmb_files:
        print("  - 現有 CMB 檔案範例:")
        for i, cmb_file in enumerate(cmb_files[:3]):  # 顯示前3個
            print(f"    {i+1}. {os.path.basename(cmb_file)}")
    
    # 檢查是否有對應的資料檔案
    data_files = glob.glob("./_data/pyqt_viewer/*_data.csv")
    print(f"✓ 找到 {len(data_files)} 個資料檔案")
    
    return True

def test_step2_prediction():
    """測試第二步：CNN-LSTM 預測功能"""
    print("\n=== 第二步測試：CNN-LSTM 預測功能 ===")
    
    # 檢查預測目錄中的檔案
    prediction_files = glob.glob("./_data/training/prediction/cleaned_*.csv")
    print(f"✓ 預測目錄中有 {len(prediction_files)} 個 cleaned_ 檔案")
    
    if prediction_files:
        print("  - 預測檔案範例:")
        for i, pred_file in enumerate(prediction_files[:3]):  # 顯示前3個
            print(f"    {i+1}. {os.path.basename(pred_file)}")
        
        # 測試單一檔案預測
        test_file = prediction_files[0]
        print(f"\n測試單一檔案預測: {os.path.basename(test_file)}")
        
        cmd = [
            "python", "model/CNN-LSTM.py",
            "--predict-new",
            "--prediction-file", test_file
        ]
        
        print(f"執行命令: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                print("✓ 單一檔案預測測試成功")
                print("  - 輸出摘要:")
                output_lines = result.stdout.split('\n')
                for line in output_lines[-10:]:  # 顯示最後10行
                    if line.strip():
                        print(f"    {line}")
            else:
                print("✗ 單一檔案預測測試失敗")
                print(f"  - 錯誤訊息: {result.stderr}")
                return False
        except subprocess.TimeoutExpired:
            print("✗ 預測測試超時（5分鐘）")
            return False
        except Exception as e:
            print(f"✗ 執行預測時發生錯誤: {str(e)}")
            return False
    else:
        print("✗ 預測目錄中沒有檔案，無法測試")
        return False
    
    return True

def test_batch_prediction():
    """測試批量預測功能"""
    print("\n=== 測試批量預測功能 ===")
    
    cmd = [
        "python", "model/CNN-LSTM.py",
        "--predict-new",
        "--prediction-dir", "./_data/training/prediction"
    ]
    
    print(f"執行命令: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode == 0:
            print("✓ 批量預測測試成功")
            
            # 檢查生成的結果檔案
            result_files = glob.glob("./_logs/bed_monitor_test_sum/predictions_*.csv")
            print(f"✓ 生成了 {len(result_files)} 個預測結果檔案")
            
            return True
        else:
            print("✗ 批量預測測試失敗")
            print(f"  - 錯誤訊息: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("✗ 批量預測測試超時（10分鐘）")
        return False
    except Exception as e:
        print(f"✗ 執行批量預測時發生錯誤: {str(e)}")
        return False

def main():
    """主函數"""
    print("開始測試工作流程...")
    
    # 檢查必要條件
    if not check_requirements():
        print("\n❌ 必要條件檢查失敗，請先解決上述問題")
        return False
    
    # 測試第一步
    if not test_step1_simulation():
        print("\n❌ 第一步測試失敗")
        return False
    
    # 測試第二步
    if not test_step2_prediction():
        print("\n❌ 第二步測試失敗")
        return False
    
    # 測試批量預測
    if not test_batch_prediction():
        print("\n❌ 批量預測測試失敗")
        return False
    
    print("\n✅ 所有測試通過！工作流程可以正常運行")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 