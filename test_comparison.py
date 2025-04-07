#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.signal import lfilter, savgol_filter
import matplotlib.pyplot as plt

# 導入兩個程式的關鍵函數
sys.path.append('.')
from localrawviewer1217_elastic import BedParameters, process_sensor_data, calculate_movement_indicators, detect_bed_events
import importlib.util

# 定義測試數據路徑
TEST_DATA_PATH = '_data/local_viewer/test_data.csv'

def load_test_data(filepath):
    """載入測試數據"""
    print(f"載入測試數據: {filepath}")
    df = pd.read_csv(filepath)
    
    # 轉換為sensor_data格式
    sensor_data = []
    for _, row in df.iterrows():
        data_row = {
            'timestamp': row['timestamp'],
            'ch0': row['ch0'],
            'ch1': row['ch1'],
            'ch2': row['ch2'],
            'ch3': row['ch3'],
            'ch4': row['ch4'],
            'ch5': row['ch5'],
            'created_at': row['created_at']
        }
        sensor_data.append(data_row)
    
    print(f"載入了 {len(sensor_data)} 筆數據")
    return sensor_data

def run_localrawviewer_test(sensor_data):
    """使用localrawviewer1217_elastic.py處理數據"""
    print("\n=== 使用 localrawviewer1217_elastic.py 處理數據 ===")
    
    # 初始化參數 - 使用mqtt_parameters_SPS2024PA000355.csv中的值
    params = BedParameters()
    params.noise_1 = 80  # 從CSV的Noise 1
    params.noise_2 = 80  # 從CSV的Noise 2
    params.bed_threshold = 30000  # 從CSV的Total Sum
    params.movement_threshold = 150  # 從CSV的Set Flip
    params.is_air_mattress = 0  # 從CSV的Air mattress
    
    # 設定通道參數
    params.ch_preload = [40000, 40000, 40000, 40000, 40000, 40000]  # 從CSV的Channel X min_preload
    params.ch_threshold_1 = [60000, 60000, 60000, 60000, 60000, 60000]  # 從CSV的Channel X threshold_1
    params.ch_threshold_2 = [90000, 90000, 90000, 90000, 90000, 90000]  # 從CSV的Channel X threshold_2
    
    print(f"參數設定: noise_1={params.noise_1}, noise_2={params.noise_2}, bed_threshold={params.bed_threshold}")
    print(f"通道閾值1: {params.ch_threshold_1}")
    print(f"通道閾值2: {params.ch_threshold_2}")
    print(f"通道預載: {params.ch_preload}")
    
    # 處理數據
    processed_data = process_sensor_data(sensor_data, params)
    if processed_data:
        print(f"數據處理完成，長度: {len(processed_data['d10'][0])}")
        
        # 計算位移指標
        movement_data = calculate_movement_indicators(processed_data, params)
        if movement_data:
            print(f"位移指標計算完成")
            print(f"在床狀態: 在床={np.sum(movement_data['onbed'])}, 離床={len(movement_data['onbed']) - np.sum(movement_data['onbed'])}")
        
        # 檢測床上事件
        events = detect_bed_events(processed_data, params)
        if events:
            print(f"事件檢測完成")
            print(f"翻身點數: {len(events['flip_points'])}")
            print(f"在床狀態: 在床={np.sum(events['bed_status'])}, 離床={len(events['bed_status']) - np.sum(events['bed_status'])}")
            
            # 保存結果
            results = {
                'timestamp': [row['timestamp'] for row in sensor_data[:len(events['bed_status'])]],
                'bed_status': events['bed_status'],
                'rising_dist': events['rising_dist'],
                'rising_dist_air': events['rising_dist_air']
            }
            results_df = pd.DataFrame(results)
            results_df.to_csv('localrawviewer_results.csv', index=False)
            print("結果已保存至 localrawviewer_results.csv")
            
            return events
    
    return None

def prepare_onoff_bed_data(sensor_data):
    """準備onoff_bed_0803-H.py所需的數據格式"""
    # 轉換時間戳
    timestamps = []
    for row in sensor_data:
        dt = datetime.strptime(str(row['timestamp']), "%Y%m%d%H%M%S")
        timestamps.append(dt + timedelta(hours=8))
    
    # 準備通道數據
    channels = {}
    for ch in range(6):
        ch_data = [row[f'ch{ch}'] for row in sensor_data]
        channels[f'ch{ch}'] = np.array(ch_data)
    
    return timestamps, channels

def calculate_rising_dist(dist):
    """計算翻身指標的輔助函數"""
    dist_series = pd.Series(dist)
    shift = 60
    rising_dist = dist_series.shift(-shift) - dist_series
    rising_dist = np.int32(rising_dist.fillna(0))
    rising_dist[rising_dist < 0] = 0
    rising_dist = rising_dist // 127
    rising_dist[rising_dist > 1000] = 1000
    return rising_dist

def run_onoff_bed_test(sensor_data):
    """模擬onoff_bed_0803-H.py的處理邏輯"""
    print("\n=== 模擬 onoff_bed_0803-H.py 處理數據 ===")
    
    # 初始化日誌檔案
    log_file = open('onoff_bed_simulation_log.txt', 'w', encoding='utf-8')
    log_file.write("=== onoff_bed_0803-H.py 模擬處理日誌 ===\n")
    
    # 準備數據
    timestamps, channels = prepare_onoff_bed_data(sensor_data)
    print(f"數據準備完成，長度: {len(timestamps)}")
    log_file.write(f"處理數據長度: {len(timestamps)}\n")
    
    # 初始化參數 - 使用mqtt_parameters_SPS2024PA000355.csv中的值
    noise_onbed = 80  # 從CSV的Noise 1
    noise_offbed = 80  # 從CSV的Noise 2
    bed_threshold = 30000  # 從CSV的Total Sum
    movement_threshold = 150  # 從CSV的Set Flip
    is_air_mattress = 0  # 從CSV的Air mattress
    
    # 通道參數
    ch_preload = [40000, 40000, 40000, 40000, 40000, 40000]  # 從CSV的Channel X min_preload
    ch_threshold_1 = [60000, 60000, 60000, 60000, 60000, 60000]  # 從CSV的Channel X threshold_1
    ch_threshold_2 = [90000, 90000, 90000, 90000, 90000, 90000]  # 從CSV的Channel X threshold_2
    
    log_file.write(f"參數設定: noise_onbed={noise_onbed}, noise_offbed={noise_offbed}, bed_threshold={bed_threshold}\n")
    log_file.write(f"通道閾值1: {ch_threshold_1}\n")
    log_file.write(f"通道閾值2: {ch_threshold_2}\n")
    log_file.write(f"通道預載: {ch_preload}\n")
    
    print(f"參數設定: noise_onbed={noise_onbed}, noise_offbed={noise_offbed}, bed_threshold={bed_threshold}")
    print(f"通道閾值1: {ch_threshold_1}")
    print(f"通道閾值2: {ch_threshold_2}")
    print(f"通道預載: {ch_preload}")
    
    # 初始化結果數組
    n10 = []
    d10 = []
    x10 = []
    dist = 0
    dist_air = 0
    
    # 處理每個通道
    for ch in range(6):
        log_file.write(f"\n處理通道 {ch}:\n")
        data = channels[f'ch{ch}']
        
        # 計算噪聲值
        hp = np.convolve(data, [-1, -2, -3, -4, 4, 3, 2, 1], mode='same')
        lpf = [26, 28, 32, 39, 48, 60, 74, 90, 108, 126, 146, 167, 187, 208, 227, 246, 264, 280, 294, 306, 315, 322, 326, 328, 326, 322, 315, 306, 294, 280, 264, 246, 227, 208, 187, 167, 146, 126, 108, 90, 74, 60, 48, 39, 32, 28, 26]
        n = np.convolve(np.abs(hp / 16), lpf, mode='full')
        n = n[:len(data)] / 4096
        n10.append(np.int32(n))
        
        log_file.write(f"  通道 {ch} 噪聲值前10個值: {n10[ch][:10]}\n")
        
        # 特別記錄索引235的噪聲值
        if len(data) > 235:
            log_file.write(f"  通道 {ch} 索引235的噪聲值: {n10[ch][235]}\n")
        
        # 計算移動平均
        data_pd = pd.Series(data)
        med10 = data_pd.rolling(window=10, min_periods=1, center=True).mean()
        d10.append(np.int32(med10))
        
        log_file.write(f"  通道 {ch} 原始數據前10個值: {d10[ch][:10]}\n")
        
        # 特別記錄索引235的原始數據
        if len(data) > 235:
            log_file.write(f"  通道 {ch} 索引235的原始數據: {d10[ch][235]}\n")
        
        # 計算移動最大值
        max10 = data_pd.rolling(window=10, min_periods=1, center=True).max()
        x10.append(np.int32(max10))
        
        log_file.write(f"  通道 {ch} 最大值前10個值: {x10[ch][:10]}\n")
        
        # 特別記錄索引235的最大值
        if len(data) > 235:
            log_file.write(f"  通道 {ch} 索引235的最大值: {x10[ch][235]}\n")
        
        # 計算位移
        a = [1, -1023/1024]
        b = [1/1024, 0]
        pos_iirmean = lfilter(b, a, med10)
        mean_30sec = med10.rolling(window=30, min_periods=1, center=False).mean()
        mean_30sec = np.int32(mean_30sec)
        diff = (mean_30sec - pos_iirmean) / 256
        if ch == 1:
            diff = diff / 3
        dist = dist + np.square(diff)
        dist[dist > 8000000] = 8000000
        
        # 計算氣墊床位移
        mean_60sec = med10.rolling(window=60, min_periods=1, center=False).mean()
        mean_60sec = np.int32(mean_60sec)
        a = np.zeros(780)
        a[0] = 1
        b = np.zeros(780)
        for s in range(10):
            b[s*60 + 180] = -0.1
        b[60] = 1
        diff = lfilter(b, a, mean_60sec)
        if ch == 1:
            diff = diff / 3
        dist_air = dist_air + np.square(diff / 256)
    
    # 計算位移差值
    rising_dist = calculate_rising_dist(dist)
    rising_dist_air = calculate_rising_dist(dist_air)
    
    # 初始化在床狀態
    l = len(d10[0])
    onbed = np.zeros((l,))
    onload = []
    total = 0
    
    # 計算在床狀態
    for ch in range(6):
        log_file.write(f"\n計算通道 {ch} 在床狀態:\n")
        # 設定通道參數
        offset = 0
        preload = ch_preload[ch]
        th1 = ch_threshold_1[ch]
        th2 = ch_threshold_2[ch]
        
        log_file.write(f"  通道 {ch} 偏移值: {offset}\n")
        log_file.write(f"  通道 {ch} 預載值: {preload}\n")
        log_file.write(f"  通道 {ch} 閾值1: {th1}\n")
        log_file.write(f"  通道 {ch} 閾值2: {th2}\n")
        
        max10_ch = x10[ch] + offset
        med10_ch = d10[ch] + offset
        n_ch = n10[ch]
        
        # 零點判定
        zeroing = np.less(n_ch * np.right_shift(max10_ch, 5), 
                        noise_offbed * np.right_shift(preload, 5))
        
        log_file.write(f"  通道 {ch} 零點判定前10個值: {zeroing[:10]}\n")
        
        # 特別記錄索引235的零點判定
        if l > 235:
            log_file.write(f"  通道 {ch} 索引235的零點判定: {zeroing[235]}\n")
            log_file.write(f"  通道 {ch} 索引235的零點判定計算: {n_ch[235]} * {np.right_shift(max10_ch[235], 5)} < {noise_offbed} * {np.right_shift(preload, 5)}\n")
        
        # 計算基線參數
        approach = max10_ch - (th1 + th2)
        speed = n_ch // (noise_onbed * 4)
        np.clip(speed, 1, 16, out=speed)
        app_sp = approach * speed
        sp_1024 = 1024 - speed
        
        log_file.write(f"  通道 {ch} 接近度前10個值: {approach[:10]}\n")
        log_file.write(f"  通道 {ch} 速度前10個值: {speed[:10]}\n")
        
        # 特別記錄索引235的基線參數
        if l > 235:
            log_file.write(f"  通道 {ch} 索引235的接近度: {approach[235]}\n")
            log_file.write(f"  通道 {ch} 索引235的速度: {speed[235]}\n")
            log_file.write(f"  通道 {ch} 索引235的app_sp: {app_sp[235]}\n")
            log_file.write(f"  通道 {ch} 索引235的sp_1024: {sp_1024[235]}\n")
        
        # 動態基線計算
        base = (app_sp[0] // 1024 + med10_ch[0]) // 2
        base = np.int64(base)
        baseline = np.zeros_like(med10_ch)
        
        for i in range(l):
            if zeroing[i]:
                base = np.int64(med10_ch[i])
            base = (base * sp_1024[i] + app_sp[i]) // 1024
            baseline[i] = base
        
        log_file.write(f"  通道 {ch} 基線前10個值: {baseline[:10]}\n")
        
        # 特別記錄索引235的基線
        if l > 235:
            log_file.write(f"  通道 {ch} 索引235的基線: {baseline[235]}\n")
        
        # 計算負載和在床狀態
        channel_total = med10_ch - baseline
        
        log_file.write(f"  通道 {ch} 負載前10個值: {channel_total[:10]}\n")
        
        # 特別記錄索引235的負載
        if l > 235:
            log_file.write(f"  通道 {ch} 索引235的負載: {channel_total[235]}\n")
            log_file.write(f"  通道 {ch} 索引235的負載計算: {med10_ch[235]} - {baseline[235]} = {channel_total[235]}\n")
        
        total = total + channel_total
        o = np.less(th1, channel_total)
        
        log_file.write(f"  通道 {ch} 負載狀態前10個值: {o[:10]}\n")
        
        # 特別記錄索引235的負載狀態
        if l > 235:
            log_file.write(f"  通道 {ch} 索引235的負載狀態: {o[235]}\n")
            log_file.write(f"  通道 {ch} 索引235的負載狀態計算: {th1} < {channel_total[235]} = {o[235]}\n")
        
        onload.append(o)
        onbed = onbed + o
    
    # 最終在床判定
    log_file.write(f"\n總負載前10個值: {total[:10]}\n")
    bed_threshold_check = np.less(bed_threshold, total)
    log_file.write(f"床閾值檢查前10個值: {bed_threshold_check[:10]}\n")
    
    # 特別記錄索引235的總負載和床閾值檢查
    if l > 235:
        log_file.write(f"\n索引235的總負載: {total[235]}\n")
        log_file.write(f"索引235的床閾值檢查: {bed_threshold} < {total[235]} = {bed_threshold_check[235]}\n")
        log_file.write(f"索引235的在床狀態（閾值檢查前）: {onbed[235]}\n")
    
    onbed = onbed + bed_threshold_check
    onbed = np.int32(onbed > 0)
    
    log_file.write(f"最終在床狀態前10個值: {onbed[:10]}\n")
    log_file.write(f"在床狀態統計: 在床={np.sum(onbed)}, 離床={len(onbed) - np.sum(onbed)}\n")
    
    # 特別記錄索引235的最終在床狀態
    if l > 235:
        log_file.write(f"索引235的最終在床狀態: {onbed[235]}\n")
    
    # 檢測翻身事件
    log_file.write(f"\n檢測翻身事件:\n")
    if is_air_mattress == 0:
        flip_mask = (rising_dist > movement_threshold)
        log_file.write(f"使用一般床墊位移差值檢測翻身 (閾值={movement_threshold})\n")
    else:
        flip_mask = (rising_dist_air > movement_threshold)
        log_file.write(f"使用氣墊床位移差值檢測翻身 (閾值={movement_threshold})\n")
    
    log_file.write(f"翻身檢測結果前10個值: {flip_mask[:10]}\n")
    log_file.write(f"翻身檢測統計: 翻身點數={np.sum(flip_mask)}, 總點數={len(flip_mask)}\n")
    
    # 特別記錄索引235的翻身檢測
    if l > 235:
        log_file.write(f"索引235的翻身檢測: {flip_mask[235]}\n")
    
    # 找出連續翻身區間
    flip_regions = []
    start_idx = None
    
    for i in range(len(flip_mask)):
        if flip_mask[i] and start_idx is None:
            start_idx = i
        elif not flip_mask[i] and start_idx is not None:
            flip_regions.append((start_idx, i-1))
            start_idx = None
    
    # 如果最後一個區間還沒結束
    if start_idx is not None:
        flip_regions.append((start_idx, len(flip_mask)-1))
    
    log_file.write(f"連續翻身區間: {flip_regions}\n")
    
    # 計算每個翻身區間的中間點
    flip_points = [(start + (end - start)//2) for start, end in flip_regions]
    log_file.write(f"翻身中間點: {flip_points}\n")
    
    # 關閉日誌檔案
    log_file.close()
    
    print(f"位移指標計算完成")
    print(f"在床狀態: 在床={np.sum(onbed)}, 離床={len(onbed) - np.sum(onbed)}")
    print(f"翻身點數: {len(flip_points)}")
    
    # 保存結果
    results = {
        'timestamp': [row['timestamp'] for row in sensor_data[:len(onbed)]],
        'bed_status': onbed,
        'rising_dist': rising_dist,
        'rising_dist_air': rising_dist_air
    }
    results_df = pd.DataFrame(results)
    results_df.to_csv('onoff_bed_results.csv', index=False)
    print("結果已保存至 onoff_bed_results.csv")
    
    # 返回結果
    return {
        'bed_status': onbed,
        'rising_dist': rising_dist,
        'rising_dist_air': rising_dist_air,
        'flip_points': flip_points
    }

def compare_results(localrawviewer_results, onoff_bed_results):
    """比較兩個程式的結果"""
    print("\n=== 比較兩個程式的結果 ===")
    
    # 比較在床狀態
    local_bed_status = localrawviewer_results['bed_status']
    onoff_bed_status = onoff_bed_results['bed_status']
    
    min_length = min(len(local_bed_status), len(onoff_bed_status))
    local_bed_status = local_bed_status[:min_length]
    onoff_bed_status = onoff_bed_status[:min_length]
    
    bed_status_diff = np.sum(local_bed_status != onoff_bed_status)
    bed_status_diff_percent = bed_status_diff / min_length * 100
    
    print(f"在床狀態差異: {bed_status_diff} 點 ({bed_status_diff_percent:.2f}%)")
    
    # 找出差異點
    diff_indices = np.where(local_bed_status != onoff_bed_status)[0]
    if len(diff_indices) > 0:
        print("\n在床狀態差異點:")
        for idx in diff_indices:
            timestamp = sensor_data[idx]['timestamp'] if idx < len(sensor_data) else "未知"
            print(f"索引: {idx}, 時間戳: {timestamp}, localrawviewer: {local_bed_status[idx]}, onoff_bed: {onoff_bed_status[idx]}")
            
            # 輸出該點的原始數據
            if idx < len(sensor_data):
                print(f"原始數據: ch0={sensor_data[idx]['ch0']}, ch1={sensor_data[idx]['ch1']}, ch2={sensor_data[idx]['ch2']}, ch3={sensor_data[idx]['ch3']}, ch4={sensor_data[idx]['ch4']}, ch5={sensor_data[idx]['ch5']}")
    
    # 比較位移差值
    local_rising_dist = localrawviewer_results['rising_dist']
    onoff_rising_dist = onoff_bed_results['rising_dist']
    
    min_length = min(len(local_rising_dist), len(onoff_rising_dist))
    local_rising_dist = local_rising_dist[:min_length]
    onoff_rising_dist = onoff_rising_dist[:min_length]
    
    rising_dist_diff = np.abs(local_rising_dist - onoff_rising_dist)
    rising_dist_diff_mean = np.mean(rising_dist_diff)
    rising_dist_diff_max = np.max(rising_dist_diff)
    
    print(f"\n位移差值平均差異: {rising_dist_diff_mean:.2f}")
    print(f"位移差值最大差異: {rising_dist_diff_max}")
    
    # 找出位移差值最大的點
    if rising_dist_diff_max > 0:
        max_diff_indices = np.where(rising_dist_diff == rising_dist_diff_max)[0]
        print("\n位移差值最大點:")
        for idx in max_diff_indices[:5]:  # 只顯示前5個
            timestamp = sensor_data[idx]['timestamp'] if idx < len(sensor_data) else "未知"
            print(f"索引: {idx}, 時間戳: {timestamp}, localrawviewer: {local_rising_dist[idx]}, onoff_bed: {onoff_rising_dist[idx]}, 差異: {rising_dist_diff[idx]}")
    
    # 比較翻身點
    local_flip_points = set(localrawviewer_results['flip_points'])
    onoff_flip_points = set(onoff_bed_results['flip_points'])
    
    common_points = local_flip_points.intersection(onoff_flip_points)
    local_only = local_flip_points - onoff_flip_points
    onoff_only = onoff_flip_points - local_flip_points
    
    print(f"\n共同翻身點: {len(common_points)}")
    print(f"僅localrawviewer檢測到的翻身點: {len(local_only)}")
    print(f"僅onoff_bed檢測到的翻身點: {len(onoff_only)}")
    
    if len(local_only) > 0:
        print("\n僅localrawviewer檢測到的翻身點:")
        for idx in sorted(list(local_only))[:5]:  # 只顯示前5個
            timestamp = sensor_data[idx]['timestamp'] if idx < len(sensor_data) else "未知"
            print(f"索引: {idx}, 時間戳: {timestamp}")
    
    if len(onoff_only) > 0:
        print("\n僅onoff_bed檢測到的翻身點:")
        for idx in sorted(list(onoff_only))[:5]:  # 只顯示前5個
            timestamp = sensor_data[idx]['timestamp'] if idx < len(sensor_data) else "未知"
            print(f"索引: {idx}, 時間戳: {timestamp}")
    
    # 繪製比較圖
    plt.figure(figsize=(15, 10))
    
    # 在床狀態比較
    plt.subplot(3, 1, 1)
    plt.plot(local_bed_status, label='localrawviewer')
    plt.plot(onoff_bed_status, label='onoff_bed', linestyle='--')
    plt.title('在床狀態比較')
    plt.legend()
    plt.grid(True)
    
    # 位移差值比較
    plt.subplot(3, 1, 2)
    plt.plot(local_rising_dist, label='localrawviewer')
    plt.plot(onoff_rising_dist, label='onoff_bed', linestyle='--')
    plt.title('位移差值比較')
    plt.legend()
    plt.grid(True)
    
    # 差異
    plt.subplot(3, 1, 3)
    plt.plot(rising_dist_diff, label='位移差值差異')
    plt.plot(local_bed_status != onoff_bed_status, label='在床狀態差異', linestyle='--')
    plt.title('差異比較')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('comparison_results.png')
    print("比較圖已保存至 comparison_results.png")
    
    # 保存差異數據
    diff_data = {
        'timestamp': [sensor_data[i]['timestamp'] if i < len(sensor_data) else "未知" for i in range(min_length)],
        'localrawviewer_bed_status': local_bed_status,
        'onoff_bed_bed_status': onoff_bed_status,
        'bed_status_diff': local_bed_status != onoff_bed_status,
        'localrawviewer_rising_dist': local_rising_dist,
        'onoff_bed_rising_dist': onoff_rising_dist,
        'rising_dist_diff': rising_dist_diff
    }
    diff_df = pd.DataFrame(diff_data)
    diff_df.to_csv('comparison_diff.csv', index=False)
    print("差異數據已保存至 comparison_diff.csv")

def main():
    """主函數"""
    # 載入測試數據
    global sensor_data
    sensor_data = load_test_data(TEST_DATA_PATH)
    
    # 使用localrawviewer1217_elastic.py處理數據
    localrawviewer_results = run_localrawviewer_test(sensor_data)
    
    # 使用onoff_bed_0803-H.py處理數據
    onoff_bed_results = run_onoff_bed_test(sensor_data)
    
    # 比較結果
    if localrawviewer_results and onoff_bed_results:
        compare_results(localrawviewer_results, onoff_bed_results)
    else:
        print("處理失敗，無法比較結果")

if __name__ == "__main__":
    main() 