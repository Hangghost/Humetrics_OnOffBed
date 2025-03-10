# -*- coding: utf-8 -*-

import pymysql
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import threading
import os
import json
from scipy.signal import savgol_filter, lfilter
import numpy as np
import traceback
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter, AutoLocator, MaxNLocator
from elasticsearch import Elasticsearch
import logging
from dotenv import load_dotenv
import paho.mqtt.client as mqtt
import ssl
import time
import csv
import sys

# 載入環境變數
load_dotenv()

# 確保 log_file 目錄存在
LOG_DIR = "./_log_file"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

DATA_DIR = "./_data/local_viewer"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# 在檔案開頭添加資料來源設定
DATA_SOURCE = "elastic"  # 可選值: "mysql" 或 "elastic"

# 在檔案開頭添加 MQTT 設定
MQTT_CONFIG = {
    'normal': {
        'server': "mqtt.humetrics.ai",
        'port': 8883,
        'username': "device",
        'password': "!dF-9DXbpVKHDRgBryRJJBEdqCihwN",
        'cert_path': './cert/humetric_mqtt_certificate.pem'
    },
    'test': {
        'server': "rdtest.mqtt.humetrics.ai",
        'port': 1883,
        'username': "device",
        'password': "BMY4dqh2pcw!rxa4hdy"
    }
}

# 添加預設參數常數
DEFAULT_PARAMETERS = {
    "41": "30000",  # Total Sum
    "42": "40000", "43": "40000", "44": "40000",  # min_preload
    "45": "40000", "46": "40000", "47": "40000",
    "48": "60000", "49": "60000", "50": "60000",  # threshold_1
    "51": "60000", "52": "60000", "53": "60000",
    "54": "80",    # Noise 2
    "55": "80",    # Noise 1
    "56": "400",   # Set Flip
    "57": "0",     # Air mattress
    "58": "90000", "59": "90000", "60": "90000",  # threshold_2
    "61": "90000", "62": "90000", "63": "90000"
}

# MySQL 資料庫連接信息
db_sensor_config = {
    "host": "61.220.29.122",
    "port": 23060,
    "user": "readonly_user",
    "password": "jF@0McVu1KcP",
    "database": "humetrics"
}

# Elasticsearch 連接信息
es_sensor_config = {
    "hosts": os.getenv('ELASTICSEARCH_HOST', 'http://192.168.1.68:9200'),
    "api_key": os.getenv('ELASTICSEARCH_API_KEY'),
    "verify_certs": False,
    "request_timeout": 30,
    "retry_on_timeout": True,
    "max_retries": 3,
    "ssl_show_warn": False
}

# 檢查必要的設定
if not es_sensor_config["api_key"]:
    logging.error("錯誤：未提供 ELASTICSEARCH_API_KEY 環境變數")
    # 可以選擇在這裡拋出異常或處理錯誤
    
if not es_sensor_config["hosts"]:
    logging.error("錯誤：未提供 ELASTICSEARCH_HOST 環境變數")
    # 可以選擇在這裡拋出異常或處理錯誤

query_done = threading.Event()  # 查询完成状态标记

# 在文件開頭添加全局變數
global startday, n10, d10, x10, data_resp, rising_dist, rising_dist_air, base_final, parameter_table

# 在文件開頭添加必要的參數結構
class BedParameters:
    def __init__(self):
        # 基本閾值
        self.bed_threshold = 0      # 判斷在床/離床的基本閾值
        self.noise_1 = 0           # 第一噪音閾值
        self.noise_2 = 0           # 第二噪音閾值
        self.movement_threshold = 0  # 移動判定閾值
        
        # 各通道參數
        self.channel_params = {
            'preload': [0] * 6,     # 預載值
            'threshold1': [0] * 6,  # 主要閾值
            'threshold2': [0] * 6,  # 次要閾值
            'offset': [0] * 6       # 偏移值
        }
        
        # 床墊類型
        self.is_air_mattress = False  # True為空氣床墊，False為一般床墊

# 添加參數表初始化
def init_parameter_table(root):
    """初始化參數表"""
    # 創建一個框架來容納表格
    table_frame = ttk.Frame(root)
    
    # 創建通道參數表格
    channel_headers = ['Channel 1', 'Channel 2', 'Channel 3', 'Channel 4',
                      'Channel 5', 'Channel 6']
              
    channel_row_headers = ['min_preload', 'threshold_1', 'threshold_2', 'offset level']
    
    # 創建通道參數表格
    entries = []
    # 添加列標題
    for j, header in enumerate(channel_headers):
        label = ttk.Label(table_frame, text=header, anchor="center")
        label.grid(row=0, column=j+1, padx=2, pady=2)
    
    # 添加通道參數行和輸入框
    for i, row_header in enumerate(channel_row_headers):
        # 行標題
        label = ttk.Label(table_frame, text=row_header, anchor="e")
        label.grid(row=i+1, column=0, padx=2, pady=2)
        
        # 輸入框
        row_entries = []
        for j in range(6):  # 6個通道
            entry = ttk.Entry(table_frame, width=8)
            entry.insert(0, "0")  # 設定預設值為 0
            entry.grid(row=i+1, column=j+1, padx=2, pady=2)
            row_entries.append(entry)
        entries.append(row_entries)
    
    # 創建一個框架來容納表格和按鈕
    control_frame = ttk.Frame(root)
    control_frame.grid(row=4, column=7, columnspan=3, padx=5, pady=5, sticky="nw")
    
    # 添加獨立參數
    single_params = ['Total Sum', 'Noise 1', 'Noise 2', 'Set Flip']
    single_entries = []
    
    # 創建獨立參數的標籤和輸入框
    for i, param in enumerate(single_params):
        # 標籤
        label = ttk.Label(control_frame, text=param, anchor="e")
        label.grid(row=i, column=0, padx=5, pady=2)
        
        # 輸入框
        entry = ttk.Entry(control_frame, width=8)
        entry.insert(0, "0")  # 設定預設值為 0
        entry.grid(row=i, column=1, padx=5, pady=2)
        single_entries.append(entry)
    
    # 添加 MQTT 模式選擇
    mode_frame = ttk.Frame(control_frame)
    mode_frame.grid(row=len(single_params), column=0, columnspan=2, pady=5)
    
    mqtt_mode = tk.StringVar(value="normal")
    ttk.Radiobutton(mode_frame, text="Normal", variable=mqtt_mode, value="normal").pack(side=tk.LEFT, padx=5)
    ttk.Radiobutton(mode_frame, text="Test", variable=mqtt_mode, value="test").pack(side=tk.LEFT, padx=5)
    
    # 添加 MQTT Get Para 按鈕
    mqtt_button = ttk.Button(control_frame, text="MQTT Get Para", command=lambda: mqtt_get_parameters(mqtt_mode.get()))
    mqtt_button.grid(row=len(single_params)+1, column=0, columnspan=2, pady=5)
    
    # 將表格和控制框架添加到主窗口
    table_frame.grid(row=4, column=0, columnspan=7, padx=5, pady=5)
    control_frame.grid(row=4, column=7, columnspan=3, padx=5, pady=5, sticky="nw")
    
    return {
        'frame': table_frame,
        'entries': entries,
        'single_frame': control_frame,
        'single_entries': single_entries,
        'mqtt_mode': mqtt_mode  # 添加 mqtt_mode 到返回的字典中
    }

def get_parameters_from_table(parameter_table):
    """從參數表獲取參數值"""
    params = BedParameters()
    
    try:
        # 獲取通道參數
        for i in range(6):  # 6個通道
            params.channel_params['preload'][i] = int(parameter_table['entries'][0][i].get())
            params.channel_params['threshold1'][i] = int(parameter_table['entries'][1][i].get())
            params.channel_params['threshold2'][i] = int(parameter_table['entries'][2][i].get())
            params.channel_params['offset'][i] = int(parameter_table['entries'][3][i].get())
        
        # 獲取獨立參數
        params.bed_threshold = int(parameter_table['single_entries'][0].get())    # Total Sum
        params.noise_1 = int(parameter_table['single_entries'][1].get())         # Noise 1
        params.noise_2 = int(parameter_table['single_entries'][2].get())         # Noise 2
        params.movement_threshold = int(parameter_table['single_entries'][3].get()) # Set Flip
        
    except ValueError as e:
        messagebox.showerror("Error", f"參數格式錯誤: {str(e)}")
        return None
    except Exception as e:
        messagebox.showerror("Error", f"參數獲取錯誤: {str(e)}")
        traceback.print_exc()
        return None
        
    return params

def mqtt_get_parameters(mqtt_mode):
    """從 MQTT 讀取參數並更新參數表"""
    try:
        # 獲取設備序號
        serial_id = serial_id_entry.get()
        if not serial_id:
            messagebox.showerror("Error", "請輸入設備序號")
            return

        # 獲取 MQTT 配置
        config = MQTT_CONFIG[mqtt_mode]
        
        # 初始化變數
        reg_table = {}
        timeout = 20
        start_time = time.time()

        # 定義訊息接收回調
        def on_message(client, userdata, message):
            nonlocal timeout
            try:
                reg = json.loads(message.payload)
                if "taskID" in reg:
                    del reg["taskID"]
                reg_table.update(reg)
                
                # 根據收到的參數調整超時時間
                if reg and int(list(reg.keys())[0]) >= 63:
                    timeout = 30
                else:
                    timeout = 120
                    
            except Exception as e:
                logging.error(f"處理 MQTT 訊息時發生錯誤: {str(e)}")

        # 建立 MQTT 客戶端
        client = mqtt.Client()
        client.on_message = on_message
        client.username_pw_set(config['username'], config['password'])

        # 根據模式設定連接
        if mqtt_mode == 'normal':
            client.tls_set(config['cert_path'], None, None, cert_reqs=ssl.CERT_NONE)
            client.connect(config['server'], config['port'], 60)
        else:
            client.connect(config['server'], config['port'], 60)

        # 訂閱主題
        topic_get_regs = f"algoParam/{serial_id}/get"
        client.subscribe(topic_get_regs)

        # 發布獲取參數請求
        topic_reg_mode = f"systemManage/{serial_id}"
        payload = {"command": "08", "parameter": "", "taskID": 0}
        client.publish(topic_reg_mode, json.dumps(payload))

        # 等待接收參數
        while True:
            client.loop()
            current_time = time.time()
            if current_time - start_time > timeout:
                logging.warning(f"MQTT {timeout} 秒超時！")
                break
            if "999" in reg_table:
                break

        client.disconnect()

        # 如果成功獲取參數，更新參數表
        if reg_table:

            # 定義參數說明
            param_descriptions = {
                '41': 'Total Sum',
                '42': 'Channel 1 min_preload',
                '43': 'Channel 2 min_preload',
                '44': 'Channel 3 min_preload',
                '45': 'Channel 4 min_preload',
                '46': 'Channel 5 min_preload',
                '47': 'Channel 6 min_preload',
                '48': 'Channel 1 threshold_1',
                '49': 'Channel 2 threshold_1',
                '50': 'Channel 3 threshold_1',
                '51': 'Channel 4 threshold_1',
                '52': 'Channel 5 threshold_1',
                '53': 'Channel 6 threshold_1',
                '54': 'Noise 2',
                '55': 'Noise 1',
                '56': 'Set Flip',
                '57': 'Air mattress',
                '58': 'Channel 1 threshold_2',
                '59': 'Channel 2 threshold_2',
                '60': 'Channel 3 threshold_2',
                '61': 'Channel 4 threshold_2',
                '62': 'Channel 5 threshold_2',
                '63': 'Channel 6 threshold_2'
            }

            # 將reg_table 存成 CSV，只保存重要參數
            csv_file = f"./_data/local_viewer/mqtt_parameters_{serial_id}.csv"
            with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                # 寫入標題
                writer.writerow(['Parameter ID', 'Description', 'Value'])
                
                # 寫入資料
                for param_id, description in param_descriptions.items():
                    if param_id in reg_table and reg_table[param_id] != -1:
                        writer.writerow([param_id, description, reg_table[param_id]])

            messagebox.showinfo("Success", f"成功從 MQTT 獲取參數並儲存至 {csv_file}")

            update_parameter_table(reg_table)
        else:
            # 如果獲取失敗，使用預設參數

            update_parameter_table(DEFAULT_PARAMETERS)

            messagebox.showwarning("Warning", "無法從 MQTT 獲取參數，使用預設值")

    except Exception as e:
        logging.error(f"MQTT 參數獲取失敗: {str(e)}")
        messagebox.showerror("Error", f"MQTT 參數獲取失敗: {str(e)}")
        # 使用預設參數
        update_parameter_table(DEFAULT_PARAMETERS)

def update_parameter_table(reg_table):
    """更新參數表的值"""
    try:
        # 更新通道參數
        for ch in range(6):
            # min_preload
            parameter_table['entries'][0][ch].delete(0, tk.END)
            parameter_table['entries'][0][ch].insert(0, str(reg_table[str(ch + 42)]))
            
            # threshold_1
            parameter_table['entries'][1][ch].delete(0, tk.END)
            parameter_table['entries'][1][ch].insert(0, str(reg_table[str(ch + 48)]))
            
            # threshold_2
            parameter_table['entries'][2][ch].delete(0, tk.END)
            parameter_table['entries'][2][ch].insert(0, str(reg_table[str(ch + 58)]))

        # 更新獨立參數
        # Total Sum
        parameter_table['single_entries'][0].delete(0, tk.END)
        parameter_table['single_entries'][0].insert(0, str(reg_table['41']))
        
        # Noise 1
        parameter_table['single_entries'][1].delete(0, tk.END)
        parameter_table['single_entries'][1].insert(0, str(reg_table['55']))
        
        # Noise 2
        parameter_table['single_entries'][2].delete(0, tk.END)
        parameter_table['single_entries'][2].insert(0, str(reg_table['54']))
        
        # Set Flip
        parameter_table['single_entries'][3].delete(0, tk.END)
        parameter_table['single_entries'][3].insert(0, str(reg_table['56']))

    except Exception as e:
        logging.error(f"更新參數表時發生錯誤: {str(e)}")
        messagebox.showerror("Error", f"更新參數表時發生錯誤: {str(e)}")


def save_sensor_data(sensor_data, filename=None):
    """
    將 sensor_data 存成檔案
    
    Args:
        sensor_data: 感測器數據
        filename: 指定的檔名，如果為 None 則自動生成
    """
    try:
        # 如果沒有指定檔名，則自動生成
        if filename is None:
            serial_id = serial_id_entry.get()
            start_time = start_time_entry.get()
            end_time = end_time_entry.get()
            # 處理datetime格式
            start_time = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
            end_time = datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")


            filename = f"{serial_id}_{start_time}_{end_time}.json"
        
        # 確保 DATA_DIR 存在
        if not os.path.exists(DATA_DIR):
            os.makedirs(DATA_DIR)
        
        filepath_json = os.path.join(DATA_DIR, filename)
        filepath_csv = os.path.join(DATA_DIR, filename.replace('.json', '.csv'))
        
        # 準備要存檔的數據
        save_data = {
            'serial_id': serial_id_entry.get(),
            'start_time': start_time_entry.get(),
            'end_time': end_time_entry.get(),
            'data': []
        }
        
        # 整理數據格式
        for i in range(len(sensor_data)):
            record = {
                'created_at': sensor_data[i]['created_at'],
                'ch0': sensor_data[i]['ch0'],
                'ch1': sensor_data[i]['ch1'],
                'ch2': sensor_data[i]['ch2'],
                'ch3': sensor_data[i]['ch3'],
                'ch4': sensor_data[i]['ch4'],
                'ch5': sensor_data[i]['ch5'],                
                'timestamp': sensor_data[i]['timestamp']
            }
            
            # 如果有通知狀態，也加入到記錄中
            if 'notify_status' in sensor_data[i]:
                record['notify_status'] = sensor_data[i]['notify_status']
                
            save_data['data'].append(record)
        
        # 寫入 JSON 檔案
        with open(filepath_json, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, default=str)

        # 修改 CSV 存檔部分
        with open(filepath_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # 檢查是否有通知狀態欄位
            has_notify = any('notify_status' in record for record in sensor_data)
            
            # 寫入標題列
            headers = ['created_at', 'ch0', 'ch1', 'ch2', 'ch3', 'ch4', 'ch5', 'timestamp']
            if has_notify:
                headers.append('notify_status')
            writer.writerow(headers)
            
            # 寫入數據列
            for record in sensor_data:
                row = [
                    record['created_at'],
                    record['ch0'],
                    record['ch1'],
                    record['ch2'],
                    record['ch3'],
                    record['ch4'],
                    record['ch5'],                    
                    record['timestamp']
                ]
                
                # 如果有通知狀態，也加入到CSV中
                if has_notify:
                    row.append(record.get('notify_status', ''))
                    
                writer.writerow(row)

        print(f"成功儲存資料到 {filepath_json} 和 {filepath_csv}")
        return filepath_json
        
    except Exception as e:
        messagebox.showerror("Error", f"儲存數據時發生錯誤: {str(e)}")
        return None
    
def load_sensor_data(filepath):
    """
    讀取已存檔的感測器數據
    
    Args:
        filepath: 數據檔案路徑
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            saved_data = json.load(f)
            
        # 還原數據格式
        sensor_data = saved_data['data']
        
        # 更新輸入欄位
        serial_id_entry.delete(0, tk.END)
        serial_id_entry.insert(0, saved_data['serial_id'])
        
        start_time_entry.delete(0, tk.END)
        start_time_entry.insert(0, saved_data['start_time'])
        
        end_time_entry.delete(0, tk.END)
        end_time_entry.insert(0, saved_data['end_time'])
        
        return sensor_data
        
    except Exception as e:
        messagebox.showerror("Error", f"讀取數據時發生錯誤: {str(e)}")
        return None

def check_local_data(serial_id, start_time, end_time):
    """
    檢查是否有符合條件的本地檔案
    
    Args:
        serial_id: 設備序號
        start_time: 開始時間
        end_time: 結束時間
    
    Returns:
        str or None: 找到的檔案路徑，如果沒有則返回 None
    """
    try:
        # 確保 DATA_DIR 存在
        if not os.path.exists(DATA_DIR):
            return None
            
        # 搜尋所有 json 檔案
        for filename in os.listdir(DATA_DIR):
            if not filename.endswith('.json'):
                continue
                
            filepath = os.path.join(DATA_DIR, filename)
            
            # 讀取檔案內容
            with open(filepath, 'r', encoding='utf-8') as f:
                saved_data = json.load(f)
                
            # 檢查序號和時間是否符合
            if (saved_data['serial_id'] == serial_id and
                saved_data['start_time'] == start_time and
                saved_data['end_time'] == end_time):
                return filepath
                
        return None
        
    except Exception as e:
        print(f"檢查本地檔案時發生錯誤: {str(e)}")
        return None

def fetch_sensor_data(serial_id, start_time, end_time):
    """
    從資料庫獲取數據並自動存檔，如果本地已有檔案則直接讀取
    """
    try:
        # 先檢查本地檔案
        print(f"檢查本地檔案: {serial_id}, {start_time}, {end_time}")
        local_file = check_local_data(serial_id, start_time, end_time)
        if local_file:
            return load_sensor_data(local_file)
            
        # 根據資料來源選擇不同的查詢方式
        if DATA_SOURCE == "mysql":
            print(f"從 MySQL 獲取數據: {serial_id}, {start_time}, {end_time}")
            return fetch_from_mysql(serial_id, start_time, end_time)
        elif DATA_SOURCE == "elastic":
            print(f"從 Elasticsearch 獲取數據: {serial_id}, {start_time}, {end_time}")
            return fetch_from_elastic(serial_id, start_time, end_time)
        else:
            raise ValueError(f"不支援的資料來源: {DATA_SOURCE}")
            
    except Exception as e:
        messagebox.showerror("Error", f"獲取感測器數據時發生錯誤: {str(e)}")
        return None

def fetch_from_mysql(serial_id, start_time, end_time):
    """從 MySQL 獲取數據"""
    try:
        # 檢查是否已安裝 cryptography
        try:
            import cryptography
        except ImportError:
            messagebox.showerror("Error", "請先安裝 cryptography 套件:\npip install cryptography")
            return None
            
        query = """
            SELECT serial_id, ch0, ch1, ch2, ch3, ch4, ch5, timestamp, created_at 
            FROM sensor_data 
            WHERE serial_id = %s 
            AND created_at >= %s 
            AND created_at <= %s
            ORDER BY timestamp ASC;
        """
        
        connection = pymysql.connect(**db_sensor_config)
        cursor = connection.cursor(pymysql.cursors.DictCursor)
        cursor.execute(query, (serial_id, start_time, end_time))
        result = cursor.fetchall()

        if result:
            # 自動存檔
            save_sensor_data(result)
            
            # 處理時間戳
            for row in result:
                timestamp_str = str(row['timestamp'])
                timestamp_dt = datetime.strptime(timestamp_str, "%Y%m%d%H%M%S")
                time_at = timestamp_dt + timedelta(hours=8)
                row['time_at'] = time_at.strftime("%Y-%m-%d %H:%M:%S")

            return result
        else:
            return None
            
    finally:
        if 'connection' in locals():
            connection.close()

def fetch_from_elastic(serial_id, start_time, end_time):
    """從 Elasticsearch 獲取數據"""
    try:
        # 檢查時間範圍
        start_dt = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
        end_dt = datetime.strptime(end_time, '%Y-%m-%d %H:%M:%S')
        
        # 檢查是否為未來時間
        current_time = datetime.now()
        if start_dt > current_time or end_dt > current_time:
            messagebox.showwarning("Warning", "查詢時間不能是未來時間！請修改查詢時間範圍。")
            return None
            
        # 檢查時間範圍是否超過24小時
        if (end_dt - start_dt).days > 1:
            messagebox.showwarning("Warning", "查詢時間範圍不能超過24小時！")
            return None

        # 設定 Elasticsearch 連線
        es_config = {
            "hosts": es_sensor_config["hosts"],
            "request_timeout": 30,
            "retry_on_timeout": True,
            "max_retries": 3,
            "ssl_show_warn": False
        }
        
        if es_sensor_config.get("api_key"):
            es_config["api_key"] = es_sensor_config["api_key"]
            
        if es_sensor_config["hosts"].startswith("https"):
            es_config["verify_certs"] = es_sensor_config["verify_certs"]
            
        es = Elasticsearch(**es_config)

        # 先檢查索引是否存在
        indices = es.indices.get_alias(index="sensor_data-*").keys()
        if not indices:
            messagebox.showerror("Error", "找不到有效的資料索引！")
            return None

        # 檢查設備是否存在
        device_query = {
            "query": {
                "match": {
                    "serial_id": serial_id
                }
            },
            "size": 1
        }
        
        device_check = es.search(
            index="sensor_data-*",
            body=device_query
        )
        
        if device_check['hits']['total']['value'] == 0:
            messagebox.showwarning("Warning", f"找不到設備 {serial_id} 的資料！")
            return None
        
        print(f"找到 {device_check['hits']['total']} 筆資料")

        # 轉換為 ISO 格式
        start_iso = start_dt.strftime('%Y-%m-%dT%H:%M:%S+08:00')
        end_iso = end_dt.strftime('%Y-%m-%dT%H:%M:%S+08:00')

        print(f"查詢時間範圍：{start_iso} 到 {end_iso}")

        # 建立查詢條件
        query_sensor_data = {
            "query": {
                "bool": {
                    "must": [
                        {"match": {"serial_id": serial_id}},
                        {"range": {"created_at": {
                            "gte": start_iso,
                            "lte": end_iso
                        }}}
                    ]
                }
            },
            "sort": [{"created_at": "asc"}],
            "size": 500
        }

        query_notify_data = {
                "query": {
                    "bool": {
                        "must": [
                            {"match": {"serial_id": serial_id}},
                            {"range": {"created_at": {
                                "gte": start_iso,
                                "lte": end_iso
                            }}}
                        ]
                    }
                },
                "size": 500
            }
        
        # 使用 scroll API 獲取資料
        page = es.search(
            index="sensor_data",  # 修改為與 elastic_data_loader.py 一致的索引名稱
            body=query_sensor_data,
            scroll='5m'
        )
        
        scroll_id = page['_scroll_id']
        hits = page['hits']['hits']

        print(f"獲取到 {len(hits)} 筆資料")
     
        # 用於儲存所有記錄
        all_records = []
        limit = 90000  # 設定資料筆數限制
        
        # 當還有資料時，持續獲取，但不超過限制
        while len(hits) > 0 and len(all_records) < limit:
            # 處理當前批次的資料
            records = []
            for hit in hits:
                if len(all_records) >= limit:
                    break
                source = hit['_source']
                records.append({
                    'created_at': source.get('created_at'),
                    'serial_id': source.get('serial_id'),
                    'ch0': source.get('ch0', source.get('Channel_1_Raw')),  # 嘗試兩種可能的鍵名
                    'ch1': source.get('ch1', source.get('Channel_2_Raw')),
                    'ch2': source.get('ch2', source.get('Channel_3_Raw')),
                    'ch3': source.get('ch3', source.get('Channel_4_Raw')),
                    'ch4': source.get('ch4', source.get('Channel_5_Raw')),
                    'ch5': source.get('ch5', source.get('Channel_6_Raw')),
                    'Angle': source.get('angle'),
                    'timestamp': source.get('timestamp')
                })
            
            all_records.extend(records)
            
            # 顯示進度
            print(f"已獲取 {len(all_records)}/{limit} 筆資料")
            
            # 如果已達到限制，跳出迴圈
            if len(all_records) >= limit:
                break
                
            # 獲取下一批資料
            page = es.scroll(
                scroll_id=scroll_id,
                scroll='5m'
            )
            hits = page['hits']['hits']

        # 清理 scroll
        es.clear_scroll(scroll_id=scroll_id)

        # 獲取通知資料
        notify_page = es.search(
            index="notify-*",
            body=query_notify_data,
            scroll='5m'
        )

        notify_hits = notify_page['hits']['hits']
        notify_dict = {}
        
        # 處理通知資料
        notify_scroll_id = notify_page.get('_scroll_id')
        notify_count = 0
        notify_limit = 90000  # 設定通知資料筆數限制
        
        print(f"開始獲取通知資料...")
        
        # 當還有通知資料時，持續獲取
        while notify_hits and notify_count < notify_limit:
            # 處理當前批次的通知資料
            for hit in notify_hits:
                if notify_count >= notify_limit:
                    break
                    
                source = hit['_source']
                timestamp = source.get('timestamp')
                if timestamp:
                    notify_dict[timestamp] = source.get('statusType')
                    notify_count += 1
            
            # 如果已達到限制，跳出迴圈
            if notify_count >= notify_limit:
                break
                
            # 獲取下一批通知資料
            if notify_scroll_id:
                notify_page = es.scroll(
                    scroll_id=notify_scroll_id,
                    scroll='5m'
                )
                notify_hits = notify_page['hits']['hits']
            else:
                break
        
        # 清理通知資料的 scroll
        if notify_scroll_id:
            es.clear_scroll(scroll_id=notify_scroll_id)
        
        print(f"成功獲取 {notify_count} 筆通知資料")
        
        # 將通知資料整合到感測器資料中
        if notify_dict and all_records:
            matched_count = 0
            for record in all_records:
                timestamp = record.get('timestamp')
                if timestamp in notify_dict:
                    record['notify_status'] = notify_dict[timestamp]
                    matched_count += 1
            
            print(f"成功將 {matched_count} 筆通知資料整合到感測器資料中")

        if all_records:
            # 自動存檔
            save_sensor_data(all_records)
            
            # # 處理時間戳??
            # for record in all_records:
            #     timestamp_str = str(record['timestamp'])
            #     timestamp_dt = datetime.strptime(timestamp_str, "%Y%m%d%H%M%S")
            #     time_at = timestamp_dt + timedelta(hours=8)
            #     record['time_at'] = time_at.strftime("%Y-%m-%d %H:%M:%S")

            print(f"成功獲取 {len(all_records)} 筆資料")
            return all_records
        else:
            print("未找到符合條件的資料")
            return None
            
    except Exception as e:
        error_msg = f"從 Elasticsearch 獲取資料時發生錯誤: {str(e)}"
        logging.error(error_msg)
        messagebox.showerror("Error", error_msg)
        return None
    
# 在各繪圖環節添加長度校驗
def safe_plot(ax, x, y, **kwargs):
    """安全繪圖函數，自動對齊數據長度"""
    min_len = min(len(x), len(y))
    return ax.plot(x[:min_len], y[:min_len], **kwargs)

# 函数：绘制合并图表
def plot_combined_data(sensor_data):
    if not sensor_data:
        messagebox.showwarning("Warning", "缺少数据，无法绘制图表。")
        return

    def show_plot():
        try:
            # 使用全局變數
            global parameter_table
            
            # 從參數表獲取參數
            params = get_parameters_from_table(parameter_table)
            if params is None:
                messagebox.showerror("Error", "無法獲取參數設置")
                return
            
            # 使用 nonlocal 聲明 sensor_data 是外部變數
            nonlocal sensor_data
            
            # 印出 sensor_data 的所有欄位
            print(f"sensor_data 的所有欄位: {sensor_data[0].keys()}")
            
            # 處理數據
            # 先按照 timestamp 欄位對 sensor_data 進行排序
            sensor_data = sorted(sensor_data, key=lambda x: x['timestamp'])
            
            processed_data = process_sensor_data(sensor_data, params)
            print(f"processed_data的長度: {len(processed_data)}")
            # print(f"processed_data: {processed_data}")
            if processed_data is None:
                messagebox.showerror("Error", "數據處理失敗")
                return

            # # 將數據存成dataframe後CSV
            processed_df = pd.DataFrame(processed_data)
            processed_df.to_csv('processed_data.csv', index=False)
            
            # 生成時間軸 - 移除降採樣
            timestamps = [
                datetime.strptime(str(row['timestamp']), "%Y%m%d%H%M%S") + timedelta(hours=8)
                for row in sensor_data
            ][:len(processed_data['d10'][0])]  # 嚴格對齊處理後數據長度

            # 統一降採樣計算方式
            sample_rate = max(len(timestamps) // 87000, 1)  # 動態計算採樣率
            print(f"sample_rate!!!!: {sample_rate}")
            sampled_timestamps = timestamps[::sample_rate]

            # 第一次偵測可以儲存為全局變數
            initial_events = detect_bed_events(processed_data, params)
            if initial_events is None:
                messagebox.showerror("Error", "事件檢測失敗")
                return
            
            def update_plot():
                try:
                    new_params = get_parameters_from_table(parameter_table)
                    if new_params:
                        # 只有當參數改變時才重新偵測
                        if params != new_params:
                            new_events = detect_bed_events(processed_data, new_params)
                            print("參數改變，重新偵測")
                        else:
                            new_events = initial_events
                            print("參數未改變，使用初始事件")
                            
                        if new_events:
                            
                            # 印出 new_events 各欄位的資料長度
                            print(f'bed_status 長度: {len(new_events["bed_status"])}')
                            print(f'movement 長度: {len(new_events["movement"])}')
                            print(f'flip_points 長度: {len(new_events["flip_points"])}')
                            print(f'rising_dist 長度: {len(new_events["rising_dist"])}')
                            print(f'rising_dist_air 長度: {len(new_events["rising_dist_air"])}')
                            print(f'onload 長度: {len(new_events["onload"])}')
                            
                            # 額外印出 onload 中每個陣列的長度
                            for i, onload_arr in enumerate(new_events["onload"]):
                                print(f'onload[{i}] 長度: {len(onload_arr)}')

                            # 讀取原始CSV檔案
                            serial_id = serial_id_entry.get()
                            start_time = start_time_entry.get()
                            end_time = end_time_entry.get()
                            csv_file = f"./_data/local_viewer/{serial_id}_{start_time}_{end_time}.csv"
                            
                            if os.path.exists(csv_file):
                                df = pd.read_csv(csv_file)
                                data_length = len(df)
                                
                                # 確保所有陣列長度與 df 相同
                                # 添加演算法判斷結果
                                bed_status_arr = new_events['bed_status']
                                if len(bed_status_arr) > data_length:
                                    bed_status_arr = bed_status_arr[:data_length]
                                elif len(bed_status_arr) < data_length:
                                    # 如果陣列較短，用最後一個值填充
                                    bed_status_arr = np.pad(bed_status_arr, 
                                        (0, data_length - len(bed_status_arr)), 
                                        'edge')
                                df['Bed_Status'] = bed_status_arr

                                # 處理 Rising_Dist_Normal
                                rising_dist = new_events['rising_dist']
                                if len(rising_dist) > data_length:
                                    rising_dist = rising_dist[:data_length]
                                elif len(rising_dist) < data_length:
                                    rising_dist = np.pad(rising_dist, 
                                        (0, data_length - len(rising_dist)), 
                                        'edge')
                                df['Rising_Dist_Normal'] = rising_dist

                                # 處理 Rising_Dist_Air
                                rising_dist_air = new_events['rising_dist_air']
                                if len(rising_dist_air) > data_length:
                                    rising_dist_air = rising_dist_air[:data_length]
                                elif len(rising_dist_air) < data_length:
                                    rising_dist_air = np.pad(rising_dist_air, 
                                        (0, data_length - len(rising_dist_air)), 
                                        'edge')
                                df['Rising_Dist_Air'] = rising_dist_air
                                
                                # 添加翻身事件
                                flip_events = np.zeros(data_length)
                                timestamps = pd.to_datetime(df['timestamp'])  # 將時間欄位轉換為datetime格式
                                
                                # 對每個翻身時間點進行處理
                                flip_times = []
                                for idx in new_events['flip_points']:
                                    # print(f"處理索引 {idx}")
                                    if idx < len(timestamps):
                                        flip_times.append(timestamps[idx])
                                        # print(f"成功添加時間點: {timestamps[idx]}")
                                    else:
                                        # print(f"索引超出範圍: {idx} >= {len(timestamps)}")
                                        pass

                                # print(f"生成的flip_times長度: {len(flip_times)}")

                                # 繪製翻身標記
                                if flip_times and show_vars['flip'].get():
                                    # print(f"準備繪製 {len(flip_times)} 個翻身標記")
                                    ax2.scatter(flip_times, 
                                               [1.1] * len(flip_times),
                                               marker='v',
                                               color='#FF0000',
                                               s=100,
                                               zorder=10,
                                               label='Flip')
                                    # print(f"成功繪製 {len(flip_times)} 個翻身標記")
                                else:
                                    # print(f"未能繪製翻身標記: flip_times為空={not bool(flip_times)}, show_flip={show_vars['flip'].get()}")
                                    pass

                                # 添加在床狀態
                                # bed_status_arr = new_events['bed_status']
                                # if len(bed_status_arr) > data_length:
                                #     bed_status_arr = bed_status_arr[:data_length]
                                # elif len(bed_status_arr) < data_length:
                                #     # 如果陣列較短，用最後一個值填充
                                #     bed_status_arr = np.pad(bed_status_arr, 
                                #         (0, data_length - len(bed_status_arr)), 
                                #         'edge')
                                # df['Bed_Status'] = bed_status_arr
                                
                                # 保存更新後的CSV
                                df.to_csv(csv_file, index=False)
                                print(f"已更新演算法判斷結果到檔案: {csv_file}")
                            
                            # 清除現有圖表
                            ax2.clear()
                            
                            # 繪製各通道在床狀態
                            for ch in range(6):
                                if show_vars['channels'][ch].get():
                                    data = new_events['onload'][ch][:len(timestamps)]  # 嚴格截斷
                                    sampled_data = data[::sample_rate]
                                    safe_plot(ax2, 
                                             sampled_timestamps, 
                                             sampled_data,
                                             label=f'Ch{ch+1}',
                                             alpha=0.5)
                            print("在床狀態繪製完成")
                            
                            # 繪製整體在床狀態
                            if show_vars['bed_status'].get():
                                print(f"timestamps長度: {len(timestamps)}")
                                data = new_events['bed_status'][:len(timestamps)]  # 嚴格截斷
                                sampled_data = data[::sample_rate]
                                print(f"data長度: {len(data)}")
                                print(f"sampled_data長度: {len(sampled_data)}")
                                safe_plot(ax2,
                                         sampled_timestamps,
                                         sampled_data,
                                         label='Bed Status', 
                                         color='blue', 
                                         linewidth=2)
                                print("整體在床狀態繪製完成")
                            
                            # 設置圖表格式
                            ax2.xaxis.set_major_locator(MaxNLocator(nbins=12))
                            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
                            ax2.xaxis.set_minor_locator(AutoLocator())
                            ax2.set_ylabel('Events')
                            ax2.set_ylim(-0.2, 1.2)
                            ax2.legend()
                            ax2.grid(True)
                            fig.autofmt_xdate()
                            canvas.draw()
                            
                            # 在更新圖表後強制同步x軸範圍
                            ax2.set_xlim(ax1.get_xlim())

                except Exception as e:
                    print(f"更新圖表和保存數據時發生錯誤: {str(e)}")
            
            # 創建新視窗來顯示圖表
            plot_window = tk.Toplevel(root)
            plot_window.title("Interactive Plot")
            
            # 創建控制面板框架
            control_frame = ttk.Frame(plot_window)
            control_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
            
            # 創建顯示控制變數
            show_vars = {
                'bed_status': tk.BooleanVar(value=True),
                'flip': tk.BooleanVar(value=True),
                'channels': [tk.BooleanVar(value=True) for _ in range(6)]  # 為每個通道創建變數
            }
            
            # 創建事件控制選項
            ttk.Checkbutton(control_frame, text="在床狀態", 
                          variable=show_vars['bed_status'],
                          command=lambda: update_plot()).pack(side=tk.LEFT, padx=5)
            
            ttk.Checkbutton(control_frame, text="翻身狀態", 
                          variable=show_vars['flip'],
                          command=lambda: update_plot()).pack(side=tk.LEFT, padx=5)
            
            # 添加通道控制
            for i in range(6):
                ttk.Checkbutton(control_frame, 
                              text=f"Ch{i+1}",
                              variable=show_vars['channels'][i],
                              command=lambda: update_plot()).pack(side=tk.LEFT, padx=2)
            
            # 使用 matplotlib 的 tkinter 後端
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
            from matplotlib.figure import Figure
            
            # 創建圖表
            fig = Figure(figsize=(14, 10))
            ax1 = fig.add_subplot(211)
            ax2 = fig.add_subplot(212, sharex=ax1)
            
            # 設置時間軸格式器
            def format_time(x, p):
                date = mdates.num2date(x)
                # 獲取當前視圖的x軸範圍
                current_range = ax1.get_xlim()
                range_width = current_range[1] - current_range[0]
                
                # 根據視圖範圍決定時間格式
                if range_width < 0.05:  # 極度放大時（約1小時範圍）
                    return date.strftime('%H:%M:%S')
                elif range_width < 0.5:  # 中等放大時（約12小時範圍）
                    return date.strftime('%H:%M')
                else:  # 正常視圖
                    return date.strftime('%H:%M')
            
            # 使用正確的格式器和定位器
            ax1.xaxis.set_major_formatter(FuncFormatter(format_time))
            
            def update_ticks(axes):
                # 使用 MaxNLocator 限制最大刻度數量
                axes.xaxis.set_major_locator(MaxNLocator(nbins=12))  # 限制最多12個主要刻度
                axes.xaxis.set_minor_locator(AutoLocator())  # 次要刻度自動調整
                
                # 自定義時間格式
                def time_formatter(x, pos=None):
                    dt = mdates.num2date(x)
                    if axes.get_xlim()[1] - axes.get_xlim()[0] < 1/24:  # 小於1小時範圍
                        return dt.strftime('%H:%M:%S')
                    else:
                        return dt.strftime('%m-%d %H:%M')
                
                axes.xaxis.set_major_formatter(FuncFormatter(time_formatter))
            
            # 綁定縮放事件
            def on_xlims_change(axes):
                update_ticks(axes)
            
            ax1.callbacks.connect('xlim_changed', on_xlims_change)
            ax2.callbacks.connect('xlim_changed', on_xlims_change)
            
            # 初始化刻度設置
            update_ticks(ax1)
            update_ticks(ax2)
            
            # 繪製數據（使用時間軸）
            for ch in range(6):
                # 確保數據長度一致
                plot_length = min(len(timestamps), len(processed_data['d10'][ch]))
                ax1.plot(timestamps[:plot_length], 
                        processed_data['d10'][ch][:plot_length], 
                        label=f'CH{ch}', 
                        alpha=0.7)
            
            # 設置x軸格式
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))
            fig.autofmt_xdate()  # 自動旋轉日期標籤
            
            ax1.set_ylabel('Sensor Values')
            ax1.legend()
            ax1.grid(True)
            
            # 添加網格線（包括次要網格）
            ax1.grid(True, which='major', linestyle='-')
            ax1.grid(True, which='minor', linestyle=':')
            ax2.grid(True, which='major', linestyle='-')
            ax2.grid(True, which='minor', linestyle=':')
            
            # 確保時間軸標籤不重疊
            fig.autofmt_xdate()
            
            # 創建畫布
            canvas = FigureCanvasTkAgg(fig, master=plot_window)
            canvas.draw()
            
            # 添加工具欄
            toolbar = NavigationToolbar2Tk(canvas, plot_window)
            toolbar.update()
            
            # 創建一個框架來容納控制元件
            control_frame = ttk.Frame(plot_window)
            control_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

            # 在控制框架中放置更新按鈕
            update_button = ttk.Button(control_frame, text="更新參數", command=update_plot)
            update_button.pack(side=tk.RIGHT, padx=5)  # 改為靠右對齊

            # 確保控制框架在圖表下方
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
            toolbar.pack(side=tk.TOP, fill=tk.X)
            control_frame.pack(side=tk.TOP, fill=tk.X)
            
            # 初始繪製事件
            update_plot()
            
            # 添加更新按鈕
            update_button = ttk.Button(plot_window, text="更新參數", command=update_plot)
            update_button.pack(side=tk.BOTTOM, pady=5)

        except Exception as e:
            messagebox.showerror("Error", f"繪圖錯誤: {str(e)}")
            traceback.print_exc()

    root.after(100, show_plot)


# 查询函数，带有超时检查
def query_and_plot():
    query_done.clear()

    def query_thread():
        serial_id = serial_id_entry.get()
        start_time = start_time_entry.get()
        end_time = end_time_entry.get()

        if not serial_id or not start_time or not end_time:
            messagebox.showwarning("Warning", "请填写所有输入框！")
            query_done.set()
            return

        # 时间检查
        try:
            start_time_dt = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
            end_time_dt = datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")
            if end_time_dt > start_time_dt + timedelta(hours=24):
                messagebox.showwarning("Warning", "End Time 不可超过 Start Time 24 小时以上！")
                query_done.set()
                return
        except ValueError:
            messagebox.showerror("Error", "时间格式错误，请使用 YYYY-MM-DD HH:MM:SS 格式！")
            query_done.set()
            return

        sensor_data = fetch_sensor_data(serial_id, start_time, end_time)

        print(f"sensor_data的長度: {len(sensor_data)}")

        if sensor_data:
            plot_combined_data(sensor_data)
        query_done.set()

    threading.Thread(target=query_thread).start()


def process_sensor_data(sensor_data, params):
    """處理感測器數據"""
    try:
        # 轉換數據格式
        channels = {}
        for ch in range(6):
            ch_data = [row[f'ch{ch}'] for row in sensor_data]
            channels[f'ch{ch}'] = np.array(ch_data)

        # 初始化結果字典
        results = {
            'n10': [],  # 高通濾波後的噪聲值
            'd10': [],  # 移動平均
            'x10': []   # 最大值
        }

        # 處理每個通道
        for ch in range(6):
            data = channels[f'ch{ch}']
            
            # 應用偏移值
            data = data + params.channel_params['offset'][ch]
            
            # 計算高通濾波噪聲值
            hp = np.convolve(data, [-1, -2, -3, -4, 4, 3, 2, 1], mode='same')
            lpf = [26, 28, 32, 39, 48, 60, 74, 90, 108, 126, 146, 167, 187, 208, 
                  227, 246, 264, 280, 294, 306, 315, 322, 326, 328, 326, 322, 315,
                  306, 294, 280, 264, 246, 227, 208, 187, 167, 146, 126, 108, 90,
                  74, 60, 48, 39, 32, 28, 26]
            n = np.convolve(np.abs(hp / 16), lpf, mode='full')
            n = n / 4096  # 修正：確保長度一致
            results['n10'].append(np.int32(n[:len(data)]))  # 確保長度與原始數據相同

            # 計算移動平均
            data_pd = pd.Series(data)
            med10 = data_pd.rolling(window=10, min_periods=1, center=True).mean()
            results['d10'].append(np.int32(med10))

            # 計算移動最大值
            max10 = data_pd.rolling(window=10, min_periods=1, center=True).max()
            results['x10'].append(np.int32(max10))

        # 確保所有數組長度一致
        min_length = min(len(results['n10'][0]), len(results['d10'][0]), len(results['x10'][0]))
        
        # 截斷所有數組到相同長度
        for key in results:
            results[key] = [arr[:min_length] for arr in results[key]]

        return results

    except Exception as e:
        print(f"數據處理錯誤: {str(e)}")
        return None

def calculate_movement_indicators(processed_data, params):
    """計算位移指標和基線"""
    try:
        # 初始化
        l = len(processed_data['d10'][0])
        onbed = np.zeros((l,))
        onload = []
        total = 0
        zdata_final = []
        base_final = []
        
        # 對每個通道進行處理
        for ch in range(6):
            # 取得數據
            max10 = processed_data['x10'][ch] + params.channel_params['offset'][ch]
            med10 = processed_data['d10'][ch] + params.channel_params['offset'][ch]
            n = processed_data['n10'][ch]
            preload = params.channel_params['preload'][ch]
            # print("數據處理完成")
            
            # 零點判定
            zeroing = np.less(n * np.right_shift(max10, 5), 
                            params.noise_2 * np.right_shift(preload, 5))
            
            # 計算基線參數
            th1 = params.channel_params['threshold1'][ch]
            th2 = params.channel_params['threshold2'][ch]
            approach = max10 - (th1 + th2)
            speed = n // (params.noise_1 * 4)
            np.clip(speed, 1, 16, out=speed)
            app_sp = approach * speed
            sp_1024 = 1024 - speed
            # print('計算基線參數完成')
            
            # 動態基線計算
            base = (app_sp[0] // 1024 + med10[0]) // 2
            base = np.int64(base)
            baseline = np.zeros_like(med10)
            
            for i in range(l):
                if zeroing[i]:
                    base = np.int64(med10[i])
                base = (base * sp_1024[i] + app_sp[i]) // 1024
                baseline[i] = base
            # print('動態基線計算完成')
            
            # 計算負載和在床狀態
            total = total + med10[:] - baseline
            o = np.less(th1, med10[:] - baseline)
            onload.append(o)
            onbed = onbed + o
            # print('計算負載和在床狀態完成')
            
            # 保存零點數據和基線
            d_zero = med10 - baseline
            zdata_final.append(d_zero)
            base_final.append(baseline)
            # print('保存零點數據和基線完成')
        
        # 最終在床判定
        onbed = onbed + np.less(params.bed_threshold, total)
        onbed = np.int32(onbed > 0)
        # print('最終在床判定完成')
        
        # 返回所有計算結果
        return {
            'onbed': onbed,
            'onload': onload,
            'zdata': zdata_final,
            'baseline': base_final,
            'total': total
        }
        
    except Exception as e:
        print(f"位移指標計算錯誤: {str(e)}")
        return None

def detect_bed_events(processed_data, params):
    """檢測床上事件（離床/上床/翻身）"""
    try:
        l = len(processed_data['d10'][0])
        events = {
            'bed_status': np.zeros((l,)),
            'movement': [],
            'flip_points': [],  # 改為儲存翻身的時間點
            'rising_dist': np.zeros((l,)),
            'rising_dist_air': np.zeros((l,))
        }
        
        # 計算位移指標
        dist = 0       # 一般床墊
        dist_air = 0   # 氣墊床
        
        for ch in range(6):
            # refer to onoff_bed_0803-H.py line 498
            med10 = processed_data['d10'][ch]
            med10_pd = pd.Series(med10)
            
            # 一般床墊位移計算
            a = [1, -1023/1024]
            b = [1/1024, 0]
            pos_iirmean = lfilter(b, a, med10)
            mean_30sec = med10_pd.rolling(window=30, min_periods=1, center=False).mean()
            mean_30sec = np.int32(mean_30sec)
            diff = (mean_30sec - pos_iirmean) / 256
            if ch == 1:
                diff = diff / 3
            dist = dist + np.square(diff)
            dist[dist > 8000000] = 8000000
            
            # 氣墊床位移計算
            mean_60sec = med10_pd.rolling(window=60, min_periods=1, center=False).mean()
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
        shift = 60
        # 一般床墊
        dist_series = pd.Series(dist)
        rising_dist = dist_series.shift(-shift) - dist_series
        rising_dist = rising_dist.fillna(0)
        rising_dist = np.int32(rising_dist)
        rising_dist[rising_dist < 0] = 0
        rising_dist = rising_dist // 127
        rising_dist[rising_dist > 1000] = 1000
        
        # 氣墊床
        dist_air_series = pd.Series(dist_air)
        rising_dist_air = dist_air_series.shift(-shift) - dist_air_series
        rising_dist_air = rising_dist_air.fillna(0)
        rising_dist_air = np.int32(rising_dist_air)
        rising_dist_air[rising_dist_air < 0] = 0
        rising_dist_air = rising_dist_air // 127
        rising_dist_air[rising_dist_air > 1000] = 1000
        
        # 儲存位移結果
        events['rising_dist'] = rising_dist
        events['rising_dist_air'] = rising_dist_air
        
        # 檢測翻身事件並找出中間點
        if params.is_air_mattress == 0:
            flip_mask = (rising_dist > params.movement_threshold)
        else:
            flip_mask = (rising_dist_air > params.movement_threshold)
            
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
        
        # 計算每個翻身區間的中間點
        events['flip_points'] = [(start + (end - start)//2) for start, end in flip_regions]
        
        # 取得在床狀態
        movement_data = calculate_movement_indicators(processed_data, params)
        if movement_data:
            events['bed_status'] = movement_data['onbed']
            events['onload'] = movement_data['onload']
        
        return events
        
    except Exception as e:
        print(f"事件檢測錯誤: {str(e)}")
        return None

def save_processed_data(sensor_data, processed_data, parameters):
    """保存處理後的數據"""
    try:
        # 建立時間戳列
        timestamps = [
            datetime.strptime(str(row['timestamp']), "%Y%m%d%H%M%S") + timedelta(hours=8)
            for row in sensor_data
        ][:len(processed_data['d10'][0])]  # 嚴格對齊處理後數據長度

        # 取得最短的數據長度，確保所有數據長度一致
        min_length = min(
            len(timestamps),
            min(len(arr) for arr in processed_data['d10']),
            min(len(arr) for arr in processed_data['n10']),
            min(len(arr) for arr in processed_data['x10'])
        )
        
        # 截取所有數據到相同長度
        timestamps = timestamps[:min_length]
        
        # 準備數據字典
        data_dict = {
            'Timestamp': timestamps
        }
        
        # 添加各通道的數據
        for ch in range(6):
            data_dict[f'Channel_{ch+1}_Raw'] = processed_data['d10'][ch][:min_length]
            data_dict[f'Channel_{ch+1}_Noise'] = processed_data['n10'][ch][:min_length]
            data_dict[f'Channel_{ch+1}_Max'] = processed_data['x10'][ch][:min_length]
        
        # 添加位移數據
        movement_data = calculate_movement_indicators(processed_data, parameters)
        data_dict['Rising_Dist_Normal'] = movement_data['rising_dist'][:min_length]
        data_dict['Rising_Dist_Air'] = movement_data['rising_dist_air'][:min_length]
        
        # 檢查所有數據長度是否一致
        lengths = [len(arr) for arr in data_dict.values()]
        if len(set(lengths)) > 1:
            print("Warning: Data lengths before saving:", lengths)
            min_length = min(lengths)
            for key in data_dict:
                data_dict[key] = data_dict[key][:min_length]
        
        # 保存為CSV
        df = pd.DataFrame(data_dict)
        filename = f"local_processed_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        filepath = os.path.join(DATA_DIR, filename)
        df.to_csv(filepath, index=False)
        
        print(f"Successfully saved data with length {min_length}")
        return filepath
        
    except Exception as e:
        print(f"Error details: {str(e)}")
        messagebox.showerror("Error", f"保存數據時發生錯誤: {str(e)}")
        return None

# 在主程式中初始化參數表
def setup_main_window():
    global parameter_table
    
    # 初始化參數表
    parameter_table = init_parameter_table(root)
    
    # 查詢按鈕
    query_button = ttk.Button(root, text="Search", command=query_and_plot)
    query_button.grid(row=5, column=0, columnspan=2, pady=10)  # 將按鈕移到參數表下方

# 在主程式開始時調用
if __name__ == "__main__":
    root = tk.Tk()
    root.title("Sensor and Notify Data Viewer")
    
    # 初始化參數表和其他UI元素
    setup_main_window()
    
    # 获取当天早上 12:00:00 和前一天 12:00:00 的时间
    current_date = datetime.now().date()
    default_start_time = datetime.combine(current_date, datetime.min.time()) - timedelta(hours=12)
    default_end_time = datetime.combine(current_date, datetime.min.time()) + timedelta(hours=12)
 
    # 输入框和标签
    ttk.Label(root, text="Serial ID:").grid(row=0, column=0, padx=5, pady=5) # default: SPS2021PA000456
    serial_id_entry = ttk.Entry(root)
    serial_id_entry.insert(0, 'SPS2025PA000146')
    serial_id_entry.grid(row=0, column=1, padx=5, pady=5)

    ttk.Label(root, text="Start Time (YYYY-MM-DD HH:MM:SS):").grid(row=2, column=0, padx=5, pady=5)
    start_time_entry = ttk.Entry(root)
    # start_time_entry.insert(0, default_start_time.strftime("%Y-%m-%d %H:%M:%S"))
    start_time_entry.insert(0, '2025-03-08 12:00:00')
    start_time_entry.grid(row=2, column=1, padx=5, pady=5)

    ttk.Label(root, text="End Time (YYYY-MM-DD HH:MM:SS):").grid(row=3, column=0, padx=5, pady=5)
    end_time_entry = ttk.Entry(root)
    # end_time_entry.insert(0, default_end_time.strftime("%Y-%m-%d %H:%M:%S"))
    end_time_entry.insert(0, '2025-03-09 12:00:00')
    end_time_entry.grid(row=3, column=1, padx=5, pady=5)

    # 启动主循环
    root.mainloop()



