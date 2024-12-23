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

# 確保 log_file 目錄存在
LOG_DIR = "./_log_file"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

DATA_DIR = "./_data"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# sensor_data 数据库连接信息
db_sensor_config = {
    "host": "61.220.29.122",
    "port": 23060,
    "user": "readonly_user",
    "password": "jF@0McVu1KcP",
    "database": "humetrics"
}

query_done = threading.Event()  # 查询完成状态标记

# 在文件開頭添加全局變數
global startday, n10, d10, x10, data_resp, rising_dist, rising_dist_air, base_final, parameter_table

# 在文件開頭添加必要的參數結構
class BedParameters:
    def __init__(self):
        # 基本閾值
        self.bed_threshold = 0      # 判斷在床/離床的基本閾值
        self.noise_onbed = 0        # 在床時的噪音閾值
        self.noise_offbed = 0       # 離床時的噪音閾值 (Noise 2)
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
    
    # 預設的通道參數值
    default_channel_values = [
        [60000, 60000, 60000, 60000, 60000, 60000],  # min_preload
        [100000, 100000, 100000, 100000, 100000, 100000],  # threshold_1
        [100000, 100000, 100000, 100000, 100000, 100000],  # threshold_2
        [0, 0, 0, 0, 0, 0]               # offset level
    ]
    
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
            entry.insert(0, str(default_channel_values[i][j]))  # 設置預設值
            entry.grid(row=i+1, column=j+1, padx=2, pady=2)
            row_entries.append(entry)
        entries.append(row_entries)
    
    # 添加獨立參數
    single_params = ['Total Sum', 'Noise 1', 'Noise 2', 'Set Flip']
    single_entries = []
    
    # 獨立參數的預設值
    default_single_values = [30000, 60, 60, 200]  # Total Sum, Noise 1, Set Flip
    
    # 創建獨立參數的標籤和輸入框
    single_frame = ttk.Frame(root)
    for i, param in enumerate(single_params):
        # 標籤
        label = ttk.Label(single_frame, text=param, anchor="e")
        label.grid(row=i, column=0, padx=5, pady=2)
        
        # 輸入框
        entry = ttk.Entry(single_frame, width=8)
        entry.insert(0, str(default_single_values[i]))  # 設置預設值
        entry.grid(row=i, column=1, padx=5, pady=2)
        single_entries.append(entry)
    
    # 將表格和獨立參數框架添加到主窗口
    table_frame.grid(row=4, column=0, columnspan=7, padx=5, pady=5)
    single_frame.grid(row=4, column=7, columnspan=3, padx=5, pady=5, sticky="nw")
    
    return {
        'frame': table_frame,
        'entries': entries,
        'single_frame': single_frame,
        'single_entries': single_entries
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
        params.noise_onbed = int(parameter_table['single_entries'][1].get())      # Noise 1
        params.noise_offbed = int(parameter_table['single_entries'][2].get())     # Noise 2
        params.movement_threshold = int(parameter_table['single_entries'][3].get()) # Set Flip
        
    except ValueError as e:
        messagebox.showerror("Error", f"參數格式錯誤: {str(e)}")
        return None
    except Exception as e:
        messagebox.showerror("Error", f"參數獲取錯誤: {str(e)}")
        traceback.print_exc()
        return None
        
    return params

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
        
        filepath = os.path.join(DATA_DIR, filename)
        
        # 準備要存檔的數據
        save_data = {
            'serial_id': serial_id_entry.get(),
            'start_time': start_time_entry.get(),
            'end_time': end_time_entry.get(),
            'data': []
        }
        
        # 整理數據格式
        for i in range(len(sensor_data)):
            save_data['data'].append({
                'timestamp': sensor_data[i]['timestamp'],
                'ch0': sensor_data[i]['ch0'],
                'ch1': sensor_data[i]['ch1'],
                'ch2': sensor_data[i]['ch2'],
                'ch3': sensor_data[i]['ch3'],
                'ch4': sensor_data[i]['ch4'],
                'ch5': sensor_data[i]['ch5'],
                'created_at': sensor_data[i]['created_at']
            })
        
        # 寫入 JSON 檔案
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, default=str)
            
        return filepath
        
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
        local_file = check_local_data(serial_id, start_time, end_time)
        if local_file:
            return load_sensor_data(local_file)
            
        # 如果沒有本地檔案，從資料庫獲取
        
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
            
    except Exception as e:
        messagebox.showerror("Error", f"Sensor Data Fetch Error: {e}")
        return None
    finally:
        if 'connection' in locals():
            connection.close()

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
                
            # 處理數據
            processed_data = process_sensor_data(sensor_data, params)
            if processed_data is None:
                messagebox.showerror("Error", "數據處理失敗")
                return
            
            # 檢測事件
            events = detect_bed_events(processed_data, params)
            if events is None:
                messagebox.showerror("Error", "事件檢測失敗")
                return
                
            # 繪製圖表
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
            
            # 上半部分：原始數據
            for ch in range(6):
                ax1.plot(processed_data['d10'][ch], label=f'CH{ch}', alpha=0.7)
            ax1.set_ylabel('Sensor Values')
            ax1.legend()
            ax1.grid(True)
            
            # 下半部分：事件標記
            ax2.plot(events['bed_status'], label='Bed Status', color='blue')
            ax2.plot(events['movement'], label='Movement', color='green')
            ax2.plot(events['flip'], label='Flip', color='red')
            ax2.set_ylabel('Events')
            ax2.set_ylim(-0.2, 1.2)
            ax2.legend()
            ax2.grid(True)
            
            plt.xlabel('Time')
            plt.show()

        except Exception as e:
            messagebox.showerror("Error", f"繪圖錯誤: {str(e)}")

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
            'd10': [],  # 10點移動平均
            'x10': [],  # 10點最大值
            'base_final': [], # 基準值
            'zdata_final': [] # 零點校正後數據
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
            n = n[10:-37] / 4096
            n = n[::10]
            results['n10'].append(np.int32(n))

            # 計算移動平均
            data_pd = pd.Series(data)
            med10 = data_pd.rolling(window=10, min_periods=1, center=True).mean()
            med10 = np.array(med10)
            med10 = med10[::10]
            results['d10'].append(np.int32(med10))

            # 計算移動最大值
            max10 = data_pd.rolling(window=10, min_periods=1, center=True).max()
            max10 = np.array(max10)
            max10 = max10[::10]
            results['x10'].append(np.int32(max10))

        return results

    except Exception as e:
        print(f"數據處理錯誤: {str(e)}")
        return None

def calculate_movement_indicators(processed_data, params):
    """計算位移指標"""
    try:
        dist = 0
        dist_air = 0
        
        for ch in range(6):
            med10 = processed_data['d10'][ch]
            
            # 計算一般床墊位移
            a = [1, -1023/1024]
            b = [1/1024, 0]
            pos_iirmean = lfilter(b, a, med10)
            
            med10_pd = pd.Series(med10)
            mean_30sec = med10_pd.rolling(window=30, min_periods=1, center=False).mean()
            mean_30sec = np.int32(mean_30sec)
            
            diff = (mean_30sec - pos_iirmean) / 256
            if ch == 1:
                diff = diff / 3
            dist = dist + np.square(diff)
            dist[dist > 8000000] = 8000000

            # 計算空氣床墊位移
            mean_60sec = med10_pd.rolling(window=60, min_periods=1, center=False).mean()
            mean_60sec = np.int32(mean_60sec)
            
            a = np.zeros([780,])
            a[0] = 1
            b = np.zeros([780,])
            for s in range(10):
                b[s*60 + 180] = -0.1
            b[60] = 1
            
            diff = lfilter(b, a, mean_60sec)
            if ch == 1:
                diff = diff / 3
            dist_air = dist_air + np.square(diff / 256)

        # 計算位移差值
        dist = pd.Series(dist)
        rising_dist = dist.shift(-60) - dist
        rising_dist = rising_dist.fillna(0)
        rising_dist = np.int32(rising_dist)
        rising_dist[rising_dist < 0] = 0
        rising_dist = rising_dist // 127
        rising_dist[rising_dist > 1000] = 1000

        # 計算空氣床墊位移差值
        dist_air = pd.Series(dist_air)
        rising_dist_air = dist_air.shift(-60) - dist_air
        rising_dist_air = rising_dist_air.fillna(0)
        rising_dist_air = np.int32(rising_dist_air)
        rising_dist_air[rising_dist_air < 0] = 0
        rising_dist_air = rising_dist_air // 127
        rising_dist_air[rising_dist_air > 1000] = 1000

        return {
            'rising_dist': rising_dist,
            'rising_dist_air': rising_dist_air
        }

    except Exception as e:
        print(f"位移指標計算錯誤: {str(e)}")
        return None

def save_processed_data(sensor_data, processed_data, parameters):
    """保存處理後的數據"""
    try:
        # 建立時間戳列
        timestamps = []
        for row in sensor_data:
            # 從 timestamp 字串轉換為 datetime 對象
            timestamp_str = str(row['timestamp'])
            timestamp_dt = datetime.strptime(timestamp_str, "%Y%m%d%H%M%S")
            # 加上 8 小時時區調整
            time_at = timestamp_dt + timedelta(hours=8)
            timestamps.append(time_at.strftime("%Y-%m-%d %H:%M:%S"))
        
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
        filename = f"processed_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        filepath = os.path.join(DATA_DIR, filename)
        df.to_csv(filepath, index=False)
        
        print(f"Successfully saved data with length {min_length}")
        return filepath
        
    except Exception as e:
        print(f"Error details: {str(e)}")
        messagebox.showerror("Error", f"保存數據時發生錯誤: {str(e)}")
        return None

def detect_bed_events(processed_data, params):
    """檢測床上事件（離床/上床/翻身）
    
    Args:
        processed_data: 處理過的感測器數據，包含：
            - d10: 10點移動平均
            - n10: 高通濾波後的噪聲值
            - x10: 10點最大值
        params: BedParameters 物件，包含：
            - bed_threshold: 判斷在床/離床的基本閾值
            - noise_onbed: 在床時的噪音閾值
            - movement_threshold: 移動判定閾值
            
    Returns:
        dict: 包含各種事件的時間點和類型
    """
    try:
        # 初始化結果
        events = {
            'bed_status': [],  # 0:離床, 1:在床
            'movement': [],    # 0:靜止, 1:移動
            'flip': [],       # 0:無翻身, 1:翻身
            'timestamps': []   # 事件發生的時間點
        }
        
        # 計算總和信號
        total_signal = np.zeros(len(processed_data['d10'][0]))
        for ch in range(6):
            total_signal += processed_data['d10'][ch]
        
        # 計算總和噪聲
        total_noise = np.zeros(len(processed_data['n10'][0]))
        for ch in range(6):
            total_noise += processed_data['n10'][ch]
        
        # 逐點檢測狀態
        for i in range(len(total_signal)):
            current_status = {
                'bed': 0,
                'move': 0,
                'flip': 0
            }
            
            # 1. 離床/在床檢測
            if total_signal[i] > params.bed_threshold:
                current_status['bed'] = 1
                # 在床時使用 noise_onbed
                noise_threshold = params.noise_onbed
            else:
                # 離床時使用 noise_offbed
                noise_threshold = params.noise_offbed
            
            # 2. 移動檢測
            if total_noise[i] > noise_threshold:
                current_status['move'] = 1
            
            # 3. 翻身檢測
            if (current_status['move'] == 1 and 
                total_noise[i] > params.movement_threshold):
                current_status['flip'] = 1
            
            # 保存結果
            events['bed_status'].append(current_status['bed'])
            events['movement'].append(current_status['move'])
            events['flip'].append(current_status['flip'])
            
        return events
        
    except Exception as e:
        print(f"事件檢測錯誤: {str(e)}")
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
    
    # 获取当天早上 6:00:00 和中午 12:00:00 的时间
    # current_date = datetime.now().date()
    # 指定為2024-12-16
    current_date = datetime(2024, 12, 18)
    default_start_time = datetime.combine(current_date, datetime.min.time()) - timedelta(hours=12)
    default_end_time = datetime.combine(current_date, datetime.min.time()) + timedelta(hours=12)

    # 输入框和标签
    ttk.Label(root, text="Serial ID:").grid(row=0, column=0, padx=5, pady=5) # default: SPS2021PA000456
    serial_id_entry = ttk.Entry(root)
    serial_id_entry.insert(0, 'SPS2021PA000456')
    serial_id_entry.grid(row=0, column=1, padx=5, pady=5)

    ttk.Label(root, text="Start Time (YYYY-MM-DD HH:MM:SS):").grid(row=2, column=0, padx=5, pady=5)
    start_time_entry = ttk.Entry(root)
    start_time_entry.insert(0, default_start_time.strftime("%Y-%m-%d %H:%M:%S"))
    start_time_entry.grid(row=2, column=1, padx=5, pady=5)

    ttk.Label(root, text="End Time (YYYY-MM-DD HH:MM:SS):").grid(row=3, column=0, padx=5, pady=5)
    end_time_entry = ttk.Entry(root)
    end_time_entry.insert(0, default_end_time.strftime("%Y-%m-%d %H:%M:%S"))
    end_time_entry.grid(row=3, column=1, padx=5, pady=5)

    # 启动主循环
    root.mainloop()


