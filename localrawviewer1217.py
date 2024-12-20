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
            
        # 如果沒有本地檔案，則從資料庫獲取
        
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
        # 將 sensor_data 轉換為 DataFrame
        sensor_df = pd.DataFrame(sensor_data)
        
        # 確保有時間欄位並轉換為 datetime
        if 'time_at' not in sensor_df.columns and 'timestamp' in sensor_df.columns:
            sensor_df['time_at'] = pd.to_datetime(sensor_df['timestamp'].astype(str), format='%Y%m%d%H%M%S')
            sensor_df['time_at'] = sensor_df['time_at'] + pd.Timedelta(hours=8)
        else:
            # 確保 time_at 是 datetime 格式
            sensor_df['time_at'] = pd.to_datetime(sensor_df['time_at'])
            
        if not sensor_df.empty:
            sensor_df.set_index('time_at', inplace=True)
            sensor_df.sort_index(inplace=True)

            # 计算时间差，标记超过 10 分钟的间隔为 NaN
            time_diff = sensor_df.index.to_series().diff().dt.total_seconds()
            for ch in ['ch0', 'ch1', 'ch2', 'ch3', 'ch4', 'ch5']:
                sensor_df.loc[time_diff > 1000, ch] = None
        else:
            messagebox.showinfo("Info", "Sensor Data 数据为空。")

        # 绘制图表
        plt.figure(figsize=(14, 7))
        
        if not sensor_df.empty:
            for ch in ['ch0', 'ch1', 'ch2', 'ch3', 'ch4', 'ch5']:
                if ch in sensor_df.columns:
                    plt.plot(sensor_df.index, sensor_df[ch], label=ch, alpha=0.7)
            plt.xlabel('Time')
            plt.ylabel('Sensor Values')
            plt.legend(loc='upper left')
            plt.grid()

        plt.title('Combined Sensor and Notify Data')
        plt.show()

    # 在主线程中执行绘图
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


# 主窗口设置
root = tk.Tk()
root.title("Sensor and Notify Data Viewer")

# 获取当天早上 6:00:00 和中午 12:00:00 的时间
current_date = datetime.now().date()
default_start_time = datetime.combine(current_date, datetime.min.time()) - timedelta(hours=12)
default_end_time = datetime.combine(current_date, datetime.min.time()) + timedelta(hours=12)

# 输入框和标签
ttk.Label(root, text="Serial ID:").grid(row=0, column=0, padx=5, pady=5) # default: SPS2021PA000456
serial_id_entry = ttk.Entry(root)
serial_id_entry.grid(row=0, column=1, padx=5, pady=5)

ttk.Label(root, text="Start Time (YYYY-MM-DD HH:MM:SS):").grid(row=2, column=0, padx=5, pady=5)
start_time_entry = ttk.Entry(root)
start_time_entry.insert(0, default_start_time.strftime("%Y-%m-%d %H:%M:%S"))
start_time_entry.grid(row=2, column=1, padx=5, pady=5)

ttk.Label(root, text="End Time (YYYY-MM-DD HH:MM:SS):").grid(row=3, column=0, padx=5, pady=5)
end_time_entry = ttk.Entry(root)
end_time_entry.insert(0, default_end_time.strftime("%Y-%m-%d %H:%M:%S"))
end_time_entry.grid(row=3, column=1, padx=5, pady=5)

# 查询按钮
query_button = ttk.Button(root, text="Search", command=query_and_plot)
query_button.grid(row=4, column=0, columnspan=2, pady=10)

# 启动主循环
root.mainloop()

