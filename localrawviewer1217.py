import pymysql
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import threading

# sensor_data 数据库连接信息
db_sensor_config = {
    "host": "61.220.29.122",
    "port": 23060,
    "user": "readonly_user",
    "password": "jF@0McVu1KcP",
    "database": "humetrics"
}

query_done = threading.Event()  # 查询完成状态标记


# 函数：从 sensor_data 获取数据
def fetch_sensor_data(serial_id, start_time, end_time):
    query = """
        SELECT serial_id, ch0, ch1, ch2, ch3, ch4, ch5, timestamp, created_at 
        FROM sensor_data 
        WHERE serial_id = %s 
        AND created_at >= %s 
        AND created_at <= %s
        ORDER BY timestamp ASC;
    """
    try:
        connection = pymysql.connect(**db_sensor_config)
        cursor = connection.cursor(pymysql.cursors.DictCursor)
        cursor.execute(query, (serial_id, start_time, end_time))
        result = cursor.fetchall()

        for row in result:
            timestamp_str = str(row['timestamp'])
            timestamp_dt = datetime.strptime(timestamp_str, "%Y%m%d%H%M%S")
            time_at = timestamp_dt + timedelta(hours=8)
            row['time_at'] = time_at.strftime("%Y-%m-%d %H:%M:%S")

        return result
    except Exception as e:
        messagebox.showerror("Error", f"Sensor Data Fetch Error: {e}")
        return None
    finally:
        if 'connection' in locals():
            connection.close()

# 函数：绘制合并图表
def plot_combined_data(sensor_data):
    if not sensor_data and not notify_data:
        messagebox.showwarning("Warning", "缺少数据，无法绘制图表。")
        return

    # 将 sensor_data 转换为 DataFrame
    sensor_df = pd.DataFrame(sensor_data)
    if not sensor_df.empty:
        sensor_df['time_at'] = pd.to_datetime(sensor_df['time_at'])
        sensor_df.set_index('time_at', inplace=True)
        sensor_df.sort_index(inplace=True)

        # 计算时间差，标记超过 10 分钟的间隔为 NaN
        time_diff = sensor_df.index.to_series().diff().dt.total_seconds()
        for ch in ['ch0', 'ch1', 'ch2', 'ch3', 'ch4', 'ch5']:
            sensor_df.loc[time_diff > 1000, ch] = None
    else:
        messagebox.showinfo("Info", "Sensor Data 数据为空。")

    # 绘制图表
    fig, ax1 = plt.subplots(figsize=(14, 7))

    # 绘制 sensor_data 的 y 轴（左侧）
    if not sensor_df.empty:
        for ch in ['ch0', 'ch1', 'ch2', 'ch3', 'ch4', 'ch5']:
            if ch in sensor_df.columns:
                ax1.plot(sensor_df.index, sensor_df[ch], label=ch, alpha=0.7)
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Sensor Values')
        ax1.legend(loc='upper left')
        ax1.grid()

    # 显示图表
    plt.title('Combined Sensor and Notify Data with Interactive Clicks')
    plt.show()


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
ttk.Label(root, text="Serial ID:").grid(row=0, column=0, padx=5, pady=5)
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