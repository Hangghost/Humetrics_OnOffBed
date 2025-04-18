# import os
# import ssl
# import paho.mqtt.client as mqtt

# def change_NightMode():
#     legend_raw = some_value  # 確保這行代碼存在且正確
#     legend_bg = some_other_value
#     legend_raw.setBrush(legend_bg)

# def MQTT_get_reg(host, device, key, text):
#     client = mqtt.Client()
#     certificate_path = '/Users/chenhunglun/Documents/Procjects/Humetrics_RR/humetric_mqtt_certificate.pem'
    
#     if not os.path.isfile(certificate_path):
#         print(f"Certificate file not found at {certificate_path}")
#         return None  # 返回 None 或其他適當的錯誤處理
#     else:
#         client.tls_set(certificate_path, None, None, cert_reqs=ssl.CERT_NONE)
    
#     # 你的其他 MQTT 配置和連接代碼
#     # 例如：
#     # client.connect(host)
#     # client.loop_start()
    
#     return client

# def OpenCMBDialog():
#     OpenCmbFile()

# def OpenCmbFile():
#     reg_table = MQTT_get_reg("mqtt.humetrics.ai", "device", "!dF-9DXbpVKHDRgBryRJJBEdqCihwN", iCueSN.text())
#     if reg_table is None:
#         print("Failed to get MQTT registration")
#         return






import os
#import struct
import numpy as np
import pandas as pd
import csv
#from sklearn.cluster import KMeans
#from scipy.interpolate import interp1d
from scipy.signal import savgol_filter, lfilter
from PyQt5 import QtWidgets
from pyqtgraph.Qt import QtCore
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QTableWidget, QTableWidgetItem, QRadioButton, QCheckBox, qApp, QHeaderView, QLabel
import pyqtgraph as pg
from PyQt5.QtCore import Qt, QSizeF
from PyQt5.QtGui import QPainter, QPdfWriter, QImage, QIcon, QColor
import paho.mqtt.client as mqtt
#from hyperopt import hp, fmin, tpe, Trials
#import sys
import shutil
import time
from datetime import datetime, timedelta
import json
import ssl
#import threading

global TAB_K
TAB_K = 8

# 確保 log_file 目錄存在
LOG_DIR = "./_logs/pyqt-viewer"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

DATA_DIR = "./_data/pyqt_viewer"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

#--------------------------------------------------------------------------
class CustomViewBox(pg.ViewBox):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setMouseMode(self.PanMode)  # Set initial mode to pan mode

    def mouseDragEvent(self, ev, axis=None):
        if ev.button() == QtCore.Qt.LeftButton:
            if axis is None:
                axis = [True, True]  # Allow panning on both X and Y axes by default

            # If the mouse is on the left side of the y-axis, allow Y-axis panning
            if ev.pos().x() >= self.boundingRect().left() + 50:
                axis = 0  # Disallow Y-axis panning
            
            super().mouseDragEvent(ev, axis=axis)
        else:
            super().mouseDragEvent(ev, axis=axis)

    def wheelEvent(self, ev, axis=None):
        axis = 0  # Only allow zooming on the X axis
        super().wheelEvent(ev, axis=axis)
#--------------------------------------------------------------------------
class PlotWidgetWithZoom(pg.PlotWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(viewBox=CustomViewBox(), *args, **kwargs)
        self.plotItem.getAxis('bottom').setPen(pg.mkPen(color=(128, 128, 128), style=QtCore.Qt.DotLine)) # Set X-axis grid color to green
        # 移除這裡的 vLine 初始化
        
    def wheelEvent(self, event):
        # Determine mouse position in plot coordinates
        mousePoint = self.plotItem.vb.mapSceneToView(event.pos())

        # Check if the mouse is on the left side of the y-axis
        if event.pos().x() < self.plotItem.vb.boundingRect().left() + 50:
            # Determine zoom factor based on scroll direction
            factor = 0.9 if event.angleDelta().y() > 0 else 1.1
            center = mousePoint.y()  # Center point for scaling
            self.plotItem.vb.scaleBy((1, factor), center=(0, center))  # Scale y-axis around center
        else:
            super().wheelEvent(event)  # Call default wheel event behavior

#--------------------------------------------------------------------------
class TimeAxisItem(pg.AxisItem): 
    # 覆寫 tickStrings 方法，根據 values 參數的值來取得對應的時間字串
    # 回傳一個清單，清單中每個元素都是呼叫 format_time 方法後所得到的結果
    def tickStrings(self, values, scale, spacing):
        return [self.format_time(value) for value in values]
    
    def tickSpacing(self, minVal, maxVal, size):
        # 計算時間跨度
        time_span = maxVal - minVal
        
        # 根據時間跨度自動決定刻度間隔
        if time_span <= 95:  # 小於等於10
            return [(1, 0)]        
        #elif time_span <= 65:  # 小於等於1分鐘
        #    return [(5, 0)]
        #elif time_span <= 180:  # 小於等於2分鐘
        #    return [(10, 0)]        
        elif time_span <= 330:  # 小於等於5分鐘
            return [(10, 0)]
        elif time_span <= 5400:  # 小於等於1小時
            return [(300, 0)]
        elif time_span <= 22000:  # 小於等於6小時
            return [(1800, 0)]
        elif time_span <= 44000:  # 小於等於12小時
            return [(3600, 0)]
        elif time_span <= 130000:  # 小於等於1.5天
            return [(7200, 0)]
        else:
            return [(43200, 0)]
        
    # 定義一個靜態方法 format_time，用來格式化時間
    # 計算小時、分鐘、秒數
    # 回傳一個字串，格式為「時:分:秒」，其中時、分、秒都是兩位數
    @staticmethod
    def format_time(value):
        day = value // 86400
        value = value % 86400
        if value == 0: 
            global startday
            return (startday + timedelta(days=day)).strftime('%Y/%m/%d')
        else:
            hours = int(value // 3600)
            minutes = int((value % 3600) // 60)
            seconds = int(value % 60)
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
       
#-----------------------------------------------------------------------
# 建立主應用程式視窗
app = QtWidgets.QApplication([])
mw = QtWidgets.QMainWindow()
cw = QtWidgets.QWidget()
mw.setCentralWidget(cw)

time_axis1 = TimeAxisItem(orientation='bottom')
wav_plot = PlotWidgetWithZoom(axisItems={'bottom': time_axis1})
wav_plot.plotItem.setMouseEnabled(y=False)
wav_plot.showGrid(x = True, y = False, alpha = 1) 

time_axis2 = TimeAxisItem(orientation='bottom')
raw_plot = PlotWidgetWithZoom(axisItems={'bottom': time_axis2})
raw_plot.showGrid(x = True, alpha = 1)

time_axis3 = TimeAxisItem(orientation='bottom')
bit_plot = PlotWidgetWithZoom(axisItems={'bottom': time_axis3})
bit_plot.plotItem.setMouseEnabled(y=False)

global bcg_width
bcg_width = 20

#--------------------------------------------------------------------------
def MQTT_get_reg(mqtt_server, username, password, sn):

    timeout = 30
    reg_table = {}
    start_time = time.time()

    # Define the callback function for incoming messages
    def on_message_reg_get(client, userdata, message):
        reg = json.loads(message.payload)
        del reg["taskID"]
        reg_table.update(reg)
        status_bar.showMessage('MQTT:' + str(reg))
        QApplication.processEvents()
        nonlocal timeout
        if int(list(reg.keys())[0]) >= 63:
            timeout = 30
        else:
            timeout = 120

    # Create a MQTT client
    client = mqtt.Client()

    # Set the callback function
    client.on_message = on_message_reg_get

    client.username_pw_set(username, password)
    if radio_Normal.isChecked():
        # 使用相對路徑獲取憑證檔案
        cert_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),  # 獲取當前腳本所在目錄
            'cert',  # 建議將憑證放在專案的 cert 資料夾下
            'humetric_mqtt_certificate.pem'
        )
        try:
            if not os.path.exists(cert_path):
                raise FileNotFoundError(f"找不到憑證檔案：{cert_path}")
            client.tls_set(cert_path, None, None, cert_reqs=ssl.CERT_NONE)
        except Exception as e:
            print(f"設定 TLS 時發生錯誤：{str(e)}")
            return
        client.connect(mqtt_server, 8883, 60)
    else:
        client.connect(mqtt_server, 1883, 60)

    topic_get_regs = "algoParam/" + sn + "/get"
    # Subscribe to the topic
    client.subscribe(topic_get_regs)

    # Create the message payload / Publish the message
    topic_reg_mode = "systemManage/" + sn
    payload = {"command": "08", "parameter": "", "taskID": 0}
    client.publish(topic_reg_mode, json.dumps(payload))

    # Start a background thread to handle MQTT network traffic
    while True:
        client.loop()
        current_time = time.time()
        if current_time - start_time > timeout:
            status_bar.showMessage(f"MQTT {timeout} seconds timeout reached !")
            break        
        if "999" in reg_table.keys():
            break

    client.disconnect() # Stop the MQTT client loop when done

    return reg_table


#-----------------------------------------------------------------------
# 定義一組十六進位顏色碼
hex_colors = ('FF7575', 'FF9D6F', '66B3FF', '93FF93', 'B8B8DC', '8E8E8E', 'E8D191', 'B0B9A1', '639AA9')

# 定義一個函式，用於將十六進位顏色碼轉換為 RGB 值
def hex_to_rgb(hex_color):
    # 從十六進位顏色碼中提取紅色、綠色和藍色分量
    red = int(hex_color[0:2], 16)
    green = int(hex_color[2:4], 16)
    blue = int(hex_color[4:6], 16)
    return (red, green, blue)
#-----------------------------------------------------------------------   
def toggle_marker():
    """切換標記線的顯示狀態"""
    # 檢查是否已載入資料
    if not hasattr(raw_plot, 'vLine') or not hasattr(bit_plot, 'vLine'):
        status_bar.showMessage('請先載入資料!')
        return
    
    try:
        if raw_plot.vLine.isVisible():
            raw_plot.vLine.hide()
            bit_plot.vLine.hide()
            marker_btn.setText('開始標記')
            status_bar.showMessage('標記模式已關閉')
        else:
            # 取得目前視圖的中心位置
            x_range = raw_plot.viewRange()[0]
            center_x = (x_range[0] + x_range[1]) / 2
            
            # 設定標記線的位置到視圖中心
            raw_plot.vLine.setPos(center_x)
            bit_plot.vLine.setPos(center_x)
            
            # 移除舊的連接（如果存在）
            try:
                raw_plot.vLine.sigPositionChanged.disconnect()
                bit_plot.vLine.sigPositionChanged.disconnect()
            except:
                pass
                
            # 同步兩個圖表的標記線
            raw_plot.vLine.sigPositionChanged.connect(
                lambda: bit_plot.vLine.setPos(raw_plot.vLine.value())
            )
            bit_plot.vLine.sigPositionChanged.connect(
                lambda: raw_plot.vLine.setPos(bit_plot.vLine.value())
            )
            
            raw_plot.vLine.show()
            bit_plot.vLine.show()
            marker_btn.setText('結束標記')
            status_bar.showMessage('標記模式已開啟')
    except Exception as e:
        status_bar.showMessage(f'標記線錯誤: {str(e)}')
#-----------------------------------------------------------------------   
def save_marker():
    """儲存目前標記線的位置"""
    if not raw_plot.vLine.isVisible():
        status_bar.showMessage('請先開啟標記線!')
        return
        
    if not 't1sec' in globals():
        status_bar.showMessage('請先載入資料!')
        return

    # 取得標記線位置的時間
    marker_pos = raw_plot.vLine.value()
    
    # 使用 startday 作為基準日期，加上標記位置的秒數
    timestamp = startday + timedelta(seconds=int(marker_pos))
    
    # 儲存標記
    filename = f"{cmb_name[:-4]}_manual_marks.csv"
    filepath = os.path.join(DATA_DIR, filename)
    
    # 檢查檔案是否存在，決定是否需要寫入標題
    file_exists = os.path.isfile(filepath)
    
    # 取得目前選擇的事件類型
    event_type = marker_type_combo.currentText()
    
    with open(filepath, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:  # 如果是新檔案，寫入標題列
            writer.writerow(['Timestamp', 'Event'])
        writer.writerow([timestamp.strftime('%Y-%m-%d %H:%M:%S'), event_type])
    
    status_bar.showMessage(f'已儲存{event_type}標記點: {timestamp.strftime("%Y-%m-%d %H:%M:%S")}')
#-----------------------------------------------------------------------    
def GetParaTable():
    global preload_edit
    global th1_edit
    global th2_edit
    global offset_edit
    global bed_threshold
    global noise_onbed
    global noise_offbed
    global dist_thr
    global air_mattress

    preload_edit = []
    th1_edit = []
    th2_edit = []
    offset_edit = []
    
    for ch in range(6):
        preload_edit.append(int(para_table.item(0, ch).text()))
        th1_edit.append(int(para_table.item(1, ch).text()))
        th2_edit.append(int(para_table.item(2, ch).text()))
        offset_edit.append(int(para_table.item(3, ch).text()))

    bed_threshold = int(para_table.item(0, 6).text())
    noise_onbed   = int(para_table.item(0, 7).text())
    noise_offbed  = int(para_table.item(2, 7).text())
    dist_thr      = int(para_table.item(0, 8).text())
    air_mattress  = int(para_table.item(2, 8).text())

#-----------------------------------------------------------------------    
def OpenCmbFile():
    global txt_path, cmb_path, int_data, time_array, data_bcg, int_data, cmb_name
    
    # 添加數據源記錄
    log_file = open(f'{LOG_DIR}/data_source_log.txt', 'w', encoding='utf-8')
    log_file.write(f"數據來源: {data_source.currentText()}\n")
    
    if data_source.currentText() == 'Elastic':
        log_file.write("使用Elastic數據源\n")
        # 記錄Elastic數據的基本信息
        try:
            log_file.write(f"設備序號: {iCueSN.text()}\n")
            log_file.write(f"開始時間: {start_time.text()}\n")
            log_file.write(f"結束時間: {end_time.text()}\n")
        except Exception as e:
            log_file.write(f"記錄Elastic參數時出錯: {str(e)}\n")
    else:
        log_file.write("使用CMB文件數據源\n")
        log_file.write(f"CMB文件名: {cmb_name}\n")
    
    log_file.close()

    # 從 MQTT 獲取參數
    if radio_Normal.isChecked():
        reg_table = MQTT_get_reg("mqtt.humetrics.ai", "device", "!dF-9DXbpVKHDRgBryRJJBEdqCihwN", iCueSN.text())
    else:
        reg_table = MQTT_get_reg("rdtest.mqtt.humetrics.ai", "device", "BMY4dqh2pcw!rxa4hdy", iCueSN.text())

    for ch in range(6):
        para_table.item(0, ch).setText(str(reg_table[str(ch+42)]))
        para_table.item(1, ch).setText(str(reg_table[str(ch+48)]))
        para_table.item(2, ch).setText(str(reg_table[str(ch+58)]))
    
    para_table.item(0, 6).setText(str(reg_table[str(41)]))
    para_table.item(2, 7).setText(str(reg_table[str(54)]))
    para_table.item(0, 7).setText(str(reg_table[str(55)]))
    para_table.item(0, 8).setText(str(reg_table[str(56)]))
    para_table.item(2, 8).setText(str(reg_table[str(57)]))
    
    if data_source.currentText() == 'Elastic':

        # 從 Elastic 資料庫取得資料
        status_bar.showMessage('從 Elastic 資料庫取得資料中...')
        QApplication.processEvents()
        
        # 準備時間參數
        start_date_str = start_time.text()
        end_date_str = end_time.text()
        
        # 將時間格式從 yyyyMMdd_HHMMSS 轉為 yyyy-MM-dd HH:MM:SS
        start_dt = datetime.strptime(start_date_str, '%Y%m%d_%H%M%S')
        end_dt = datetime.strptime(end_date_str, '%Y%m%d_%H%M%S')
        
        start_iso = start_dt.strftime('%Y-%m-%d %H:%M:%S')
        end_iso = end_dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # 設定 cmb_name 為當前時間+序號
        current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        cmb_name = f"{iCueSN.text()}_{current_time}.cmb"
        
        try:
            # 設定 Elasticsearch 連線
            es_sensor_config = {
                "hosts": "https://es.humetrics.ai",
                "verify_certs": False,
                "api_key": "RUdMcDhJc0JYdktWbjlFeEVZZGY6b1NNSUZzMUZTQkdXN1E1NFgteFZTUQ=="
            }
            
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
                
            from elasticsearch import Elasticsearch
            es = Elasticsearch(**es_config)

            # 轉換為 ISO 格式
            start_iso_es = start_dt.strftime('%Y-%m-%dT%H:%M:%S+08:00')
            end_iso_es = end_dt.strftime('%Y-%m-%dT%H:%M:%S+08:00')

            print(f"查詢時間範圍：{start_iso_es} 到 {end_iso_es}")
            status_bar.showMessage(f'查詢時間範圍：{start_iso_es} 到 {end_iso_es}')
            QApplication.processEvents()

            # 建立查詢條件
            query_sensor_data = {
                "query": {
                    "bool": {
                        "must": [
                            {"match": {"serial_id": iCueSN.text()}},
                            {"range": {"created_at": {
                                "gte": start_iso_es,
                                "lte": end_iso_es
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
                            {"match": {"serial_id": iCueSN.text()}},
                            {"range": {"created_at": {
                                "gte": start_iso_es,
                                "lte": end_iso_es   
                            }}}
                        ]
                    }
                },
                "size": 500
            }
            
            # 使用 scroll API 獲取資料
            page = es.search(
                index="sensor_data",
                body=query_sensor_data,
                scroll='5m'
            )
            
            scroll_id = page['_scroll_id']
            hits = page['hits']['hits']

            print(f"獲取到 {len(hits)} 筆資料")
            status_bar.showMessage(f'獲取到 {len(hits)} 筆資料')
            QApplication.processEvents()
         
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
                        'ch0': source.get('ch0', source.get('Channel_1_Raw')),
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
                status_bar.showMessage(f'已獲取 {len(all_records)}/{limit} 筆資料')
                QApplication.processEvents()
                
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
            status_bar.showMessage(f'開始獲取通知資料...')
            QApplication.processEvents()
            
            # 當還有通知資料時，持續獲取
            while notify_hits and notify_count < notify_limit:
                # 處理當前批次的通知資料
                for hit in notify_hits:
                    if notify_count >= notify_limit:
                        break
                        
                    source = hit['_source']
                    timestamp = source.get('timestamp')
                    if timestamp:
                        # 確保timestamp是字符串
                        timestamp_str = str(timestamp)
                        notify_dict[timestamp_str] = source.get('statusType')
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
            status_bar.showMessage(f'成功獲取 {notify_count} 筆通知資料')
            QApplication.processEvents()
            
            # 將通知資料整合到感測器資料中
            if notify_dict and all_records:
                matched_count = 0
                for record in all_records:
                    timestamp = record.get('timestamp')
                    # 轉換timestamp格式，確保匹配
                    timestamp_str = str(timestamp)
                    if timestamp_str in notify_dict:
                        record['notify_status'] = notify_dict[timestamp_str]
                        matched_count += 1
                
                print(f"成功將 {matched_count} 筆通知資料整合到感測器資料中")
                status_bar.showMessage(f'成功將 {matched_count} 筆通知資料整合到感測器資料中')
                QApplication.processEvents()
            
            if all_records:
                # 解析資料並準備int_data
                print(f"開始處理 {len(all_records)} 筆資料")
                status_bar.showMessage(f'開始處理 {len(all_records)} 筆資料')
                QApplication.processEvents()
                
                # 將資料轉換成需要的格式
                # 1. 創建時間陣列
                time_array = []
                for record in all_records:
                    timestamp_str = record['timestamp']
                    if timestamp_str:
                        try:
                            dt = datetime.strptime(timestamp_str, "%Y%m%d%H%M%S")
                            tz = pytz.timezone('Asia/Taipei')
                            tw_dt = tz.localize(dt)
                            time_array.append(tw_dt)
                        except:
                            # 如果時間解析失敗，嘗試其他方式
                            try:
                                created_at = record['created_at']
                                dt = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                                tz = pytz.timezone('Asia/Taipei')
                                tw_dt = dt.astimezone(tz)
                                time_array.append(tw_dt)
                            except:
                                # 如果還是失敗，使用現在時間
                                print(f"無法解析時間: {timestamp_str}, {record.get('created_at')}")
                                time_array.append(datetime.now(pytz.timezone('Asia/Taipei')))
                
                time_array = np.array(time_array)
                
                # 設定 startday 為第一筆資料的日期，時間為 00:00:00
                if len(time_array) > 0:
                    startday = time_array[0].replace(hour=0, minute=0, second=0, microsecond=0)
                else:
                    startday = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
                
                # 2. 創建資料陣列
                data_list = []
                for record in all_records:
                    data_row = [
                        int(record.get('ch0', 0)),
                        int(record.get('ch1', 0)),
                        int(record.get('ch2', 0)),
                        int(record.get('ch3', 0)),
                        int(record.get('ch4', 0)),
                        int(record.get('ch5', 0))
                    ]
                    data_list.append(data_row)
                
                # 轉換為 numpy 數組
                int_data = np.array(data_list).flatten()
                
                status_bar.showMessage(f'成功從 Elastic 資料庫取得 {len(all_records)} 筆資料')
                QApplication.processEvents()
            else:
                # 如果沒有資料，使用空數組
                int_data = np.array([])
                time_array = []
                bcg = False
                status_bar.showMessage(f'未找到符合條件的資料')
                QApplication.processEvents()
                
        except Exception as e:
            print(f"從 Elastic 資料庫取得資料時發生錯誤: {str(e)}")
            status_bar.showMessage(f'從 Elastic 資料庫取得資料時發生錯誤: {str(e)}')
            QApplication.processEvents()
            
            # 如果發生錯誤，使用空數組
            int_data = np.array([])
            time_array = []
            bcg = False
            
    else:
        status_bar.showMessage('Reading ' + cmb_name + ' ........')
        QApplication.processEvents()  

        global n10
        global d10
        global x10

        #---------------------------------------------------------
        txt_path = os.path.join(LOG_DIR, f'{cmb_name[:-4]}.txt')
        with open(txt_path, mode='r', newline='') as file:
            reader = csv.reader(file)
            time_array = []
            filelen = []
            # 逐行读取数据并将其添加到列表中
            for row in reader:
                dt = datetime.strptime(row[0], "%Y%m%d_%H%M%S.dat")
                gmt = pytz.timezone('GMT')  # 建立 GMT+0 時區的時間
                gmt_dt = gmt.localize(dt)                
                tz = pytz.timezone('Asia/Taipei') # 轉換成 GMT+8 時區的時間
                tw_dt = gmt_dt.astimezone(tz)
                time_array.append(tw_dt)
                filelen.append(int(row[1]))

            time_array = np.array(time_array)
            filelen = np.array(filelen)
            bcg = np.median(filelen) == 3000
  
    #---------------------------------------------------------
        with open(f'{LOG_DIR}/{cmb_name}', "rb") as f:            
            iCueSN.setText(cmb_name.split('/')[-1][0:15])
            status_bar.showMessage('Converting to 24bit data ........')
            QApplication.processEvents()

            data = f.read() # 讀取資料
            data = np.frombuffer(data, dtype=np.uint8)
            if bcg:
                # 將數據重塑為每55字節一組
                data = data.reshape(-1, 55)
            else:
                # 將數據重塑為每24字節一組
                data = data.reshape(-1, 24)
            # 提取每組的前18字節
            data18 = data[:, :18].reshape(-1)
            # 將前18字節的數據轉換為24位整數
            # 每3字節為一組，將其轉換為24位整數
            reshaped_data = np.int32(data18.reshape(-1, 3))
            int_data = reshaped_data[:, 2] + (reshaped_data[:, 1] << 8) + (reshaped_data[:, 0] << 16)
            int_data = np.where(int_data & 0x800000, int_data - 0x1000000, int_data)
    
    print(f"int_data 的長度: {len(int_data)}")
    print(f"time_array 的長度: {len(time_array)}")
    
    # # 添加數據一致性檢查
    # if len(int_data) % 6 != 0:
    #     error_msg = f"數據長度不是6的倍數: {len(int_data)}"
    #     print(error_msg)
    #     status_bar.showMessage(error_msg)
    #     QApplication.processEvents()
    #     return
    
    # if len(int_data) // 6 != len(time_array):
    #     error_msg = f"數據點數({len(int_data) // 6})與時間點數({len(time_array)})不匹配"
    #     print(error_msg)
    #     status_bar.showMessage(error_msg)
    #     QApplication.processEvents()
    #     return
    
    data = int_data.reshape(-1, 6)
    global data_bcg
    #data_bcg = [x, y, z]

    # 將未處理的訊號存成CSV，並加入時間欄位
    data_csv = pd.DataFrame(data)

    # 移除時間插值邏輯，使用原始時間點
    if len(time_array) > 0:
        # 使用原始時間點，不進行插值
        data_csv.insert(0, 'timestamp', np.repeat(time_array, len(data) // len(time_array)))
    
    data_csv.to_csv(f"{LOG_DIR}/{cmb_name[:-4]}_raw.csv", index=False)

    # 初始化日誌檔案
    preprocess_log_file = open(f'{LOG_DIR}/{cmb_name[:-4]}_preprocess_log.txt', 'w', encoding='utf-8')
    preprocess_log_file.write("=== 資料前處理函數日誌 ===\n")
    preprocess_log_file.write(f"檔案名稱: {cmb_name}\n")
    preprocess_log_file.write(f"資料形狀: {data.shape}\n")
    preprocess_log_file.write(f"是否為BCG資料: {bcg}\n")
    preprocess_log_file.write(f"資料時間點數量: {len(time_array)}\n")
    preprocess_log_file.write(f"資料時間範圍: {time_array[0]} 到 {time_array[-1]}\n\n")

    # --------------------------------------------------------------------
    lpf = [26, 28, 32, 39, 48, 60, 74, 90, 108, 126, 146, 167, 187, 208, 227, 246, 264, 280, 294, 306, 315, 322, 326, 328, 326, 322, 315, 306, 294, 280, 264, 246, 227, 208, 187, 167, 146, 126, 108, 90, 74, 60, 48, 39, 32, 28, 26]        
    
    preprocess_log_file.write("低通濾波器係數: " + str(lpf) + "\n\n")
    
    global dist
    global dist_air
    global n10
    global d10
    global x10
    global data_resp
    global rising_dist
    global rising_dist_air
    n10 = []
    d10 = []
    x10 = []
    data_resp = []
    # 初始化dist和dist_air為空numpy數組，將在後續處理時根據需要重新初始化為適當大小
    dist = np.array([0], dtype=np.float64)  # 使用numpy數組而不是單一數值
    dist_air = np.array([0], dtype=np.float64)  # 使用numpy數組而不是單一數值

    for ch in range(6):
        status_bar.showMessage(f'Processing CH{ch+1} ........')
        QApplication.processEvents()
        preprocess_log_file.write(f"正在處理通道 {ch+1}:\n")
        
        # --------------------------------------------------------
        hp = np.convolve(data[:,ch], [-1, -2, -3, -4, 4, 3, 2, 1], mode='same')
        preprocess_log_file.write(f"  通道 {ch+1} 高通濾波後前10個值: {hp[:10]}\n")
        
        n = np.convolve(np.abs(hp / 16), lpf, mode='full')
        n = n[10:-37] / 4096
        n = n[::10]
        n10.append(np.int32(n))
        preprocess_log_file.write(f"  通道 {ch+1} 噪聲值前10個元素: {n[:10]}\n")
        
        # --------------------------------------------------------
        data_pd = pd.Series(data[:,ch]) # 將通道的數據轉換為Pandas的Series數據結構
        med10 = data_pd.rolling(window=10, min_periods=1, center=True).mean() # 計算每個窗口的中位數，窗口大小為10
        med10 = np.array(med10)
        med10 = med10[::10]
        d10.append(np.int32(med10))
        preprocess_log_file.write(f"  通道 {ch+1} 中值前10個元素: {med10[:10]}\n")
        
        # --------------------------------------------------------
        max10 = data_pd.rolling(window=10, min_periods=1, center=True).max() # 計算每個窗口的最大值，窗口大小為10 
        max10 = np.array(max10)
        max10 = np.int32(max10[::10])
        x10.append(np.int32(max10))
        preprocess_log_file.write(f"  通道 {ch+1} 最大值前10個元素: {max10[:10]}\n")
        
        # --------------------------------------------------------
        resp = data[:,ch] - savgol_filter(data[:,ch], 105, 3)
        resp = np.repeat(resp, 10).astype(np.float64)
        resp = savgol_filter(resp, 151, 3)
        #data_bcg.append(resp)
        preprocess_log_file.write(f"  通道 {ch+1} 呼吸訊號前10個元素: {resp[:10]}\n")
        
        # 計算 dist ----------------------------------------------   
        a = [1, -1023/1024]
        b = [1/1024, 0]
        preprocess_log_file.write(f"  IIR濾波器係數 a: {a}, b: {b}\n")
        
        # 確保dist在第一次使用前已被正確初始化為numpy數組
        if ch == 0:
            # 第一個通道時初始化dist為全零數組
            dist = np.zeros_like(med10, dtype=np.float64)
            dist_air = np.zeros_like(med10, dtype=np.float64)
            
        pos_iirmean = lfilter(b, a, med10) # 1 second # IIR濾波
        preprocess_log_file.write(f"  通道 {ch+1} IIR濾波後前10個元素: {pos_iirmean[:10]}\n")
        
        med10_pd = pd.Series(med10)
        mean_30sec = med10_pd.rolling(window=30, min_periods=1, center=False).mean() # 計算每個窗口的平均值，窗口大小為30 
        mean_30sec = np.int32(mean_30sec)        
        preprocess_log_file.write(f"  通道 {ch+1} 30秒平均值前10個元素: {mean_30sec[:10]}\n")
        
        diff = (mean_30sec - pos_iirmean) / 256
        if ch == 1:
            diff = diff / 3
            preprocess_log_file.write(f"  通道 {ch+1} 差值除以3後前10個元素: {diff[:10]}\n")
        else:
            preprocess_log_file.write(f"  通道 {ch+1} 差值前10個元素: {diff[:10]}\n")
            
        dist_before = dist.copy()
        dist = dist + np.square(diff) # 累加平方差
        dist[dist > 8000000] = 8000000 # 限制最大值
        preprocess_log_file.write(f"  累加通道 {ch+1} 後的位移前10個元素: {dist[:10]}\n")
        preprocess_log_file.write(f"  通道 {ch+1} 對位移的貢獻前10個元素: {(dist - dist_before)[:10]}\n")
        
        # 計算 dist (air mattress) -------------------------------
        mean_60sec = med10_pd.rolling(window=60, min_periods=1, center=False).mean() # 計算每個窗口的平均值，窗口大小為60 
        mean_60sec = np.int32(mean_60sec)
        preprocess_log_file.write(f"  通道 {ch+1} 60秒平均值前10個元素: {mean_60sec[:10]}\n")
        
        # [60][60][60][60][60][60][60][60][60][60]
        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^        
        #                     10 minutes mean ->|         b[60]
        #                                     [60][60][60][60]                                    
        a = np.zeros([780,])
        a[0] = 1
        b = np.zeros([780,])
        for s in range(10):
            b[s*60 + 180] = -0.1
        b[60] = 1
        preprocess_log_file.write(f"  空氣床濾波器係數 a 非零元素位置: {np.where(a != 0)[0]}, 值: {a[np.where(a != 0)[0]]}\n")
        preprocess_log_file.write(f"  空氣床濾波器係數 b 非零元素位置: {np.where(b != 0)[0]}, 值: {b[np.where(b != 0)[0]]}\n")
        
        diff = lfilter(b, a, mean_60sec) # 1 second
        if ch == 1:
            diff = diff / 3
            preprocess_log_file.write(f"  通道 {ch+1} 空氣床差值除以3後前10個元素: {diff[:10]}\n")
        else:
            preprocess_log_file.write(f"  通道 {ch+1} 空氣床差值前10個元素: {diff[:10]}\n")
        
        dist_air_before = dist_air.copy()
        dist_air = dist_air + np.square(diff / 256)
        preprocess_log_file.write(f"  累加通道 {ch+1} 後的空氣床位移前10個元素: {dist_air[:10]}\n")
        preprocess_log_file.write(f"  通道 {ch+1} 對空氣床位移的貢獻前10個元素: {(dist_air - dist_air_before)[:10]}\n\n")

    # 一般床墊的位移差值計算
    preprocess_log_file.write("計算一般床墊的位移差值:\n")
    # Convert to pandas Series
    dist = pd.Series(dist)
    # Calculate the difference with a shift of 60
    shift = 60
    preprocess_log_file.write(f"  位移差值計算使用的偏移量: {shift}\n")
    
    rising_dist = calculate_rising_dist(dist, preprocess_log_file)
    preprocess_log_file.write(f"  計算後的位移差值前10個元素: {rising_dist[:10]}\n\n")

    # 空氣床墊的位移差值計算
    preprocess_log_file.write("計算空氣床墊的位移差值:\n")
    # Convert to pandas Series
    dist_air = pd.Series(dist_air)
    # Calculate the difference with a shift of 60
    shift = 60
    preprocess_log_file.write(f"  空氣床位移差值計算使用的偏移量: {shift}\n")
    
    rising_dist_air = calculate_rising_dist(dist_air, preprocess_log_file)
    preprocess_log_file.write(f"  計算後的空氣床位移差值前10個元素: {rising_dist_air[:10]}\n\n")

    global idx10
    global idx1sec
    # 由檔案開始時間/資料長度計算對應 index --------------------------------------------   
    preprocess_log_file.write("計算時間索引:\n")
    idx1sec = np.array([])
    idx100 = np.array([])
    idx10 = np.array([])
    idx100_sum = 0
    idx10_sum = 0

    # 迭代資料的每一個索引 i
    for i in range(filelen.shape[0]):
        preprocess_log_file.write(f"  處理時間索引 {i}:\n")
        
        if i < filelen.shape[0] - 1:
            t_diff = np.int32((time_array[i + 1] - time_array[i]).total_seconds())  # 計算相鄰時間之差（秒）
        else:
            t_diff = 600  # 若為最後一個數據，設定時間差為 600 秒
        preprocess_log_file.write(f"    時間差: {t_diff} 秒\n")

        if t_diff > 660:
            t_blank = t_diff - 600  # 若時間差超過 660 秒，計算空白時間
            t_diff = 600  # 設定時間差為 600 秒
            preprocess_log_file.write(f"    時間差過大，調整為 {t_diff} 秒，空白時間: {t_blank} 秒\n")
        else:
            t_blank = 0
            preprocess_log_file.write(f"    時間差正常，空白時間: {t_blank} 秒\n")

        # 將索引值加入對應的陣列中
        # 生成三種不同採樣率的索引：
        # - idx1sec: 1秒一個點
        # - idx100: 100ms一個點
        # - idx10: 10ms一個點
        idx1sec_old_len = len(idx1sec)
        idx100_old_len = len(idx100)
        idx10_old_len = len(idx10)
        
        idx1sec = np.append(idx1sec, np.floor(np.linspace(0, filelen[i]//10 - 1, t_diff)) + idx100_sum//10)
        idx100 = np.append(idx100, np.floor(np.linspace(0, filelen[i] - 1, t_diff*10)) + idx100_sum)
        idx10 = np.append(idx10, np.floor(np.linspace(0, filelen[i]*10 - 1, t_diff*100)) + idx10_sum)
        
        preprocess_log_file.write(f"    新增索引點數: idx1sec: {len(idx1sec) - idx1sec_old_len}, idx100: {len(idx100) - idx100_old_len}, idx10: {len(idx10) - idx10_old_len}\n")
        
        # 處理空白時間
        if t_blank > 0:
            blank_1sec_old_len = len(idx1sec)
            blank_100_old_len = len(idx100)
            blank_10_old_len = len(idx10)
            
            idx1sec = np.append(idx1sec, np.tile(filelen[i]//10 - 1 + idx100_sum//10, (t_blank, 1)))  # 將空白時間的索引值重複添加
            idx100 = np.append(idx100, np.tile(filelen[i] - 1 + idx100_sum, (t_blank*10, 1)))  # 將空白時間的索引值重複添加
            idx10 = np.append(idx10, np.tile(filelen[i]*10 - 1 + idx10_sum, (t_blank*100, 1)))  # 將空白時間的索引值重複添加
            
            preprocess_log_file.write(f"    空白時間索引點數: idx1sec: {len(idx1sec) - blank_1sec_old_len}, idx100: {len(idx100) - blank_100_old_len}, idx10: {len(idx10) - blank_10_old_len}\n")

        idx100_sum = idx100_sum + filelen[i]  # 更新索引值總和
        idx10_sum = idx10_sum + filelen[i]*10  # 更新索引值總和
        preprocess_log_file.write(f"    索引累計: idx100_sum: {idx100_sum}, idx10_sum: {idx10_sum}\n")
        

    # 計算不同取樣率對應的時間 --------------------------------------------------------    
    st = (time_array[0] - time_array[0].replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds()
    # 不在這裡宣告global startday，而是直接使用全域變數
    global t10ms
    global t1sec
    startday = time_array[0].replace(hour=0, minute=0, second=0, microsecond=0)
    preprocess_log_file.write(f"\n基準時間: {startday}, 起始秒數: {st}\n")

    t1sec = np.array(range(np.int32((time_array[-1] - time_array[0]).total_seconds()) + 600)) + st
    idx1sec = np.int32(idx1sec)
    idx100 = np.int32(idx100)
    idx10 = np.int32(idx10)
    t100ms = np.linspace(t1sec[0], t1sec[-1], t1sec.shape[0]*10)
    t10ms = np.linspace(t1sec[0], t1sec[-1], t1sec.shape[0]*100)
    t100ms = np.around(t100ms, decimals=1)
    t10ms = np.around(t10ms, decimals=2)
    
    preprocess_log_file.write(f"時間向量: t1sec 長度: {len(t1sec)}, 範圍: {t1sec[0]} 到 {t1sec[-1]}\n")
    preprocess_log_file.write(f"時間向量: t100ms 長度: {len(t100ms)}, 範圍: {t100ms[0]} 到 {t100ms[-1]}\n")
    preprocess_log_file.write(f"時間向量: t10ms 長度: {len(t10ms)}, 範圍: {t10ms[0]} 到 {t10ms[-1]}\n")
    
    preprocess_log_file.write(f"索引向量: idx1sec 長度: {len(idx1sec)}, 範圍: {idx1sec[0] if len(idx1sec) > 0 else 'N/A'} 到 {idx1sec[-1] if len(idx1sec) > 0 else 'N/A'}\n")
    preprocess_log_file.write(f"索引向量: idx100 長度: {len(idx100)}, 範圍: {idx100[0] if len(idx100) > 0 else 'N/A'} 到 {idx100[-1] if len(idx100) > 0 else 'N/A'}\n")
    preprocess_log_file.write(f"索引向量: idx10 長度: {len(idx10)}, 範圍: {idx10[0] if len(idx10) > 0 else 'N/A'} 到 {idx10[-1] if len(idx10) > 0 else 'N/A'}\n")
    
    preprocess_log_file.write("\n前處理完成\n")
    preprocess_log_file.close()

    # for ch in range(9):   
    #     data_bcg[ch] = data_bcg[ch][idx10]

    ##--------------------------------------
    x_range, y_range = raw_plot.viewRange()
    center = (x_range[0] + x_range[1]) / 2

    # 設定新的顯示範圍，以中心點為基準，前後各10單位
    st = center - 10
    ed = center + 10

    # 調整顯示範圍，確保不超出資料範圍
    if st < t100ms[0]:
        st = t100ms[0]
        ed = st + 20
    if ed > t100ms[-1]:
        ed = t100ms[-1]
        st = ed - 20

    # --------------------------------------------------------------------
    global legend_raw
    global legend_bit
    # Customize the legend background color
    if check_NightMode.isChecked():
        legend_bg = pg.mkBrush(color=hex_to_rgb('202020'))  # Red background for the legend
    else:
        legend_bg = pg.mkBrush(color=hex_to_rgb('e0e0e0'))  # Red background for the legend
    
    legend_raw = raw_plot.addLegend()
    legend_raw.setBrush(legend_bg)
    update_raw_plot()

    legend_bit = bit_plot.addLegend()
    legend_bit.setBrush(legend_bg)
    update_bit_plot()

    status_bar.showMessage(cmb_name)    

    # 在所有資料處理完成後，最後初始化標記線
    try:
        raw_plot.vLine.hide()
        bit_plot.vLine.hide()
    except:
        pass
        
    # 重新創建標記線
    raw_plot.vLine = pg.InfiniteLine(
        angle=90, 
        movable=True,
        pen=pg.mkPen(color='r', width=2)
    )
    bit_plot.vLine = pg.InfiniteLine(
        angle=90, 
        movable=True,
        pen=pg.mkPen(color='r', width=2)
    )
    
    # 添加到圖表中
    raw_plot.addItem(raw_plot.vLine)
    bit_plot.addItem(bit_plot.vLine)
    
    # 預設隱藏
    raw_plot.vLine.hide()
    bit_plot.vLine.hide()
    
    status_bar.showMessage(cmb_name)

    # --------------------------------------------------------------------
    # 在處理完所有數據後，加入以下代碼來保存CSV
    try:
        # 建立輸出檔案名稱
        csv_filename = f"{cmb_name[:-4]}_data.csv"
        csv_filepath = os.path.join(DATA_DIR, csv_filename)
        
        # 準備數據
        data_dict = {
            'DateTime': [startday + timedelta(seconds=t) for t in t1sec],
            'Timestamp': t1sec
        }
        
        # 添加各通道的數據
        for ch in range(6):
            data_dict[f'Channel_{ch+1}_Raw'] = d10[ch][idx1sec]
            data_dict[f'Channel_{ch+1}_Noise'] = n10[ch][idx1sec]
            data_dict[f'Channel_{ch+1}_Max'] = x10[ch][idx1sec]
        
        # 添加位移和翻身數據
        data_dict['Rising_Dist_Normal'] = rising_dist[idx1sec]
        data_dict['Rising_Dist_Air'] = rising_dist_air[idx1sec]
        data_dict['OnBed_Status'] = onbed
        
        # 轉換為DataFrame並保存
        df = pd.DataFrame(data_dict)
        df.to_csv(csv_filepath, index=False)
        
        # 儲存參數設定
        param_filename = f"{cmb_name[:-4]}_parameters.csv"
        param_filepath = os.path.join(DATA_DIR, param_filename)
        
        # 準備參數數據
        param_dict = {
            'Parameter': [],
            'Channel_1': [], 'Channel_2': [], 'Channel_3': [],
            'Channel_4': [], 'Channel_5': [], 'Channel_6': []
        }
        
        # 收集參數表中的所有數據
        param_names = ['min_preload', 'threshold_1', 'threshold_2', 'offset level']
        for row in range(4):
            param_dict['Parameter'].append(param_names[row])
            for col in range(6):  # 前6個通道
                value = para_table.item(row, col).text()
                param_dict[f'Channel_{col+1}'].append(value)
        
        # 添加其他重要參數
        additional_params = [
            ('Total', str(bed_threshold)),
            ('Noise_1', str(noise_onbed)),
            ('Noise_2', str(noise_offbed)),
            ('Set Flip', str(dist_thr)),
            ('Air_Mattress', str(air_mattress)),
            ('Device_SN', iCueSN.text()),
            ('Start_Time', str(startday))
        ]

        for param_name, param_value in additional_params:
            param_dict['Parameter'].append(param_name)
            param_dict['Channel_1'].append(param_value)
            # 只填充其他通道的空值
            for col in ['Channel_2', 'Channel_3', 'Channel_4', 'Channel_5', 'Channel_6']:
                param_dict[col].append('')
        
        # 轉換為DataFrame並保存
        df_param = pd.DataFrame(param_dict)
        df_param.to_csv(param_filepath, index=False)
        
        status_bar.showMessage(f'數據和參數已保存至 {csv_filename} 和 {param_filename}')
        QApplication.processEvents()
        
        # 儲存實驗用的降採樣資料
        output_dir = os.path.join('_data', 'experiment')
        os.makedirs(output_dir, exist_ok=True)
        
        # 取得裝置序號和時間
        serial_id = iCueSN.text()
        timestamp_str = startday.strftime('%Y%m%d%H%M%S')
        
        # 建立保存檔案路徑
        downsampled_csv_path = os.path.join(output_dir, f'{serial_id}_downsampled_{timestamp_str}.csv')
        downsampled_json_path = os.path.join(output_dir, f'{serial_id}_downsampled_{timestamp_str}.json')
        
        # 準備CSV資料
        csv_data = []
        timestamps = []
        
        # 確保所有通道的d10長度一致
        min_length = min(len(d10[ch]) for ch in range(6))
        min_length = min(min_length, len(t1sec))  # 也與t1sec比較
        
        # 取得降採樣後的資料
        for i in range(min_length):
            # 建立與localrawviewer1217_elastic.py相容的時間戳格式
            iso_timestamp = startday + timedelta(seconds=i)
            timestamp = iso_timestamp.strftime('%Y%m%d%H%M%S')
            timestamps.append(timestamp)
            
            # 建立一筆資料
            row = {
                'created_at': iso_timestamp.isoformat(),
                'timestamp': timestamp,
                'ch0': int(d10[0][i]),
                'ch1': int(d10[1][i]),
                'ch2': int(d10[2][i]),
                'ch3': int(d10[3][i]),
                'ch4': int(d10[4][i]),
                'ch5': int(d10[5][i])
            }
            csv_data.append(row)
        
        # 準備JSON資料
        json_data = {
            'serial_id': serial_id,
            'start_time': startday.isoformat(),
            'end_time': (startday + timedelta(seconds=min_length)).isoformat(),
            'data': csv_data,
            # 加入處理後的資料結構，以便直接被localrawviewer1217_elastic.py使用
            'n10': [n10_array.tolist() for n10_array in n10],
            'd10': [d10_array.tolist() for d10_array in d10],
            'x10': [x10_array.tolist() for x10_array in x10],
            'rising_dist': rising_dist.tolist() if isinstance(rising_dist, np.ndarray) else rising_dist,
            'rising_dist_air': rising_dist_air.tolist() if isinstance(rising_dist_air, np.ndarray) else rising_dist_air
        }
        
        # 儲存CSV檔案
        with open(downsampled_csv_path, 'w', newline='') as csvfile:
            fieldnames = ['created_at', 'timestamp', 'ch0', 'ch1', 'ch2', 'ch3', 'ch4', 'ch5']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in csv_data:
                writer.writerow(row)
        
        # 儲存JSON檔案
        with open(downsampled_json_path, 'w') as jsonfile:
            json.dump(json_data, jsonfile, indent=2)
        
        # 儲存參數檔案供實驗使用
        parameter_path = os.path.join(output_dir, f'{serial_id}_parameters_{timestamp_str}.csv')
        with open(parameter_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Parameter ID', 'Description', 'Value'])
            
            # 儲存總和閾值
            writer.writerow(['41', 'Total Sum', str(bed_threshold)])
            
            # 儲存預載值
            for ch in range(6):
                writer.writerow([f'{42+ch}', f'Channel {ch+1} min_preload', str(preload_edit[ch])])
            
            # 儲存閾值1
            for ch in range(6):
                writer.writerow([f'{48+ch}', f'Channel {ch+1} threshold_1', str(th1_edit[ch])])
            
            # 儲存噪聲參數
            writer.writerow(['54', 'Noise 2', str(noise_offbed)])
            writer.writerow(['55', 'Noise 1', str(noise_onbed)])
            
            # 儲存翻身閾值
            writer.writerow(['56', 'Set Flip', str(dist_thr)])
            
            # 儲存氣墊床設定
            writer.writerow(['57', 'Air mattress', str(air_mattress)])
            
            # 儲存閾值2
            for ch in range(6):
                writer.writerow([f'{58+ch}', f'Channel {ch+1} threshold_2', str(th2_edit[ch])])
        
        status_bar.showMessage(f'數據和參數已保存至 {csv_filename} 和 {param_filename}，實驗用降採樣資料已保存至 {downsampled_csv_path}')


        # 測試計算值顯示
        update_calculated_value(122)

        QApplication.processEvents()
        
    except Exception as e:
        status_bar.showMessage(f'保存檔案時發生錯誤: {str(e)}')
        QApplication.processEvents()

#-----------------------------------------------------------------------
def update_raw_plot():
    global raw_plot_ch
    GetParaTable()
    # --------------------------------------------------------------------
    data_median = []
    data_max = []
    data_noise = []
    for ch in range(6):
        data_median.append(d10[ch][idx1sec] + offset_edit[ch])
    # Save show / hide settings
    isVisible = []
    try:
        raw_plot_ch
        for ch in range(6):
            isVisible.append(raw_plot_ch[ch].isVisible())
    except:
        None
    # --------------------------------------------------------------------
    # 在清除之前先保存當前的視圖範圍
    x_range = raw_plot.viewRange()[0]
    y_range = raw_plot.viewRange()[1]
    
    # 清除並重新繪製
    raw_plot.clear()
    raw_plot_ch = []
    for ch in range(6):            
        pen = pg.mkPen(color=hex_to_rgb(hex_colors[ch]))
        x = t1sec
        y = data_median[ch]
        raw_plot_ch.append(raw_plot.plot(x, y, pen=pen, name=f'Ch{ch+1}'))

    # Get original Y range
    x_range, y_range = raw_plot.viewRange()

    flip_interval = 60

    # 繪製一般床墊的翻身數據
    pen = pg.mkPen(color=hex_to_rgb(hex_colors[6]))
    x = t1sec[::flip_interval]
    y = rising_dist[idx1sec[::flip_interval]] * -100
    raw_plot_ch.append(raw_plot.plot(x, y, pen=pen, name=f'Normal'))

    raw_plot_ch[6].hide()

    # 繪製氣墊床的翻身數據
    pen = pg.mkPen(color=hex_to_rgb(hex_colors[7]))
    x = t1sec[::flip_interval]
    y = rising_dist_air[idx1sec[::flip_interval]] * -100
    raw_plot_ch.append(raw_plot.plot(x, y, pen=pen, name=f'Air'))    

    raw_plot_ch[7].hide()

    # 繪製閾值線
    pen = pg.mkPen(color=hex_to_rgb(hex_colors[8]))
    x = np.array([t1sec[0], t1sec[-1]])
    y = np.array([dist_thr, dist_thr]) * -100
    raw_plot_ch.append(raw_plot.plot(x, y, pen=pen, name=f'Threshold'))  

    if air_mattress == 0:
        flip = (rising_dist > dist_thr) * dist_thr
    else:
        flip = (rising_dist_air > dist_thr) * dist_thr

    # 繪製翻身數據
    x = t1sec[::flip_interval]
    y = flip[idx1sec[::flip_interval]] * -100
    raw_plot_ch.append(raw_plot.plot(x, y, fillLevel=0, brush=pg.mkBrush(color=hex_to_rgb(hex_colors[8])), pen=None, name='Flip'))

    # Restore original Y range
    y_range[0] = dist_thr * -200
    raw_plot.setYRange(y_range[0], y_range[1])

    # Restore show / hide settings
    if len(isVisible) == 6:
        for ch in range(6):
            if isVisible[ch]:
                raw_plot_ch[ch].show()
            else:
                raw_plot_ch[ch].hide()

    # 恢復原來的視圖範圍
    raw_plot.setXRange(x_range[0], x_range[1], padding=0)
    raw_plot.setYRange(y_range[0], y_range[1], padding=0)

#-----------------------------------------------------------------------
def update_bit_plot():
    global onload
    global onbed
    #global bit_plot_label
    global t1sec
    global idx1sec

    global bit_plot_ch
    global bit_plot_sum
    global bit_plot_onff

    try:
        t1sec
    except NameError:
        return
    # --------------------------------------------------------------------
    GetParaTable()
    # --------------------------------------------------------------------    
    onload, onbed = EvalParameters()
    # --------------------------------------------------------------------
    for ch in range(6):
        onload[ch] = onload[ch][idx1sec]
    onbed = onbed[idx1sec]
    # --------------------------------------------------------------------
    # Save show / hide settings
    isVisible = []
    try:
        bit_plot_ch
        for ch in range(6):
            isVisible.append(bit_plot_ch[ch].isVisible())
    except:
        None
    bit_plot.clear()
    # --------------------------------------------------------------------
    onload_sum = np.sum(np.int32(onload), axis=0)
    onload_avg = np.sum(onbed * onload_sum) / np.sum(onbed)
    bit_plot_sum = bit_plot.plot(t1sec, onload_sum-7.5, fillLevel=-7.5, brush=pg.mkBrush(color=(100,100,100)), pen=None, name='SUM')
    bit_plot_onff = bit_plot.plot(t1sec, onbed - 8.5, fillLevel=-7.5, brush=pg.mkBrush(color=hex_to_rgb(hex_colors[0])), pen=None, name='OFFBED')

    bit_plot_ch = []
    for ch in range(6):
        #brush = pg.mkBrush(color=hex_to_rgb(hex_colors[ch]))
        pen = pg.mkPen(color=hex_to_rgb(hex_colors[ch]))
        bit_plot_ch.append(bit_plot.plot(t1sec, onload[ch]*0.75 - ch - 2.5, pen=pen, name=f'Ch{ch+1}'))
    # --------------------------------------------------------------------
    # Restore show / hide settings 
    if len(isVisible) == 6:
        for ch in range(6):
            if isVisible[ch]:
                bit_plot_ch[ch].show()
            else:
                bit_plot_ch[ch].hide()

    bit_plot_sum.hide()

#----------------------------------------------------
def EvalParameters(): 
    
    global n10
    global d10
    global x10

    global onload
    global onbed
    
    global dist
    global dist_air
    global rising_dist
    global rising_dist_air

    global zdata_final
    zdata_final = []

    global base_final
    base_final = []

    # 初始化日誌檔案
    log_file = open(f'{LOG_DIR}/onoff_bed_log.txt', 'w', encoding='utf-8')
    log_file.write("=== EvalParameters 函數日誌 ===\n")
    log_file.write(f"處理數據長度: {d10[0].shape[0]}\n")
    log_file.write(f"參數設定: noise_onbed={noise_onbed}, noise_offbed={noise_offbed}, bed_threshold={bed_threshold}\n")
    
    # 確保全域變數 dist 和 dist_air 已被正確初始化
    if not isinstance(dist, pd.Series) and (not isinstance(dist, np.ndarray) or len(dist) <= 1):
        log_file.write("注意：正在初始化 dist 變數\n")
        dist = np.zeros(d10[0].shape[0], dtype=np.float64)
    
    if not isinstance(dist_air, pd.Series) and (not isinstance(dist_air, np.ndarray) or len(dist_air) <= 1):
        log_file.write("注意：正在初始化 dist_air 變數\n")
        dist_air = np.zeros(d10[0].shape[0], dtype=np.float64)

    l = d10[0].shape[0]
    onbed = np.zeros((l,))  # 初始化在床狀態陣列
    onload = []             # 初始化負載狀態列表
    total = 0              # 初始化總負載

    for ch in range(6):
        log_file.write(f"\n處理通道 {ch}:\n")
        max10 = x10[ch] + offset_edit[ch]    # 最大值加上偏移
        med10 = d10[ch] + offset_edit[ch]     # 中值加上偏移
        n = n10[ch]                           # 噪聲值
        preload = preload_edit[ch]            # 預載值
        
        log_file.write(f"  通道 {ch} 偏移值: {offset_edit[ch]}\n")
        log_file.write(f"  通道 {ch} 預載值: {preload}\n")
        log_file.write(f"  通道 {ch} 閾值1: {th1_edit[ch]}\n")
        log_file.write(f"  通道 {ch} 閾值2: {th2_edit[ch]}\n")
        log_file.write(f"  通道 {ch} 原始數據前10個值: {med10[:10]}\n")
        log_file.write(f"  通道 {ch} 噪聲值前10個值: {n[:10]}\n")
        
        # 判斷是否為零點（無負載狀態）
        zeroing = np.less(n * np.right_shift(max10, 5), noise_offbed * np.right_shift(preload, 5))
        log_file.write(f"  通道 {ch} 零點判定前10個值: {zeroing[:10]}\n")
        
        th1 = th1_edit[ch]
        th2 = th2_edit[ch]
        approach = max10 - (th1 + th2)           # 計算接近度
        speed = n // (noise_onbed * 4)           # 計算速度
        np.clip(speed, 1, 16, out=speed)         # 限制速度範圍
        app_sp = approach * speed                # 接近度與速度的乘積
        sp_1024 = 1024 - speed                   # 速度的補數
        
        log_file.write(f"  通道 {ch} 接近度前10個值: {approach[:10]}\n")
        log_file.write(f"  通道 {ch} 速度前10個值: {speed[:10]}\n")
        
        #----------------------------------------------------
        base = (app_sp[0] // 1024 + med10[0]) // 2
        base = np.int64(base)
        baseline = np.zeros_like(med10)           
        for i in range(l):
            if zeroing[i]:
                base = np.int64(med10[i])        # 如果是零點，直接使用當前值
            base = (base * sp_1024[i] + app_sp[i]) // 1024  # 動態更新基線
            baseline[i] = base
        
        log_file.write(f"  通道 {ch} 基線前10個值: {baseline[:10]}\n")
        
        channel_total = med10[:] - baseline      # 計算通道負載
        log_file.write(f"  通道 {ch} 負載前10個值: {channel_total[:10]}\n")
        
        total = total + channel_total      # 累加所有通道的淨負載
        o = np.less(th1, channel_total)    # 判斷是否超過閾值
        log_file.write(f"  通道 {ch} 負載狀態前10個值: {o[:10]}\n")
        
        onload.append(o)                         # 記錄該通道的負載狀態
        onbed = onbed + o                        # 累加到總體在床狀態

        d_zero = med10 - baseline
        zdata_final.append(d_zero)
        base_final.append(baseline)

    log_file.write(f"\n總負載前10個值: {total[:10]}\n")
    bed_threshold_check = np.less(bed_threshold, total)
    log_file.write(f"床閾值檢查前10個值: {bed_threshold_check[:10]}\n")
    
    onbed = onbed + bed_threshold_check 
    onbed = np.int32(onbed > 0)
    
    log_file.write(f"最終在床狀態前10個值: {onbed[:10]}\n")
    log_file.write(f"在床狀態統計: 在床={np.sum(onbed)}, 離床={len(onbed) - np.sum(onbed)}\n")
    
    # 關閉日誌檔案
    log_file.close()
    
    # 儲存結果檔案用於比較
    try:
        # 建立輸出路徑
        output_dir = os.path.join('_data', 'experiment')
        os.makedirs(output_dir, exist_ok=True)
        
        # 取得裝置序號和時間
        serial_id = iCueSN.text()
        timestamp_str = startday.strftime('%Y%m%d%H%M%S')
        
        # # 獲取dist和dist_air的全域變數
        # global dist
        # global dist_air
        
        # 確保dist和dist_air變數存在
        if not isinstance(dist, (pd.Series, np.ndarray)) or len(dist) == 0:
            log_file.write("警告：dist變數未正確定義，將使用默認值\n")
            dist = np.zeros(d10[0].shape[0], dtype=np.float64)
        
        if not isinstance(dist_air, (pd.Series, np.ndarray)) or len(dist_air) == 0:
            log_file.write("警告：dist_air變數未正確定義，將使用默認值\n")
            dist_air = np.zeros(d10[0].shape[0], dtype=np.float64)
        
        # 計算位移差值 - 同樣的算法，確保結果一致
        dist_series = pd.Series(dist)
        shift = 60
        rising_dist = dist_series.shift(-shift) - dist_series
        rising_dist = rising_dist.fillna(0)
        rising_dist = np.int32(rising_dist)
        rising_dist[rising_dist < 0] = 0
        rising_dist = rising_dist // 127
        rising_dist[rising_dist > 1000] = 1000
        
        # 計算氣墊床位移差值
        dist_air_series = pd.Series(dist_air)
        rising_dist_air = dist_air_series.shift(-shift) - dist_air_series
        rising_dist_air = rising_dist_air.fillna(0)
        rising_dist_air = np.int32(rising_dist_air)
        rising_dist_air[rising_dist_air < 0] = 0
        rising_dist_air = rising_dist_air // 127
        rising_dist_air[rising_dist_air > 1000] = 1000
        
        # 建立保存檔案路徑
        results_path = os.path.join(output_dir, f'{serial_id}_results_{timestamp_str}.csv')
        
        # 準備時間戳
        timestamps = [(startday + timedelta(seconds=i)).strftime('%Y%m%d%H%M%S') for i in range(len(onbed))]
        
        # 確保所有陣列長度一致
        data_length = len(onbed)
        
        # 如果rising_dist和rising_dist_air長度超過onbed長度，截斷它們
        rising_dist_trimmed = rising_dist[:data_length] if len(rising_dist) > data_length else rising_dist
        rising_dist_air_trimmed = rising_dist_air[:data_length] if len(rising_dist_air) > data_length else rising_dist_air
        
        # 如果它們長度不足，則用最後一個元素填充到相同長度
        if len(rising_dist_trimmed) < data_length:
            rising_dist_trimmed = np.pad(rising_dist_trimmed, 
                                      (0, data_length - len(rising_dist_trimmed)), 
                                      'edge')
        
        if len(rising_dist_air_trimmed) < data_length:
            rising_dist_air_trimmed = np.pad(rising_dist_air_trimmed, 
                                          (0, data_length - len(rising_dist_air_trimmed)), 
                                          'edge')
        
        # 建立結果 DataFrame
        results_df = pd.DataFrame({
            'timestamp': timestamps,
            'bed_status': onbed,
            'rising_dist': rising_dist_trimmed,  # 使用處理後的陣列
            'rising_dist_air': rising_dist_air_trimmed  # 使用處理後的陣列
        })
        
        # 儲存結果
        results_df.to_csv(results_path, index=False)
        status_bar.showMessage(f'已儲存分析結果檔案至: {results_path}')
        QApplication.processEvents()
        
    except Exception as e:
        status_bar.showMessage(f'儲存結果檔案時發生錯誤: {str(e)}')
        QApplication.processEvents()
    
    return onload, onbed

#-----------------------------------------------------------------------   
from PyQt5.QtWidgets import QPushButton, QLineEdit, QComboBox, QVBoxLayout, QCalendarWidget, QFileDialog, QDialog, QStatusBar
import pytz
from ftplib import FTP

status_bar = QStatusBar()
status_bar.showMessage(None)    

#-----------------------------------------------------------------------    
def OpenCMBDialog():
    global cmb_name

    #os.chdir('C:\data\OfflineDebug')
    #os.chdir('D:\iCuePreview')
    #os.chdir('E:\cmbdata')
    options = QFileDialog.Options()
    options |= QFileDialog.ReadOnly
    cmb_name, _ = QFileDialog.getOpenFileName(mw, 'Open File', '', 'CMB Files (*.cmb);;All Files (*)', options=options)
    if cmb_name:
        OpenCmbFile()

#----------------------------------------------------------------------- 
def load_json_data(json_path):
    """讀取 JSON 檔案並轉換成與 CSV 相同的格式"""
    with open(json_path, 'r') as f:
        json_content = json.load(f)
    
    # 取得資料陣列
    data = json_content['data']
    
    # 初始化各通道的資料列表
    channels_data = [[] for _ in range(6)]
    timestamps = []
    
    # 轉換資料
    for item in data:
        # 轉換時間戳記
        timestamp = datetime.strptime(item['created_at'], '%Y-%m-%d %H:%M:%S')
        timestamps.append(timestamp)
        
        # 收集各通道資料
        channels_data[0].append(item['ch0'])  # Channel 1
        channels_data[1].append(item['ch1'])  # Channel 2
        channels_data[2].append(item['ch2'])  # Channel 3
        channels_data[3].append(item['ch3'])  # Channel 4
        channels_data[4].append(item['ch4'])  # Channel 5
        channels_data[5].append(item['ch5'])  # Channel 6
    
    return timestamps, channels_data

def OpenJsonFile():
    global d10, n10, x10, t1sec, startday, idx1sec, dist, dist_air, rising_dist, rising_dist_air, onbed
    
    json_path, _ = QtWidgets.QFileDialog.getOpenFileName(
        None, 
        "選擇 JSON 檔案", 
        DATA_DIR,
        "JSON files (*.json)"
    )
    
    if not json_path:
        return
    
    try:
        # 初始化日誌檔案
        json_log_file = open(f'{LOG_DIR}/{os.path.basename(json_path)[:-5]}_preprocess_log.txt', 'w', encoding='utf-8')
        json_log_file.write("=== JSON資料前處理函數日誌 ===\n")
        json_log_file.write(f"檔案名稱: {os.path.basename(json_path)}\n")
        
        # 讀取 JSON 資料
        json_log_file.write("正在讀取JSON資料...\n")
        timestamps, channels_data = load_json_data(json_path)
        json_log_file.write(f"讀取完成，資料點數: {len(timestamps)}\n")
        json_log_file.write(f"時間範圍: {timestamps[0]} 到 {timestamps[-1]}\n\n")
        
        # 初始化資料陣列
        d10 = []  # 中值
        n10 = []  # 雜訊
        x10 = []  # 最大值
        
        # 初始化位移計算相關變數
        dist = np.zeros(len(timestamps))
        dist_air = np.zeros(len(timestamps))
        
        json_log_file.write("開始處理通道資料...\n")
        
        # 處理每個通道的資料
        for ch, ch_data in enumerate(channels_data):
            json_log_file.write(f"\n處理通道 {ch+1}:\n")
            ch_data = np.array(ch_data)
            med10 = ch_data  # 原始值作為中值
            json_log_file.write(f"  通道 {ch+1} 原始數據前10個值: {med10[:10]}\n")
            
            # 計算雜訊值（可以根據需求調整計算方法）
            noise = np.abs(np.diff(med10, prepend=med10[0])) 
            noise = np.maximum.accumulate(noise)
            json_log_file.write(f"  通道 {ch+1} 噪聲值前10個值: {noise[:10]}\n")
            
            d10.append(med10)
            n10.append(noise)
            x10.append(med10)  # 最大值暫時使用原始值
            
            # 計算位移值（與原始程式相同的計算邏輯）
            json_log_file.write("  計算位移值:\n")
            a = [1, -1023/1024]
            b = [1/1024, 0]
            json_log_file.write(f"    IIR濾波器係數 a: {a}, b: {b}\n")
            
            pos_iirmean = lfilter(b, a, med10)
            json_log_file.write(f"    IIR濾波後前10個值: {pos_iirmean[:10]}\n")
            
            med10_pd = pd.Series(med10)
            mean_30sec = med10_pd.rolling(window=30, min_periods=1, center=False).mean()
            mean_30sec = np.int32(mean_30sec)
            json_log_file.write(f"    30秒平均值前10個值: {mean_30sec[:10]}\n")
            
            diff = (mean_30sec - pos_iirmean) / 256
            if ch == 1:  # 現在 ch 已經定義了
                diff = diff / 3
                json_log_file.write(f"    通道 {ch+1} 差值除以3後前10個值: {diff[:10]}\n")
            else:
                json_log_file.write(f"    通道 {ch+1} 差值前10個值: {diff[:10]}\n")
            
            dist_before = dist.copy()
            dist = dist + np.square(diff)
            dist[dist > 8000000] = 8000000
            json_log_file.write(f"    累加通道 {ch+1} 後的位移前10個值: {dist[:10]}\n")
            json_log_file.write(f"    通道 {ch+1} 對位移的貢獻前10個值: {(dist - dist_before)[:10]}\n")
            
            # 計算空氣床墊的位移值
            json_log_file.write("  計算空氣床墊位移值:\n")
            mean_60sec = med10_pd.rolling(window=60, min_periods=1, center=False).mean()
            mean_60sec = np.int32(mean_60sec)
            json_log_file.write(f"    60秒平均值前10個值: {mean_60sec[:10]}\n")
            
            # 濾波器係數
            a = np.zeros([780,])
            a[0] = 1
            b = np.zeros([780,])
            for s in range(10):
                b[s*60 + 180] = -0.1
            b[60] = 1
            json_log_file.write(f"    空氣床濾波器係數 a 非零元素位置: {np.where(a != 0)[0]}, 值: {a[np.where(a != 0)[0]]}\n")
            json_log_file.write(f"    空氣床濾波器係數 b 非零元素位置: {np.where(b != 0)[0]}, 值: {b[np.where(b != 0)[0]]}\n")
            
            diff = lfilter(b, a, mean_60sec)
            if ch == 1:
                diff = diff / 3
                json_log_file.write(f"    通道 {ch+1} 空氣床差值除以3後前10個值: {diff[:10]}\n")
            else:
                json_log_file.write(f"    通道 {ch+1} 空氣床差值前10個值: {diff[:10]}\n")
            
            dist_air_before = dist_air.copy()
            dist_air = dist_air + np.square(diff / 256)
            json_log_file.write(f"    累加通道 {ch+1} 後的空氣床位移前10個值: {dist_air[:10]}\n")
            json_log_file.write(f"    通道 {ch+1} 對空氣床位移的貢獻前10個值: {(dist_air - dist_air_before)[:10]}\n")
        
        # 計算翻身指標
        json_log_file.write("\n計算翻身指標:\n")
        json_log_file.write("  一般床墊:\n")
        rising_dist = calculate_rising_dist(dist, json_log_file)
        json_log_file.write(f"    翻身指標前10個值: {rising_dist[:10]}\n")
        
        json_log_file.write("  空氣床墊:\n")
        rising_dist_air = calculate_rising_dist(dist_air, json_log_file)
        json_log_file.write(f"    翻身指標前10個值: {rising_dist_air[:10]}\n")
        
        # 設定時間相關變數
        startday = timestamps[0].replace(hour=0, minute=0, second=0, microsecond=0)
        json_log_file.write(f"\n基準時間: {startday}\n")
        
        t1sec = np.array([(t - startday).total_seconds() for t in timestamps])
        json_log_file.write(f"時間向量t1sec前10個值: {t1sec[:10]}\n")
        
        idx1sec = np.arange(len(t1sec))
        json_log_file.write(f"索引向量idx1sec前10個值: {idx1sec[:10]}\n")
        
        json_log_file.write("\nJSON資料前處理完成\n")
        json_log_file.close()
        
        # 計算在床狀態
        EvalParameters()
        
        # 更新圖表
        update_raw_plot()
        update_bit_plot()
        
        status_bar.showMessage(f"已載入 JSON 檔案: {os.path.basename(json_path)}")
        
    except Exception as e:
        status_bar.showMessage(f"載入 JSON 檔案時發生錯誤: {str(e)}")
        QApplication.processEvents()

def calculate_rising_dist(dist, log_file=None):
    """計算翻身指標的輔助函數
    
    Args:
        dist: 位移資料
        log_file: 日誌文件對象，如果提供則記錄計算過程
    
    Returns:
        計算後的翻身指標
    """
    if log_file:
        log_file.write("  計算翻身指標細節:\n")
        log_file.write(f"    位移資料前10個值: {dist[:10]}\n")
    
    dist_series = pd.Series(dist)
    shift = 60
    
    if log_file:
        log_file.write(f"    使用的偏移量: {shift}\n")
    
    # 為了處理邊界情況，我們需要確保移位操作不會超出範圍
    # 先計算移位差值，然後用0填充NaN（這會在序列末尾產生）
    rising_dist = dist_series.shift(-shift) - dist_series
    
    if log_file:
        log_file.write(f"    偏移差值前10個值: {rising_dist[:10]}\n")
    
    # 用0填充NaN值，這會發生在序列的末尾
    rising_dist = rising_dist.fillna(0)
    
    # 轉換為整數陣列（這樣我們可以在numpy陣列上操作）
    rising_dist = np.int32(rising_dist)
    
    if log_file:
        log_file.write(f"    填充NaN後前10個值: {rising_dist[:10]}\n")
    
    # 負值處理
    rising_dist[rising_dist < 0] = 0
    
    if log_file:
        log_file.write(f"    負值處理後前10個值: {rising_dist[:10]}\n")
    
    # 縮放數值
    rising_dist = rising_dist // 127
    
    if log_file:
        log_file.write(f"    除以127後前10個值: {rising_dist[:10]}\n")
    
    # 限制最大值
    rising_dist[rising_dist > 1000] = 1000
    
    if log_file:
        log_file.write(f"    限制最大值後前10個值: {rising_dist[:10]}\n")
        log_file.write(f"    最終翻身指標統計：最小={np.min(rising_dist)}, 最大={np.max(rising_dist)}, 平均={np.mean(rising_dist):.2f}, 長度={len(rising_dist)}\n")
    
    return rising_dist

#----------------------------------------------------------------------- 

# 需要在業務邏輯處理的部分添加更新數值顯示的函數
def update_calculated_value(value):
    """
    更新界面上顯示的計算值
    
    Args:
        value (float): 要顯示的計算值
    """
    value_display.setText(f"{value:.2f}")  # 顯示兩位小數
    QApplication.processEvents()  # 立即更新界面
    return f"{value:.2f}"  # 同時返回格式化的值，以便在其他地方使用

#-----------------------------------------------------------------------   
class CalendarDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(calendar_title)  # 設定對話框的標題為父視窗的 calendar_title 屬性
        layout = QVBoxLayout(self)
        self.calendar = QCalendarWidget()  # 建立日曆元件
        layout.addWidget(self.calendar)  # 將日曆元件加入對話框布局
        self.calendar.clicked.connect(self.select_date)  # 設定日曆元件的點擊事件為 select_date 方法
        self.close()  # 關閉對話框

    def select_date(self):
        self.set_selected_date(self.calendar.selectedDate().toPyDate())  # 設定父視窗的選取日期為日曆元件所選擇的日期
        self.close()  # 關閉對話框
        
    def set_selected_date(self, selected_date):
        if calendar_title == 'Select Start Date':
            selected_date_string = selected_date.strftime("%Y%m%d_040000")  # 將選取的日期轉換成指定格式
            start_time.setText(selected_date_string)  # 將選取的日期設定為起始時間
            selected_date = selected_date + timedelta(days=1)  # 將選取的日期加一天
            selected_date_string = selected_date.strftime("%Y%m%d_040000") 
            end_time.setText(selected_date_string)  # 將加一天後的
        if calendar_title == 'Select End Date':
            selected_date_string = selected_date.strftime("%Y%m%d_040000")  # 將選取的日期轉換成指定格式
            end_time.setText(selected_date_string)  # 將選取的日期設定為結束時間

def start_calendar(event):
    global calendar_title
    if event.button() == Qt.LeftButton:  # 確認滑鼠按鈕為左鍵
        calendar_title = 'Select Start Date'  # 設定日曆對話框的標題為「選擇開始日期」
        dialog = CalendarDialog()  # 建立日曆對話框
        dialog.exec_()  # 顯示日曆對話框

def end_calendar(event):
    global calendar_title
    if event.button() == Qt.LeftButton:  # 確認滑鼠按鈕為左鍵
        calendar_title = 'Select End Date'  # 設定日曆對話框的標題為「選擇結束日期」
        dialog = CalendarDialog()  # 建立日曆對話框
        dialog.exec_()  # 顯示日曆對話框

#--------------------------------------------------------------------------
def download_files_by_time_range(FILE_PATH, ST_TIME, ED_TIME, FTP_ADDR, USER, PASW, FILESIZE):
    # 建立FTP連線
    ftp = FTP(FTP_ADDR)
    ftp.login(USER, PASW)
    # 列出目錄下所有檔案
    file_list = []

    try:
        status_bar.showMessage('Get FTP directory ...')
        QApplication.processEvents()
        ftp.dir(FILE_PATH, file_list.append)
    except:
        status_bar.showMessage('FTP error !!!')
        QApplication.processEvents()

    # 過濾出符合時間範圍的檔案
    filtered_list = []
    for file_info in file_list:
        file_name = file_info.split()[-1]
        file_size = int(file_info.split()[4])
        if file_name.endswith('.dat') and file_size == FILESIZE:
            file_time_str = file_name.split('.dat')[0]
            if ST_TIME <= file_time_str <= ED_TIME:
                filtered_list.append(file_name)

    # 下載檔案並記錄檔案名稱
    n = 0
    with open('filelist.txt', 'w') as f:
        for file_name in filtered_list:
            local_path = os.path.join(os.getcwd(), file_name)
            with open(local_path, 'wb') as lf:
                ftp.retrbinary(f'RETR {FILE_PATH}/{file_name}', lf.write)
            f.write(file_name + '\n')
            n = n + 1        
            status_bar.showMessage(f"Copying from {data_source.currentText()} ...   ( {file_name} )")
            QApplication.processEvents()
    # 關閉FTP連線
    ftp.quit()
    return n

#--------------------------------------------------------------------------
def loadClicked(event):
    global cmb_name  # 將global宣告移到函數開頭
    
    ST_TIME = start_time.text()
    ED_TIME = end_time.text()
    
    # 檢查預計產生的.cmb檔案是否已存在
    cmb_name = iCueSN.text() + '_' + ST_TIME[:-4] + '_' + ED_TIME[:-4] + '.cmb'
    cmb_path = os.path.join(LOG_DIR, cmb_name)
    
    if os.path.exists(cmb_path):
        status_bar.showMessage(f'檔案 {cmb_name} 已存在，直接開啟')
        QApplication.processEvents()
        OpenCmbFile()
        return
    
    # 如果選擇了先獲取參數
    if check_get_para.isChecked():
        status_bar.showMessage('正在獲取MQTT參數...')
        QApplication.processEvents()
        
        try:
            if radio_Normal.isChecked():
                reg_table = MQTT_get_reg("mqtt.humetrics.ai", "device", 
                    "!dF-9DXbpVKHDRgBryRJJBEdqCihwN", iCueSN.text())
            else:
                reg_table = MQTT_get_reg("rdtest.mqtt.humetrics.ai", "device", 
                    "BMY4dqh2pcw!rxa4hdy", iCueSN.text())
            
            # 儲存參數到檔案
            param_filename = f"{cmb_name[:-4]}_init_parameters.csv"
            param_filepath = os.path.join(DATA_DIR, param_filename)
            
            with open(param_filepath, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Register', 'Value'])
                for key, value in reg_table.items():
                    writer.writerow([key, value])
                    
            status_bar.showMessage('MQTT參數已儲存')
            QApplication.processEvents()
            
        except Exception as e:
            status_bar.showMessage(f'獲取MQTT參數失敗: {str(e)}')
            QApplication.processEvents()
            return
    
    # 繼續原有的下載邏輯
    n = 0
    bcg = 0
    if data_source.currentText() == '\\RAW':
        FTP_ADDR = 'raw.humetrics.ai'
        USER = 'Joey'
        PASW = 'JoeyJoey'
        FILE_PATH = '/ESP32_DEVICES/' + iCueSN.text() + '/RAW/'
        n = download_files_by_time_range(FILE_PATH, ST_TIME, ED_TIME, FTP_ADDR, USER, PASW, 144000)
    if data_source.currentText() == 'FTP:\\RAW':
        FTP_ADDR = 'raw.humetrics.ai'
        USER = 'Robot'
        PASW = 'HM66050660'
        FILE_PATH = '/ESP32_DEVICES/' + iCueSN.text() + '/RAW/'
        n = download_files_by_time_range(FILE_PATH, ST_TIME, ED_TIME, FTP_ADDR, USER, PASW, 144000)
    if data_source.currentText() == 'FTP:\\RAW_COLLECT':
        FTP_ADDR = 'raw.humetrics.ai'
        USER = 'ESP32'
        PASW = 'HM66050660'
        FILE_PATH = '/SmartBed_ESP32VS/' + iCueSN.text() + '/RAW_COLLECT/'
        n = download_files_by_time_range(FILE_PATH, ST_TIME, ED_TIME, FTP_ADDR, USER, PASW, 144000)
    if data_source.currentText() == 'FTP:\\BCGRAW':
        FTP_ADDR = 'raw.humetrics.ai'
        USER = 'Robot'
        PASW = 'HM66050660'
        FILE_PATH = '/ESP32_DEVICES/' + iCueSN.text() + '/BCGRAW/'
        n = download_files_by_time_range(FILE_PATH, ST_TIME, ED_TIME, FTP_ADDR, USER, PASW, 165000)
        bcg = 1
    if data_source.currentText() == 'RD_FTP:\\BCGRAW':
        FTP_ADDR = 'raw.humetrics.ai'
        USER = 'ESP32'
        PASW = 'HM66050660'
        FILE_PATH = '/SmartBed_ESP32VS/' + iCueSN.text() + '/BCGRAW/'
        n = download_files_by_time_range(FILE_PATH, ST_TIME, ED_TIME, FTP_ADDR, USER, PASW, 165000)
        bcg = 1
    if n == 0:
        status_bar.showMessage('Error: No file found !')
        QApplication.processEvents()                 
        return      
    #--------------------------------------
    status_bar.showMessage('Processing ...')
    QApplication.processEvents()

    with open('filelist.txt', 'r') as f:        
        file_list = f.read().splitlines()

    # 打開輸出檔案以二進制追加模式
    filelen = []  
    cmb_path = os.path.join(LOG_DIR, cmb_name)
    with open(cmb_path, 'ab') as comb_file:
        for file_name in file_list:
            if bcg == 1:
                filelen.append(os.path.getsize(file_name)//55)
            else:
                filelen.append(os.path.getsize(file_name)//24)
            with open(file_name, 'rb') as infile:
                # 將輸入檔案的內容複製到輸出檔案
                shutil.copyfileobj(infile, comb_file) 
                status_bar.showMessage('combining ' + file_name)   

    # 修改 .txt 檔案的存放路徑
    txt_path = os.path.join(LOG_DIR, f'{cmb_name[:-4]}.txt')
    with open(txt_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        data_to_write = [[file_list[i], filelen[i]] for i in range(len(file_list))]            
        for row in data_to_write:
            writer.writerow(row)   

    # 刪除 .dat 檔案
    try:
        for dat_file in file_list:
            if os.path.exists(dat_file):
                os.remove(dat_file)
        status_bar.showMessage('Temporary .dat files cleaned up')
    except Exception as e:
        status_bar.showMessage(f'Error cleaning up .dat files: {str(e)}')
    
    QApplication.processEvents()

    if cmb_name:
        OpenCmbFile()

#--------------------------------------------------------------------------
def MQTT_set_reg(mqtt_server, username, password, sn, payload):
    # Create a MQTT client
    client = mqtt.Client()
    client.username_pw_set(username, password)
    if radio_Normal.isChecked():
        client.tls_set('/Users/hugolin/Documents/PY/Ethan/humetric_mqtt_certificate.pem', None, None, cert_reqs=ssl.CERT_NONE)
        client.connect(mqtt_server, 8883, 60)
    else:
        client.connect(mqtt_server, 1883, 60)
    
    # Create the message payload / Publish the message
    topic_set_regs = "algoParam/" + sn + "/set"
    payload.update({'taskID':4881})
    client.publish(topic_set_regs, json.dumps(payload))

    client.disconnect() # Stop the MQTT client loop when done
    
#--------------------------------------------------------------------------
# duplicate from above
#  def MQTT_set_reg(mqtt_server, username, password, sn, payload):
#     # Create a MQTT client
#     client = mqtt.Client()
#     client.username_pw_set(username, password)
#     # Connect to the broker
#     if radio_Normal.isChecked():
#         client.tls_set('/Users/hugolin/Documents/PY/Ethan/humetric_mqtt_certificate.pem', None, None, cert_reqs=ssl.CERT_NONE)
#         client.connect(mqtt_server, 8883, 60)
#     else:
#         client.connect(mqtt_server, 1883, 60)

#     # Create the message payload / Publish the message
#     topic_set_regs = "algoParam/" + sn + "/set"
#     payload.update({'taskID':4881})
#     client.publish(topic_set_regs, json.dumps(payload))

#     client.disconnect() # Stop the MQTT client loop when done

#--------------------------------------------------------------------------
def MqttSetDialog():
    reg_table = {}

    for ch in range(6):
        reg_table[str(ch+42)] = int(para_table.item(0, ch).text())
        reg_table[str(ch+48)] = int(para_table.item(1, ch).text())
        reg_table[str(ch+58)] = int(para_table.item(2, ch).text())
    
    reg_table['41'] = int(para_table.item(0, 6).text())
    reg_table['54'] = int(para_table.item(2, 7).text())
    reg_table['55'] = int(para_table.item(0, 7).text())
    reg_table['56'] = int(para_table.item(0, 8).text())
    reg_table['57'] = int(para_table.item(2, 8).text())

    if radio_Normal.isChecked():
        MQTT_set_reg("mqtt.humetrics.ai", "device", "!dF-9DXbpVKHDRgBryRJJBEdqCihwN", iCueSN.text(), reg_table)
    else:
        MQTT_set_reg("rdtest.mqtt.humetrics.ai", "device", "BMY4dqh2pcw!rxa4hdy", iCueSN.text(), reg_table)

#--------------------------------------------------------------------------
def display_96DPI(state):
    # 解決顯示器比例(例125%)造成 pyqtgraph 軸顯示不正確
    app.setAttribute(Qt.AA_Use96Dpi)    

#--------------------------------------------------------------------------
def change_NightMode(state):
    global legend_raw
    global legend_bit    
    if check_NightMode.isChecked():
        raw_plot.setBackground([0,0,0])
        wav_plot.setBackground([0,0,0])
        bit_plot.setBackground([0,0,0])
        legend_bg = pg.mkBrush(color=hex_to_rgb('202020'))  # Red background for the legend
    else:
        raw_plot.setBackground([255,255,255])
        wav_plot.setBackground([255,255,255])
        bit_plot.setBackground([255,255,255])
        legend_bg = pg.mkBrush(color=hex_to_rgb('e0e0e0'))  # Red background for the legend
    
    # 確保legend_raw和legend_bit已被定義後再使用
    if 'legend_raw' in globals() and legend_raw is not None:
        legend_raw.setBrush(legend_bg)
    if 'legend_bit' in globals() and legend_bit is not None:
        legend_bit.setBrush(legend_bg)

#--------------------------------------------------------------------------
def horizontal_header_clicked(section):
    global raw_plot_ch
    global bit_plot_ch
    global bit_plot_sum
    global bit_plot_onff

    if section < 6:
        bit_plot_sum.hide()
        bit_plot_onff.show()
        for ch in range(6):
            raw_plot_ch[ch].hide()
            bit_plot_ch[ch].hide()
        raw_plot_ch[section].show()
        bit_plot_ch[section].show()
    elif section == 8:
        x_range, y_range = raw_plot.viewRange()
        para_table.item(0, 8).setText(str(int(y_range[0] // -1000) * 10))
        if air_mattress == 0:
            raw_plot_ch[6].show()
        else:
            raw_plot_ch[7].show()
    elif section == 9:
        update_raw_plot()
        update_bit_plot()

#--------------------------------------------------------------------------
def vertical_header_clicked(section):
    global raw_plot_ch
    global bit_plot_ch

    for ch in range(6):
        raw_plot_ch[ch].show()
        bit_plot_ch[ch].show()

def cell_clicked(row, column):
    if row == 2 and column == 8:
        air_mattress  = int(para_table.item(2, 8).text())
        air_mattress = 1 - air_mattress
        para_table.item(2, 8).setText(str(int(air_mattress)))
        if air_mattress == 0:
            raw_plot_ch[6].show()
            raw_plot_ch[7].hide()
        else:
            raw_plot_ch[7].show()
            raw_plot_ch[6].hide()

#--------------------------------------------------------------------------
def generate_annotation():
    if not 'd10' in globals():
        status_bar.showMessage('請先載入資料!')
        return
        
    status_bar.showMessage('產生標注檔案中...')
    QApplication.processEvents()
    
    # 取得所有通道的基準值回歸點
    annotations = []
    for ch in range(6):
        med10 = d10[ch] + offset_edit[ch]
        baseline = base_final[ch]
        
        # 找出數值回到基準值的時間點
        back_to_base = np.where(np.abs(med10 - baseline) < 10)[0]
        
        # 過濾出有效的離床點(前一個點要是在床上)
        valid_points = []
        for idx in back_to_base:
            if idx > 0 and np.abs(med10[idx-1] - baseline[idx-1]) > 100:
                valid_points.append(idx)
                
        annotations.append(valid_points)
    
    # 合併所有通道的標注點
    all_points = []
    for ch_points in annotations:
        all_points.extend([(point, ch) for point in ch_points for ch in range(6)])
    
    # 按時間排序
    all_points.sort()
    
    # 寫入CSV檔案
    filename = f"{cmb_name[:-4]}_annotations.csv"
    filepath = os.path.join(LOG_DIR, filename)
    
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Timestamp', 'Channel', 'Event'])
        
        for point, channel in all_points:
            timestamp = startday + timedelta(seconds=t1sec[point])
            writer.writerow([timestamp.strftime('%Y-%m-%d %H:%M:%S'), 
                           f'Channel {channel+1}',
                           'Off Bed'])
            
    status_bar.showMessage(f'標注檔案已儲存: {filename}')

#--------------------------------------------------------------------------
# Create the QTableWidget and add it to the layout
startday = datetime.now()

radio_Normal = QRadioButton('Normal')
radio_Test = QRadioButton('Test')
radio_Normal.setChecked(True)

Read_cmb = QPushButton('OPEN_CASE')
Read_cmb.clicked.connect(OpenCMBDialog)
Read_cmb.setToolTip('開啟合併檔(.CMB)檔案')

data_source = QComboBox()
data_source.addItem('FTP:\\RAW')
data_source.addItem('FTP:\\RAW_COLLECT')
data_source.addItem('RD_FTP:\\RAW')
data_source.addItem('FTP:\\BCGRAW')
data_source.addItem('RD_FTP:\\BCGRAW')
data_source.addItem('Elastic')
#data_source.addItem('Z:\RAW')
#data_source.addItem('Z:\RAW_COLLECT')
data_source.textActivated.connect(loadClicked)
data_source.setToolTip('選擇FTP下載目錄')

# 在第 0 列的第 1 個位置，加入一個 QLineEdit 物件，並設置初始值為 "SPS2021PA000000"
iCueSN = QLineEdit()
iCueSN.setText('SPS2025PA000146')
iCueSN.setFixedWidth(120)
iCueSN.setToolTip('輸入iCue編號')
# Get the current datetime in GMT
now_utc = datetime.now(pytz.utc)       
# Convert the datetime to GMT+8
gmt8 = pytz.timezone('Asia/Singapore')
today = now_utc.astimezone(gmt8)
yesterday = today - timedelta(days=1)

# 在第 0 列的第 3 個位置，加入一個 QLineEdit 物件，並設置初始值為 "20220220_000000"

start_time = QLineEdit()  # 建立起始時間的 QLineEdit 元件
start_time.setText(yesterday.strftime('%Y%m%d_035000'))  # 設定起始時間為昨天，格式為 '%Y%m%d_040000'
start_time.setFixedWidth(120)
start_time.mouseDoubleClickEvent = start_calendar  # 設定起始時間元件的雙擊事件為 start_calendar 方法
start_time.setToolTip('選擇開始日期')

end_time = QLineEdit()  # 建立結束時間的 QLineEdit 元件
end_time.setText(today.strftime('%Y%m%d_040000'))  # 設定結束時間為今天，格式為 '%Y%m%d_040000'
end_time.setFixedWidth(120)
end_time.mouseDoubleClickEvent = end_calendar  # 設定結束時間元件的雙擊事件為 end_calendar 方法
end_time.setToolTip('選擇結束日期')

check_96DPI = QCheckBox('96DPI')  # 新增一個 QCheckBox
check_96DPI.stateChanged.connect(display_96DPI)

check_NightMode = QCheckBox('Night Mode')
check_NightMode.setChecked(True)
check_NightMode.stateChanged.connect(change_NightMode)
check_NightMode.setChecked(False)

# 在 UI 元件初始化部分新增
check_get_para = QCheckBox('Get Parameters First')
check_get_para.setToolTip('在下載資料前先獲取MQTT參數')

# 在主視窗的初始化程式碼中新增：
json_button = QtWidgets.QPushButton("開啟 JSON")
json_button.clicked.connect(OpenJsonFile)

# 標記相關按鈕
marker_btn = QPushButton('開始標記')
marker_btn.clicked.connect(toggle_marker)
marker_btn.setToolTip('顯示/隱藏標記線')

marker_type_combo = QComboBox()
marker_type_combo.addItems(['離床', '上床', '翻身'])
marker_type_combo.setToolTip('選擇要標記的事件類型')

save_marker_btn = QPushButton('儲存標記')
save_marker_btn.clicked.connect(save_marker)
save_marker_btn.setToolTip('儲存目前標記線位置')

# 新增數值顯示欄位
value_display_label = QLabel("計算值: ")
value_display_label.setStyleSheet("font-weight: bold; color: black;")
value_display = QLabel("0.00")
value_display.setFixedWidth(100)  # 加大寬度
value_display.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
value_display.setStyleSheet("background-color: white; color: black; border: 2px solid #666; border-radius: 4px; padding: 3px; font-weight: bold; font-size: 14px;")
value_display.setToolTip('演算法計算的數值')

# 更新 MQTT 參數
Mqtt_set = QPushButton('MQTT_SET_PARA')
# Mqtt_set.clicked.connect(MqttSetDialog)
# Mqtt_set.setToolTip('MQTT設定��數')



Mqtt_get = QPushButton('MQTT_GET_PARA')
# Mqtt_get.clicked.connect(MQTT_get_reg)
# Mqtt_get.setToolTip('MQTT讀取參數')




#--------------------------------------------------------------------------
# Create the QTableWidget and add it to the layout
para_table = QTableWidget()
#layout.addWidget(table)
# Set the number of rows and columns in the table
para_table.setRowCount(4)
para_table.setColumnCount(10)
# Add some data to the table
para_table.setVerticalHeaderLabels(['min_preload', 'threshold_1', 'threshold_2', 'offset level'])
para_table.setHorizontalHeaderLabels(['Channel 1   ','Channel 2   ','Channel 3   ','Channel 4   ','Channel 5   ','Channel 6   ', 'Total Sum ', 'Noise 1  ', 'Set Flip ', 'UPDATE   '])
data = [
    ('0', '0', '0', '0'),
    ('0', '0', '0', '0'),
    ('0', '0', '0', '0'),
    ('0', '0', '0', '0'),
    ('0', '0', '0', '0'),
    ('0', '0', '0', '0'),
    ('0', '', '', ''),
    ('0', 'Noise 2  ', '0', ''),
    ('0', 'Normal / Air', '0', ''),
    ('', '', '', '')
]
for row, rowData in enumerate(data):
    for col, itemData in enumerate(rowData):
        item = QTableWidgetItem(itemData)
        para_table.setItem(col, row, item)
        item.setTextAlignment(Qt.AlignHCenter)
        if row < 6:
            c = hex_to_rgb(hex_colors[row])
            item.setBackground(QColor(c[0],c[1],c[2]))
        elif row == 6 and col == 0:
            c = hex_to_rgb(hex_colors[row])
            item.setBackground(QColor(c[0],c[1],c[2]))        
        elif row == 7 and (col == 0 or col == 2):
            c = hex_to_rgb(hex_colors[row])
            item.setBackground(QColor(c[0],c[1],c[2]))
        elif row == 8 and (col == 0 or col == 2):
            c = hex_to_rgb(hex_colors[row])
            item.setBackground(QColor(c[0],c[1],c[2]))

para_table.resizeColumnsToContents()
para_table.resizeRowsToContents()

# Connect the horizontal header click event
horizontal_header = para_table.horizontalHeader()
horizontal_header.sectionClicked.connect(horizontal_header_clicked)
# Connect the vertical header click event
vertical_header = para_table.verticalHeader()
vertical_header.sectionClicked.connect(vertical_header_clicked)
# Connect the cell click event
para_table.cellClicked.connect(cell_clicked)

#----------------------------------------------------------------------------------------
layout = QtWidgets.QGridLayout(cw)
cw.setLayout(layout)

# 第一行按鈕布局
row_widget = QtWidgets.QWidget()
row_layout = QtWidgets.QHBoxLayout(row_widget)
row_layout.setContentsMargins(0, 0, 0, 0)
row_layout.addWidget(iCueSN)
row_layout.addWidget(start_time)
row_layout.addWidget(end_time)
row_layout.addWidget(data_source)
row_layout.addWidget(radio_Normal)
row_layout.addWidget(radio_Test)
row_layout.addWidget(Read_cmb)
row_layout.addWidget(check_96DPI)
row_layout.addWidget(check_NightMode)
row_layout.addWidget(Mqtt_set)
row_layout.addWidget(Mqtt_get)
# 第二行按鈕布局
marker_row_widget = QtWidgets.QWidget()
marker_row_layout = QtWidgets.QHBoxLayout(marker_row_widget)
marker_row_layout.setContentsMargins(0, 0, 0, 0)
marker_row_layout.setAlignment(QtCore.Qt.AlignLeft)  # 設置按鈕靠左對齊

# 移除這些行，看起來重複的按鈕
marker_row_layout.addWidget(check_get_para)
marker_row_layout.addWidget(json_button)

marker_row_layout.addWidget(marker_btn)
marker_row_layout.addWidget(marker_type_combo)
marker_row_layout.addWidget(save_marker_btn)

# 添加數值顯示欄位
marker_row_layout.addWidget(value_display_label)
marker_row_layout.addWidget(value_display)

# 添加彈性空間，吸收多餘空間
marker_row_layout.addStretch(1)

# 將兩個行布局加入到主布局
layout.addWidget(row_widget, 0, 0, 1, 3)
layout.addWidget(marker_row_widget, 1, 0, 1, 3)
layout.addWidget(status_bar, 5, 0, 1, 9)

#row_layout.addWidget(self.wav_gain)  # 在佈局中添加self.data_source下拉選單，位置為(0, 0)
layout.addWidget(raw_plot,   2, 0, 1, 10)  # wav_plot 放置在第 1 行、第 0 列
layout.addWidget(para_table, 4, 0, 1, 10)
layout.addWidget(bit_plot,   3, 0, 1, 10)  # wav_plot 放置在第 1 行、第 0 列
layout.setColumnStretch(0,10)

layout.setRowStretch(0,1)  # 第一行按鈕
layout.setRowStretch(1,1)  # 第二行標記按鈕
layout.setRowStretch(2,20) # 原始數據圖表
layout.setRowStretch(3,20) # 位元圖表
layout.setRowStretch(4,8)  # 參數表格
layout.setRowStretch(5,1)  # 狀態欄
bit_plot.setXLink(raw_plot) 


# 獲取當前腳本檔案的絕對路徑
script_path = os.path.abspath(__file__)
# 列印檔案名稱和修改日期
script_name = script_path.split('\\')[-1]
mw.setWindowTitle(f'OnOFF Bed   ({script_name})')  # 設置視窗標題為'OnOFF Bed'
mw.setWindowIcon(QIcon('Humetrics.ico'))  # 設置視窗圖標為'Humetrics.ico'

# 顯示窗口
mw.show()

# # 使用定時器在窗口顯示後設置計算值
# timer = QtCore.QTimer()
# timer.singleShot(500, lambda: update_calculated_value(123.45))

app.exec_()

