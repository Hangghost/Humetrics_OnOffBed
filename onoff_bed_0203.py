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
from PyQt5.QtWidgets import QApplication, QTableWidget, QTableWidgetItem, QRadioButton, QCheckBox, qApp, QHeaderView
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

    timeout = 20
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
        client.tls_set('./cert/humetric_mqtt_certificate.pem', None, None, cert_reqs=ssl.CERT_NONE)
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
def GetParaTable():
    global preload_edit
    global th1_edit
    global th2_edit
    global offset_edit
    global bed_threshold
    global noise_onbed
    global noisd_offbed
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
    noisd_offbed  = int(para_table.item(2, 7).text())
    dist_thr      = int(para_table.item(0, 8).text())
    air_mattress  = int(para_table.item(2, 8).text())

#-----------------------------------------------------------------------    
def OpenCmbFile():
    global cmb_name
    status_bar.showMessage('Reading ' + cmb_name + ' ........')
    QApplication.processEvents()  

    global n10
    global d10
    global x10
    global n10_sel
    global d10_sel
    global x10_sel

    #---------------------------------------------------------
    with open(f'{cmb_name[:-4]}.txt', mode='r', newline='') as file:
        reader = csv.reader(file)
        t = []
        filelen = []
        # 逐行读取数据并将其添加到列表中
        for row in reader:
            dt = datetime.strptime(row[0], "%Y%m%d_%H%M%S.dat")
            gmt = pytz.timezone('GMT')  # 建立 GMT+0 時區的時間
            gmt_dt = gmt.localize(dt)                
            tz = pytz.timezone('Asia/Taipei') # 轉換成 GMT+8 時區的時間
            tw_dt = gmt_dt.astimezone(tz)
            t.append(tw_dt)
            filelen.append(int(row[1]))

        t = np.array(t)
        filelen = np.array(filelen)
        bcg = np.median(filelen) == 3000

    #---------------------------------------------------------
    with open(cmb_name, "rb") as f:            
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

    #---------------------------------------------------------
    if radio_Normal.isChecked():
        reg_table = MQTT_get_reg("mqtt.humetrics.ai", "device", "!dF-9DXbpVKHDRgBryRJJBEdqCihwN", iCueSN.text())
    else:
        reg_table = MQTT_get_reg("rdtest.mqtt.humetrics.ai", "device", "BMY4dqh2pcw!rxa4hdy", iCueSN.text())

    if len(reg_table) > 0:
        for ch in range(6):
            para_table.item(0, ch).setText(str(reg_table[str(ch + 42)]))
            para_table.item(1, ch).setText(str(reg_table[str(ch + 48)]))
            para_table.item(2, ch).setText(str(reg_table[str(ch + 58)]))

        para_table.item(0, 6).setText(str(reg_table[str(41)]))

        para_table.item(2, 7).setText(str(reg_table[str(54)]))
        para_table.item(0, 7).setText(str(reg_table[str(55)]))

        para_table.item(0, 8).setText(str(reg_table[str(56)]))
        para_table.item(2, 8).setText(str(reg_table[str(57)]))
    else:
        default_reg_table = {  # 預設值定義
            "41": "30000",
            "42": "40000", "43": "40000", "44": "40000", "45": "40000", "46": "40000", "47": "40000",
            "48": "60000", "49": "60000", "50": "60000", "51": "60000", "52": "60000", "53": "60000",
            "54": "80", "55": "80", "56": "400", "57": "0", 
            "58": "90000", "59": "90000", "60": "90000", "61": "90000", "62": "90000", "63": "90000"
        }
        # 使用 default_reg_table 處理
        for ch in range(6):
            para_table.item(0, ch).setText(str(default_reg_table[str(ch + 42)]))
            para_table.item(1, ch).setText(str(default_reg_table[str(ch + 48)]))
            para_table.item(2, ch).setText(str(default_reg_table[str(ch + 58)]))

        para_table.item(0, 6).setText(str(default_reg_table[str(41)]))

        para_table.item(2, 7).setText(str(default_reg_table[str(54)]))
        para_table.item(0, 7).setText(str(default_reg_table[str(55)]))

        para_table.item(0, 8).setText(str(default_reg_table[str(56)]))
        para_table.item(2, 8).setText(str(default_reg_table[str(57)]))

    # 重新塑形數組以分開通道
    data = int_data.reshape(-1, 6)
    global data_bcg
    #data_bcg = [x, y, z]

    # --------------------------------------------------------------------
    lpf = [26, 28, 32, 39, 48, 60, 74, 90, 108, 126, 146, 167, 187, 208, 227, 246, 264, 280, 294, 306, 315, 322, 326, 328, 326, 322, 315, 306, 294, 280, 264, 246, 227, 208, 187, 167, 146, 126, 108, 90, 74, 60, 48, 39, 32, 28, 26]        
    
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
    dist = 0
    dist_air = 0

    for ch in range(6):
        status_bar.showMessage(f'Processing CH{ch+1} ........')
        QApplication.processEvents()
        # --------------------------------------------------------
        hp = np.convolve(data[:,ch], [-1, -2, -3, -4, 4, 3, 2, 1], mode='same')
        n = np.convolve(np.abs(hp / 16), lpf, mode='full')
        n = n[10:-37] / 4096
        n = n[::10]
        n10.append(np.int32(n))
        # --------------------------------------------------------
        data_pd = pd.Series(data[:,ch]) # 將通道的數據轉換為Pandas的Series數據結構
        med10 = data_pd.rolling(window=10, min_periods=1, center=True, axis=0).mean() # 計算每個窗口的最大值，窗口大小為30 
        med10 = np.array(med10)
        med10 = med10[::10]
        d10.append(np.int32(med10))
        # --------------------------------------------------------
        max10 = data_pd.rolling(window=10, min_periods=1, center=True, axis=0).max() # 計算每個窗口的最大值，窗口大小為30 
        max10 = np.array(max10)
        max10 = np.int32(max10[::10])
        x10.append(np.int32(max10))
        # --------------------------------------------------------
        resp = data[:,ch] - savgol_filter(data[:,ch], 105, 3)
        resp = np.repeat(resp, 10).astype(np.float64)
        resp = savgol_filter(resp, 151, 3)
        #data_bcg.append(resp)
        # 計算 dist ----------------------------------------------   
        a = [1, -1023/1024]
        b = [1/1024, 0]
        pos_iirmean = lfilter(b, a, med10) # 1 second
        med10_pd = pd.Series(med10)
        mean_30sec = med10_pd.rolling(window=30, min_periods=1, center=False, axis=0).mean() # 計算每個窗口的最大值，窗口大小為30 
        mean_30sec = np.int32(mean_30sec)        
        diff = (mean_30sec - pos_iirmean) / 256
        if ch == 1:
            diff = diff / 3
        dist = dist + np.square(diff)
        dist[dist > 8000000] = 8000000
        # 計算 dist (air mattress) -------------------------------
        mean_60sec = med10_pd.rolling(window=60, min_periods=1, center=False, axis=0).mean() # 計算每個窗口的最大值，窗口大小為60 
        mean_60sec = np.int32(mean_60sec)
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
        diff = lfilter(b, a, mean_60sec) # 1 second
        if ch == 1:
            diff = diff / 3
        dist_air = dist_air + np.square(diff / 256)

    # Convert to pandas Series
    dist = pd.Series(dist)
    # Calculate the difference with a shift of 60
    shift = 60
    rising_dist = dist.shift(-shift) - dist
    # Fill NaN values resulting from the shift to maintain the same length as the original array
    rising_dist = np.int32(rising_dist.fillna(0))
    rising_dist[rising_dist < 0] = 0
    rising_dist = rising_dist // 127
    rising_dist[rising_dist > 1000] = 1000

    # Convert to pandas Series
    dist_air = pd.Series(dist_air)
    # Calculate the difference with a shift of 60
    shift = 60
    rising_dist_air = dist_air.shift(-shift) - dist_air
    # Fill NaN values resulting from the shift to maintain the same length as the original array
    rising_dist_air = np.int32(rising_dist_air.fillna(0))
    rising_dist_air[rising_dist_air < 0] = 0
    rising_dist_air = rising_dist_air // 127
    rising_dist_air[rising_dist_air > 1000] = 1000
    
    global idx10
    global idx1sec
    # 由檔案開始時間/資料長度計算對應 index --------------------------------------------   
    idx1sec = np.array([])
    idx100 = np.array([])
    idx10 = np.array([])
    idx100_sum = 0
    idx10_sum = 0

    # 迭代資料的每一個索引 i
    for i in range(filelen.shape[0]):

        if i < filelen.shape[0] - 1:
            t_diff = np.int32((t[i + 1] - t[i]).total_seconds())  # 計算相鄰時間之差（秒）
        else:
            t_diff = 600  # 若為最後一個數據，設定時間差為 600 秒

        if t_diff > 660:
            t_blank = t_diff - 600  # 若時間差超過 660 秒，計算空白時間
            t_diff = 600  # 設定時間差為 600 秒
        else:
            t_blank = 0

        # 將索引值加入對應的陣列中
        idx1sec = np.append(idx1sec, np.floor(np.linspace(0, filelen[i]//10 - 1, t_diff)) + idx100_sum//10)
        idx100 = np.append(idx100, np.floor(np.linspace(0, filelen[i] - 1, t_diff*10)) + idx100_sum)
        idx10 = np.append(idx10, np.floor(np.linspace(0, filelen[i]*10 - 1, t_diff*100)) + idx10_sum)
        idx1sec = np.append(idx1sec, np.tile(filelen[i]//10 - 1 + idx100_sum//10, (t_blank, 1)))  # 將空白時間的索引值重複添加
        idx100 = np.append(idx100, np.tile(filelen[i] - 1 + idx100_sum, (t_blank*10, 1)))  # 將空白時間的索引值重複添加
        idx10 = np.append(idx10, np.tile(filelen[i]*10 - 1 + idx10_sum, (t_blank*100, 1)))  # 將空白時間的索引值重複添加

        idx100_sum = idx100_sum + filelen[i]  # 更新索引值總和
        idx10_sum = idx10_sum + filelen[i]*10  # 更新索引值總和
        

    # 計算不同取樣率對應的時間 --------------------------------------------------------    
    st = (t[0] - t[0].replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds()
    global startday
    startday = t[0].replace(hour=0, minute=0, second=0, microsecond=0)

    global t10ms
    global t1sec
    t1sec = np.array(range(np.int32((t[-1] - t[0]).total_seconds()) + 600)) + st
    idx1sec = np.int32(idx1sec)
    idx100 = np.int32(idx100)
    idx10 = np.int32(idx10)
    t100ms = np.linspace(t1sec[0], t1sec[-1], t1sec.shape[0]*10)
    t10ms = np.linspace(t1sec[0], t1sec[-1], t1sec.shape[0]*100)
    t100ms = np.around(t100ms, decimals=1)
    t10ms = np.around(t10ms, decimals=2)

    # for ch in range(9):   
    #     data_bcg[ch] = data_bcg[ch][idx10]

    ##--------------------------------------
    x_range, y_range = raw_plot.viewRange()
    center = (x_range[0] + x_range[1]) / 2

    st = center - 10
    ed = center + 10
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

    pen = pg.mkPen(color=hex_to_rgb(hex_colors[6]))
    x = t1sec[::flip_interval]
    y = rising_dist[idx1sec[::flip_interval]] * -100
    raw_plot_ch.append(raw_plot.plot(x, y, pen=pen, name=f'Normal'))

    raw_plot_ch[6].hide()

    pen = pg.mkPen(color=hex_to_rgb(hex_colors[7]))
    x = t1sec[::flip_interval]
    y = rising_dist_air[idx1sec[::flip_interval]] * -100
    raw_plot_ch.append(raw_plot.plot(x, y, pen=pen, name=f'Air'))    

    raw_plot_ch[7].hide()

    pen = pg.mkPen(color=hex_to_rgb(hex_colors[8]))
    x = np.array([t1sec[0], t1sec[-1]])
    y = np.array([dist_thr, dist_thr]) * -100
    raw_plot_ch.append(raw_plot.plot(x, y, pen=pen, name=f'Threshold'))  

    if air_mattress == 0:
        flip = (rising_dist > dist_thr) * dist_thr
    else:
        flip = (rising_dist_air > dist_thr) * dist_thr

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
    bit_plot_onff = bit_plot.plot(t1sec, onbed - 8.5, fillLevel=-7.5, brush=pg.mkBrush(color=hex_to_rgb(hex_colors[0])), pen=pg.mkPen(color=hex_to_rgb(hex_colors[0])), name='OFFBED')

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

    global zdata_final
    zdata_final = []

    global base_final
    base_final = []

    l = d10[0].shape[0]
    onbed = np.zeros((l,))
    onload = []
    total = 0

    for ch in range(6):
        max10 = x10[ch] + offset_edit[ch]
        med10 = d10[ch] + offset_edit[ch]
        n = n10[ch]
        preload = preload_edit[ch]
        zeroing = np.less(n * np.right_shift(max10, 5), noisd_offbed * np.right_shift(preload, 5))
        th1 = th1_edit[ch]
        th2 = th2_edit[ch]
        approach = max10 - (th1 + th2)
        speed = n // (noise_onbed * 4)            
        np.clip(speed, 1, 16, out=speed)
        app_sp = approach * speed
        sp_1024 = 1024 - speed
        #----------------------------------------------------
        base = (app_sp[0] // 1024 + med10[0]) // 2
        base = np.int64(base)
        baseline = np.zeros_like(med10)           
        for i in range(l):
            if zeroing[i]:
                base = np.int64(med10[i])
            base = (base * sp_1024[i] + app_sp[i]) // 1024
            baseline[i] = base            
        
        total = total + med10[:] - baseline
        o = np.less(th1, med10[:] - baseline)
        onload.append(o)
        onbed = onbed + o

        d_zero = med10 - baseline
        zdata_final.append(d_zero)
        base_final.append(baseline)

    onbed = onbed + np.less(bed_threshold, total) 
    onbed = np.int32(onbed > 0)
    
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
    ST_TIME = start_time.text()
    ED_TIME = end_time.text()
    n = 0
    bcg = 0
    if data_source.currentText() == 'FTP:\RAW':
        FTP_ADDR = 'raw.humetrics.ai'
        USER = 'Robot'
        PASW = 'HM66050660'
        FILE_PATH = '/ESP32_DEVICES/' + iCueSN.text() + '/RAW/'
        n = download_files_by_time_range(FILE_PATH, ST_TIME, ED_TIME, FTP_ADDR, USER, PASW, 144000)
    if data_source.currentText() == 'RD_FTP:\RAW':
        FTP_ADDR = 'raw.humetrics.ai'
        USER = 'ESP32'
        PASW = 'HM66050660'
        FILE_PATH = '/SmartBed_ESP32VS/' + iCueSN.text() + '/RAW/'
        n = download_files_by_time_range(FILE_PATH, ST_TIME, ED_TIME, FTP_ADDR, USER, PASW, 144000)
    if data_source.currentText() == 'FTP:\BCGRAW':
        FTP_ADDR = 'raw.humetrics.ai'
        USER = 'Robot'
        PASW = 'HM66050660'
        FILE_PATH = '/ESP32_DEVICES/' + iCueSN.text() + '/BCGRAW/'
        n = download_files_by_time_range(FILE_PATH, ST_TIME, ED_TIME, FTP_ADDR, USER, PASW, 165000)
        bcg = 1
    if data_source.currentText() == 'RD_FTP:\BCGRAW':
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

    global cmb_name

    cmb_name = iCueSN.text() + '_' + ST_TIME[:-4] + '_' + ED_TIME[:-4] + '.cmb'

    # 打開輸出檔案以二進制追加模式
    filelen = []  
    with open(cmb_name, "ab") as comb_file:
        for file_name in file_list:
            if bcg == 1:
                filelen.append(os.path.getsize(file_name)//55)
            else:
                filelen.append(os.path.getsize(file_name)//24)
            with open(file_name, "rb") as infile:
                # 將輸入檔案的內容複製到輸出檔案
                shutil.copyfileobj(infile, comb_file) 
                status_bar.showMessage('combining ' + file_name)   

    with open(f'{cmb_name[:-4]}.txt', mode='w', newline='') as file:
        writer = csv.writer(file)
        data_to_write = [[file_list[i], filelen[i]] for i in range(len(file_list))]            
        # 将数据逐行写入CSV文件
        for row in data_to_write:
            writer.writerow(row)   

    para = 'del *.dat'
    returned_value = os.system(para)                

    if cmb_name:
        OpenCmbFile()

#--------------------------------------------------------------------------
def MQTT_set_reg(mqtt_server, username, password, sn, payload):
    # Create a MQTT client
    client = mqtt.Client()
    client.username_pw_set(username, password)
    if radio_Normal.isChecked():
        client.tls_set('./cert/humetric_mqtt_certificate.pem', None, None, cert_reqs=ssl.CERT_NONE)
        client.connect(mqtt_server, 8883, 60)
    else:
        client.connect(mqtt_server, 1883, 60)
    
    # Create the message payload / Publish the message
    topic_set_regs = "algoParam/" + sn + "/set"
    payload.update({'taskID':4881})
    client.publish(topic_set_regs, json.dumps(payload))

    client.disconnect() # Stop the MQTT client loop when done

#--------------------------------------------------------------------------
def MqttGetDialog():
    reg_table = {}

    if radio_Normal.isChecked():
        reg_table = MQTT_get_reg("mqtt.humetrics.ai", "device", "!dF-9DXbpVKHDRgBryRJJBEdqCihwN", iCueSN.text())
    else:
        reg_table = MQTT_get_reg("rdtest.mqtt.humetrics.ai", "device", "BMY4dqh2pcw!rxa4hdy", iCueSN.text())

    if len(reg_table) > 0:
        for ch in range(6):
            para_table.item(0, ch).setText(str(reg_table[str(ch + 42)]))
            para_table.item(1, ch).setText(str(reg_table[str(ch + 48)]))
            para_table.item(2, ch).setText(str(reg_table[str(ch + 58)]))

        para_table.item(0, 6).setText(str(reg_table[str(41)]))

        para_table.item(2, 7).setText(str(reg_table[str(54)]))
        para_table.item(0, 7).setText(str(reg_table[str(55)]))

        para_table.item(0, 8).setText(str(reg_table[str(56)]))
        para_table.item(2, 8).setText(str(reg_table[str(57)]))
    else:
        status_bar.showMessage('Get MQTT parameters Error !')
        QApplication.processEvents()


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
    legend_raw.setBrush(legend_bg)
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
# Create the QTableWidget and add it to the layout
global startday
startday = datetime.now()

radio_Normal = QRadioButton('Normal')
radio_Test = QRadioButton('RD site')
radio_Normal.setChecked(True)

Read_cmb = QPushButton('OPEN_CASE')
Read_cmb.clicked.connect(OpenCMBDialog)
Read_cmb.setToolTip('開啟合併檔(.CMB)檔案')

data_source = QComboBox()
data_source.addItem('FTP:\RAW')
data_source.addItem('RD_FTP:\RAW')
data_source.addItem('FTP:\BCGRAW')
data_source.addItem('RD_FTP:\BCGRAW')
#data_source.addItem('Z:\RAW')
#data_source.addItem('Z:\RAW_COLLECT')
data_source.textActivated.connect(loadClicked)
data_source.setToolTip('選擇FTP下載目錄')

# 在第 0 列的第 1 個位置，加入一個 QLineEdit 物件，並設置初始值為 "SPS2021PA000000"
iCueSN = QLineEdit()
iCueSN.setText('SPS2022HB000000')
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
start_time.setText(yesterday.strftime('%Y%m%d_040000'))  # 設定起始時間為昨天，格式為 '%Y%m%d_040000'
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

Mqtt_get = QPushButton('MQTT_GET_PARA')
Mqtt_get.clicked.connect(MqttGetDialog)
Mqtt_get.setToolTip('取得MQTT參數')

Mqtt_set = QPushButton('MQTT_SET_PARA')
Mqtt_set.clicked.connect(MqttSetDialog)
Mqtt_set.setToolTip('設定MQTT參數')

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

#row_layout.addWidget(self.wav_gain)  # 在佈局中添加self.data_source下拉選單，位置為(0, 0)
layout.addWidget(raw_plot,   1, 0, 1, 10)  # wav_plot 放置在第 1 行、第 0 列
layout.addWidget(para_table, 2, 0, 1, 10)
layout.addWidget(bit_plot,   3, 0, 1, 10)  # wav_plot 放置在第 1 行、第 0 列
layout.setColumnStretch(0,10)

layout.setRowStretch(0,1)
layout.setRowStretch(1,20)
layout.setRowStretch(2,8)
layout.setRowStretch(3,20)
bit_plot.setXLink(raw_plot) 

row_widget = QtWidgets.QWidget()
row_layout = QtWidgets.QHBoxLayout(row_widget)
row_layout.setContentsMargins(0, 0, 0, 0)
row_layout.addWidget(iCueSN)
row_layout.addWidget(start_time)
row_layout.addWidget(end_time)
row_layout.addWidget(data_source)
#row_layout.addWidget(Ftp_raw)
row_layout.addWidget(Read_cmb)
row_layout.addWidget(check_96DPI)
row_layout.addWidget(check_NightMode)
row_layout.addWidget(radio_Normal)
row_layout.addWidget(radio_Test)
row_layout.addWidget(Mqtt_get)
row_layout.addWidget(Mqtt_set)

layout.addWidget(row_widget,0,0,1,3)
layout.addWidget(status_bar,5,0,1,9)

# 獲取當前腳本檔案的絕對路徑
script_path = os.path.abspath(__file__)
# 列印檔案名稱和修改日期
script_name = script_path.split('\\')[-1]
mw.setWindowTitle(f'OnOFF Bed   ({script_name})')  # 設置視窗標題為'OnOFF Bed'
mw.setWindowIcon(QIcon('Humetrics.ico'))  # 設置視窗圖標為'Humetrics.ico'

mw.show()
#mw.setGeometry(1, 50, 1920, 1080)
app.exec_()
