#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
離群設備截圖PDF生成工具
用於將批量處理中識別的離群設備截圖整合成PDF報告
"""

import os
import glob
from datetime import datetime

def collect_outlier_screenshots(common_outlier_devices, log_dir):
    """收集離群設備資料的截圖文件（基於設備+日期）"""
    screenshot_dir = os.path.join(log_dir, 'batch_screenshots')
    found_screenshots = []
    
    if not os.path.exists(screenshot_dir):
        print(f"截圖目錄不存在: {screenshot_dir}")
        return found_screenshots
    
    # 列出所有截圖文件
    screenshot_files = [f for f in os.listdir(screenshot_dir) if f.endswith('.png')]
    
    for record_info in common_outlier_devices:
        try:
            device_sn = record_info['device_sn']
            start_date = record_info['start_date']
            start_hour = int(record_info['start_hour'])  # 確保轉換為整數
            end_date = record_info['end_date']
            end_hour = int(record_info['end_hour'])      # 確保轉換為整數
            
            # 構建預期的截圖文件名模式
            # 截圖文件名格式通常是: {device_sn}_{start_date}_{start_hour}_{end_date}_{end_hour}_*.png
            expected_pattern = f"{device_sn}_{start_date}_{start_hour:02d}_{end_date}_{end_hour:02d}"
            
            # 尋找匹配的截圖文件
            matching_screenshots = []
            for screenshot_file in screenshot_files:
                if screenshot_file.startswith(expected_pattern):
                    screenshot_path = os.path.join(screenshot_dir, screenshot_file)
                    if os.path.exists(screenshot_path):
                        matching_screenshots.append({
                            'device_sn': device_sn,
                            'file_path': screenshot_path,
                            'file_name': screenshot_file,
                            'record_info': record_info,
                            'time_period': f"{start_date} {start_hour:02d}:00 - {end_date} {end_hour:02d}:00"
                        })
            
            if matching_screenshots:
                # 按文件名排序（這樣時間順序會是正確的）
                matching_screenshots.sort(key=lambda x: x['file_name'])
                found_screenshots.extend(matching_screenshots)
            else:
                print(f"未找到設備 {device_sn} 在時間段 {start_date} {start_hour:02d}:00 - {end_date} {end_hour:02d}:00 的截圖文件")
        
        except (ValueError, KeyError) as e:
            print(f"處理設備資料時發生錯誤: {record_info.get('device_sn', 'Unknown')}, 錯誤: {e}")
            continue
    
    return found_screenshots

def generate_outlier_screenshots_pdf(common_outlier_devices, results_dir, timestamp, log_dir):
    """生成離群設備資料截圖的PDF報告（基於設備+日期）"""
    try:
        from reportlab.lib.pagesizes import A4, landscape
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, Table, TableStyle
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.lib import colors
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        import platform
        
        # 收集截圖文件
        screenshots = collect_outlier_screenshots(common_outlier_devices, log_dir)
        
        if not screenshots:
            print("未找到任何離群設備資料的截圖文件")
            return None
        
        # 生成PDF文件路徑
        pdf_filename = f"outlier_device_data_screenshots_{timestamp}.pdf"
        pdf_path = os.path.join(results_dir, pdf_filename)
        
        # 創建PDF文檔（使用橫向A4）
        doc = SimpleDocTemplate(pdf_path, pagesize=landscape(A4))
        story = []
        
        # 註冊中文字體
        chinese_font_name = 'ChineseFont'
        try:
            # 根據操作系統選擇字體路徑
            system = platform.system()
            if system == "Darwin":  # macOS
                font_paths = [
                    '/System/Library/Fonts/Supplemental/Arial Unicode.ttf',
                    '/System/Library/Fonts/PingFang.ttc',
                    '/System/Library/Fonts/Arial Unicode.ttc',
                    '/Library/Fonts/Arial Unicode.ttf',
                    '/System/Library/Fonts/Helvetica.ttc'
                ]
            elif system == "Windows":
                font_paths = [
                    'C:/Windows/Fonts/simhei.ttf',
                    'C:/Windows/Fonts/simsun.ttc',
                    'C:/Windows/Fonts/msyh.ttc'
                ]
            else:  # Linux
                font_paths = [
                    '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
                    '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
                ]
            
            # 嘗試註冊第一個可用的字體
            font_registered = False
            for font_path in font_paths:
                if os.path.exists(font_path):
                    try:
                        pdfmetrics.registerFont(TTFont(chinese_font_name, font_path))
                        font_registered = True
                        print(f"成功註冊字體: {font_path}")
                        break
                    except Exception as e:
                        print(f"無法註冊字體 {font_path}: {e}")
                        continue
            
            if not font_registered:
                print("警告：無法找到合適的中文字體，使用默認字體")
                chinese_font_name = 'Helvetica'
                
        except Exception as e:
            print(f"字體註冊過程中發生錯誤: {e}")
            chinese_font_name = 'Helvetica'
        
        # 獲取樣式並設定中文字體
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontName=chinese_font_name,
            fontSize=16,
            spaceAfter=30,
            alignment=1  # 居中
        )
        
        device_title_style = ParagraphStyle(
            'DeviceTitle',
            parent=styles['Heading2'],
            fontName=chinese_font_name,
            fontSize=14,
            spaceAfter=15,
            textColor=colors.darkblue
        )
        
        normal_style = ParagraphStyle(
            'ChineseNormal',
            parent=styles['Normal'],
            fontName=chinese_font_name,
            fontSize=12
        )
        
        # 添加標題
        title = Paragraph("離群設備資料截圖報告", title_style)
        story.append(title)
        story.append(Spacer(1, 20))
        
        # 添加摘要資訊
        summary_text = f"""
        <b>報告生成時間：</b>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br/>
        <b>離群設備資料數量：</b>{len(common_outlier_devices)}<br/>
        <b>找到截圖數量：</b>{len(screenshots)}<br/>
        <b>分析指標：</b>平均時差(avg_time_diff)、配對率(match_rate)、精確率(precision)<br/>
        <b>說明：</b>以下為在三個關鍵指標都為離群值的設備資料（設備+特定時間段）
        """
        summary = Paragraph(summary_text, normal_style)
        story.append(summary)
        story.append(Spacer(1, 20))
        
        # 創建設備資料統計表格
        table_data = [['設備序號', '時間段', '平均時差(秒)', '配對率', '精確率', '評分', '截圖']]
        
        for record_info in common_outlier_devices:
            try:
                device_sn = record_info['device_sn']
                # 確保 start_hour 和 end_hour 轉換為整數
                start_hour = int(record_info['start_hour'])
                end_hour = int(record_info['end_hour'])
                time_period = f"{record_info['start_date']} {start_hour:02d}:00 - {record_info['end_date']} {end_hour:02d}:00"
                
                # 檢查是否有對應的截圖
                has_screenshot = any(s['record_info']['filename'] == record_info['filename'] for s in screenshots)
                screenshot_status = "有" if has_screenshot else "無"
                
                table_data.append([
                    device_sn,
                    time_period,
                    f"{record_info['avg_time_diff']:.2f}",
                    f"{record_info['match_rate']:.2f}",
                    f"{record_info['precision']:.2f}",
                    f"{record_info['score']:.2f}",
                    screenshot_status
                ])
            except (ValueError, KeyError) as e:
                print(f"處理設備資料統計表格時發生錯誤: {record_info.get('device_sn', 'Unknown')}, 錯誤: {e}")
                # 添加錯誤行到表格
                table_data.append([
                    record_info.get('device_sn', 'Unknown'),
                    '資料錯誤',
                    'N/A',
                    'N/A',
                    'N/A',
                    'N/A',
                    'N/A'
                ])
        
        table = Table(table_data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, -1), chinese_font_name),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(table)
        story.append(PageBreak())
        
        # 按設備資料分組截圖
        screenshots_by_record = {}
        for screenshot in screenshots:
            record_key = f"{screenshot['device_sn']}_{screenshot['time_period']}"
            if record_key not in screenshots_by_record:
                screenshots_by_record[record_key] = []
            screenshots_by_record[record_key].append(screenshot)
        
        # 為每個設備資料添加截圖頁面
        for record_index, (record_key, record_screenshots) in enumerate(screenshots_by_record.items()):
            # 獲取設備資料資訊
            record_info = record_screenshots[0]['record_info']
            
            # 設備資料標題
            device_title_text = f"設備：{record_info['device_sn']}<br/>時間段：{record_screenshots[0]['time_period']}"
            device_title_text += f"<br/>檔案：{record_info['filename']}"
            device_title_text += f"<br/>時差：{record_info['avg_time_diff']:.2f}秒 | 配對率：{record_info['match_rate']:.2f} | 精確率：{record_info['precision']:.2f} | 評分：{record_info['score']:.2f}"
            
            device_title = Paragraph(device_title_text, device_title_style)
            story.append(device_title)
            story.append(Spacer(1, 10))
            
            # 添加截圖
            for i, screenshot in enumerate(record_screenshots):
                try:
                    # 添加截圖圖片
                    img = Image(screenshot['file_path'])
                    # 調整圖片大小以適合頁面（橫向A4，考慮標題佔用的空間）
                    if i == 0:
                        # 第一張截圖，與標題在同一頁，高度稍小
                        img.drawHeight = 4.5 * inch
                        img.drawWidth = 7 * inch
                    else:
                        # 後續截圖，可以使用更大的空間
                        img.drawHeight = 6 * inch
                        img.drawWidth = 8 * inch
                    
                    story.append(img)
                    
                    # 分頁邏輯：第一張截圖與標題同頁，後續截圖每張一頁
                    if i < len(record_screenshots) - 1:
                        story.append(PageBreak())
                    
                except Exception as e:
                    error_text = f"無法載入截圖：{screenshot['file_name']}，錯誤：{str(e)}"
                    error_paragraph = Paragraph(error_text, normal_style)
                    story.append(error_paragraph)
            
            # 設備資料間分頁（除非是最後一個）
            if record_index < len(screenshots_by_record) - 1:
                story.append(PageBreak())
        
        # 生成PDF
        doc.build(story)
        print(f"離群設備資料截圖PDF已生成：{pdf_path}")
        return pdf_path
        
    except ImportError:
        print("警告：缺少reportlab模組，無法生成PDF。請安裝：pip install reportlab")
        return None
    except Exception as e:
        print(f"生成PDF時發生錯誤：{str(e)}")
        return None


if __name__ == "__main__":
    # 測試用例
    test_devices = [
        {
            'filename': 'cleaned_SPS2022PA000072_20250528_04_20250529_04_data.csv',
            'device_sn': 'SPS2022PA000072', 
            'start_date': '20250528',
            'start_hour': 4,
            'end_date': '20250529',
            'end_hour': 4,
            'avg_time_diff': 2.00, 
            'match_rate': 0.20, 
            'precision': 1.00,
            'score': 65.5
        },
        {
            'filename': 'cleaned_SPS2022PA000090_20250530_06_20250531_06_data.csv',
            'device_sn': 'SPS2022PA000090', 
            'start_date': '20250530',
            'start_hour': 6,
            'end_date': '20250531',
            'end_hour': 6,
            'avg_time_diff': 1.00, 
            'match_rate': 0.17, 
            'precision': 0.06,
            'score': 45.2
        }
    ]
    
    log_dir = "_logs/pyqt-viewer"
    results_dir = "_logs/pyqt-viewer/batch_prediction_results"
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    os.makedirs(results_dir, exist_ok=True)
    
    pdf_path = generate_outlier_screenshots_pdf(test_devices, results_dir, timestamp, log_dir)
    if pdf_path:
        print(f"測試PDF生成成功：{pdf_path}")
    else:
        print("測試PDF生成失敗") 