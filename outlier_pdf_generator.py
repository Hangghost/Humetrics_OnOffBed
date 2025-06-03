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
    """收集離群設備的截圖文件"""
    screenshot_dir = os.path.join(log_dir, 'batch_screenshots')
    found_screenshots = []
    
    if not os.path.exists(screenshot_dir):
        print(f"截圖目錄不存在: {screenshot_dir}")
        return found_screenshots
    
    # 列出所有截圖文件
    screenshot_files = [f for f in os.listdir(screenshot_dir) if f.endswith('.png')]
    
    for device_info in common_outlier_devices:
        device_sn = device_info['device_sn']
        device_screenshots = []
        
        # 尋找該設備的截圖（可能有多個時間段的）
        for screenshot_file in screenshot_files:
            if screenshot_file.startswith(device_sn + '_'):
                screenshot_path = os.path.join(screenshot_dir, screenshot_file)
                if os.path.exists(screenshot_path):
                    device_screenshots.append({
                        'device_sn': device_sn,
                        'file_path': screenshot_path,
                        'file_name': screenshot_file,
                        'device_info': device_info
                    })
        
        if device_screenshots:
            # 按文件名排序（這樣時間順序會是正確的）
            device_screenshots.sort(key=lambda x: x['file_name'])
            found_screenshots.extend(device_screenshots)
        else:
            print(f"未找到設備 {device_sn} 的截圖文件")
    
    return found_screenshots

def generate_outlier_screenshots_pdf(common_outlier_devices, results_dir, timestamp, log_dir):
    """生成離群設備截圖的PDF報告"""
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
            print("未找到任何離群設備的截圖文件")
            return None
        
        # 生成PDF文件路徑
        pdf_filename = f"outlier_devices_screenshots_{timestamp}.pdf"
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
        title = Paragraph("離群設備截圖報告", title_style)
        story.append(title)
        story.append(Spacer(1, 20))
        
        # 添加摘要資訊
        summary_text = f"""
        <b>報告生成時間：</b>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br/>
        <b>離群設備數量：</b>{len(common_outlier_devices)}<br/>
        <b>找到截圖數量：</b>{len(screenshots)}<br/>
        <b>分析指標：</b>平均時差(avg_time_diff)、配對率(match_rate)、精確率(precision)
        """
        summary = Paragraph(summary_text, normal_style)
        story.append(summary)
        story.append(Spacer(1, 20))
        
        # 創建設備統計表格
        table_data = [['設備序號', '平均時差(秒)', '配對率', '精確率', '截圖數量']]
        screenshot_count_by_device = {}
        for screenshot in screenshots:
            device_sn = screenshot['device_sn']
            if device_sn not in screenshot_count_by_device:
                screenshot_count_by_device[device_sn] = 0
            screenshot_count_by_device[device_sn] += 1
        
        for device_info in common_outlier_devices:
            device_sn = device_info['device_sn']
            screenshot_count = screenshot_count_by_device.get(device_sn, 0)
            table_data.append([
                device_sn,
                f"{device_info['avg_time_diff']:.2f}",
                f"{device_info['match_rate']:.2f}",
                f"{device_info['precision']:.2f}",
                str(screenshot_count)
            ])
        
        table = Table(table_data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, -1), chinese_font_name),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(table)
        story.append(PageBreak())
        
        # 按設備分組截圖
        screenshots_by_device = {}
        for screenshot in screenshots:
            device_sn = screenshot['device_sn']
            if device_sn not in screenshots_by_device:
                screenshots_by_device[device_sn] = []
            screenshots_by_device[device_sn].append(screenshot)
        
        # 為每個設備添加截圖頁面
        for device_index, (device_sn, device_screenshots) in enumerate(screenshots_by_device.items()):
            # 獲取設備資訊
            device_info = next((d for d in common_outlier_devices if d['device_sn'] == device_sn), None)
            
            # 設備標題
            device_title_text = f"設備：{device_sn}"
            if device_info:
                device_title_text += f"<br/>時差：{device_info['avg_time_diff']:.2f}秒 | 配對率：{device_info['match_rate']:.2f} | 精確率：{device_info['precision']:.2f}"
            
            device_title = Paragraph(device_title_text, device_title_style)
            story.append(device_title)
            story.append(Spacer(1, 10))
            
            # 添加截圖
            for i, screenshot in enumerate(device_screenshots):
                try:
                    # 解析文件名中的時間資訊
                    file_name = screenshot['file_name']
                    parts = file_name.split('_')
                    if len(parts) >= 6:
                        time_info = f"時間：{parts[1]} {parts[2][:2]}:00 - {parts[3]} {parts[4][:2]}:00"
                    else:
                        time_info = f"截圖文件：{file_name}"
                    
                    time_paragraph = Paragraph(time_info, normal_style)
                    story.append(time_paragraph)
                    story.append(Spacer(1, 10))
                    
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
                    if i < len(device_screenshots) - 1:
                        story.append(PageBreak())
                    
                except Exception as e:
                    error_text = f"無法載入截圖：{screenshot['file_name']}，錯誤：{str(e)}"
                    error_paragraph = Paragraph(error_text, normal_style)
                    story.append(error_paragraph)
            
            # 設備間分頁（除非是最後一個設備）
            if device_index < len(screenshots_by_device) - 1:
                story.append(PageBreak())
        
        # 生成PDF
        doc.build(story)
        print(f"離群設備截圖PDF已生成：{pdf_path}")
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
        {'device_sn': 'SPS2022PA000072', 'avg_time_diff': 2.00, 'match_rate': 0.20, 'precision': 1.00},
        {'device_sn': 'SPS2022PA000090', 'avg_time_diff': 1.00, 'match_rate': 0.17, 'precision': 0.06}
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