import pandas as pd
import json
from datetime import datetime
import pytz

def load_csv_data(csv_path):
    # 讀取 CSV 檔案
    df = pd.read_csv(csv_path, parse_dates=['Timestamp'])
    # 將 Timestamp 設為索引
    df.set_index('Timestamp', inplace=True)
    return df

def load_json_data(json_path):
    # 讀取 JSON 檔案
    with open(json_path, 'r') as f:
        json_content = json.load(f)
    
    # 取得 data 陣列
    data = json_content['data']
    
    # 轉換成 DataFrame
    records = []
    for item in data:
        # 轉換時間戳記 (從字串轉換為datetime)
        timestamp = datetime.strptime(item['created_at'], '%Y-%m-%d %H:%M:%S')
        timestamp = pytz.timezone('Asia/Taipei').localize(timestamp)
        
        record = {
            'Timestamp': timestamp,
            'Channel_1_Raw': item['ch0'],  # 注意：通道編號從0開始
            'Channel_2_Raw': item['ch1'],
            'Channel_3_Raw': item['ch2'],
            'Channel_4_Raw': item['ch3'],
            'Channel_5_Raw': item['ch4'],
            'Channel_6_Raw': item['ch5']
        }
        records.append(record)
    
    df = pd.DataFrame(records)
    df.set_index('Timestamp', inplace=True)
    return df

def compare_data(csv_path, json_path):
    # 載入兩個資料來源
    csv_df = load_csv_data(csv_path)
    json_df = load_json_data(json_path)

    print(f"CSV 長度: {len(csv_df)}")
    print(f"JSON 長度: {len(json_df)}")
    
    # 只比較 Raw 數值
    raw_columns = [col for col in csv_df.columns if 'Raw' in col]
    csv_df = csv_df[raw_columns]
    
    # 找出共同的時間點
    common_times = csv_df.index.intersection(json_df.index)
    
    print(f"共同時間點數量: {len(common_times)}")
    
    # 比較每個時間點的數據
    differences = []
    for timestamp in common_times:
        csv_row = csv_df.loc[timestamp]
        json_row = json_df.loc[timestamp]
        
        # 比較每個通道
        for col in raw_columns:
            # 安全地取得數值
            try:
                if isinstance(csv_row[col], pd.Series):
                    csv_value = csv_row[col].iloc[0]
                else:
                    csv_value = csv_row[col]
                
                if isinstance(json_row[col], pd.Series):
                    json_value = json_row[col].iloc[0]
                else:
                    json_value = json_row[col]
                
                if csv_value != json_value:
                    differences.append({
                        'Timestamp': timestamp,
                        'Channel': col,
                        'CSV_Value': csv_value,
                        'JSON_Value': json_value,
                        'Difference': abs(csv_value - json_value)
                    })
            except Exception as e:
                print(f"處理時間點 {timestamp} 的 {col} 時發生錯誤: {str(e)}")
                continue
    
    # 輸出差異
    if differences:
        print("\n發現數據差異:")
        diff_df = pd.DataFrame(differences)
        print(diff_df)
        # 存檔
        diff_df.to_csv('differences.csv', index=False)
        
        # 計算差異統計
        print("\n差異統計:")
        print(f"總差異筆數: {len(differences)}")
        print("\n各通道差異數量:")
        print(diff_df['Channel'].value_counts())
        print("\n差異值統計:")
        print(diff_df['Difference'].describe())
    else:
        print("\n所有數據完全相同!")

if __name__ == "__main__":
    csv_path = "_data/SPS2021PA000454_20250108_04_20250109_04_data.csv"
    json_path = "_data/SPS2021PA000454_2025-01-08 12:00:00_2025-01-09 12:00:00.json"
    
    compare_data(csv_path, json_path) 