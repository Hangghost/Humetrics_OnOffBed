from elasticsearch import Elasticsearch
from datetime import datetime
import pandas as pd
import pytz
import logging

class ElasticDataLoader:
    def __init__(self, hosts, username, password, verify_certs=False):
        """
        初始化 Elasticsearch 連線
        """
        # 建立 Elasticsearch 客戶端，移除 SSL 相關設定
        self.es = Elasticsearch(
            hosts=hosts,
            basic_auth=(username, password)
        )
        
        # 設定日誌
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def fetch_data(self, index_name, start_time, end_time, device_id=None):
        """
        從 Elasticsearch 讀取指定時間範圍的資料
        
        Parameters:
        -----------
        index_name : str
            索引名稱
        start_time : str or datetime
            開始時間 (格式: "YYYY-MM-DD HH:MM:SS")
        end_time : str or datetime
            結束時間 (格式: "YYYY-MM-DD HH:MM:SS")
        device_id : str, optional
            設備 ID
            
        Returns:
        --------
        pd.DataFrame
            包含所有通道原始數據的 DataFrame
        """
        try:
            # 確保時間格式正確
            if isinstance(start_time, str):
                start_time = datetime.fromisoformat(start_time).strftime('%Y-%m-%dT%H:%M:%S+08:00')
            if isinstance(end_time, str):
                end_time = datetime.fromisoformat(end_time).strftime('%Y-%m-%dT%H:%M:%S+08:00')

            # 建立查詢條件
            query = {
                "bool": {
                    "must": [
                        {
                            "range": {
                                "created_at": {
                                    "gte": start_time,
                                    "lte": end_time
                                }
                            }
                        }
                    ]
                }
            }
            
            # 如果有指定設備 ID，加入查詢條件
            if device_id:
                query["bool"]["must"].append({
                    "term": {
                        "serial_id": device_id
                    }
                })
            
            # 執行查詢
            response = self.es.search(
                index=index_name,
                query=query,
                size=10000,  # 可以根據需求調整
                sort=[{"created_at": "asc"}]
            )
            
            # 解析查詢結果
            records = []
            for hit in response['hits']['hits']:
                source = hit['_source']
                
                # 轉換時間戳記
                timestamp = datetime.fromisoformat(source['created_at'])
                
                # 建立記錄
                record = {
                    'Timestamp': timestamp,
                    'Serial_ID': source['serial_id'],
                    'Channel_1_Raw': source['ch0'],
                    'Channel_2_Raw': source['ch1'],
                    'Channel_3_Raw': source['ch2'],
                    'Channel_4_Raw': source['ch3'],
                    'Channel_5_Raw': source['ch4'],
                    'Channel_6_Raw': source['ch5'],
                    'Angle': source['angle'],
                    'Raw_Timestamp': source['timestamp']
                }
                records.append(record)
            
            # 轉換成 DataFrame
            df = pd.DataFrame(records)
            df.set_index('Timestamp', inplace=True)
            
            self.logger.info(f"成功讀取 {len(records)} 筆資料")
            return df
            
        except Exception as e:
            self.logger.error(f"讀取資料時發生錯誤: {str(e)}")
            raise

def main():
    # Elasticsearch 連線設定
    es_config = {
        'hosts': 'http://192.168.1.68:5601',  # 使用 http 協議
        'username': 'humetrics',
        'password': 'HM66050660'
    }
    
    # 建立 loader
    loader = ElasticDataLoader(**es_config)
    
    try:
        # 先列出所有可用的索引
        indices = loader.es.indices.get_alias().keys()
        print(f"可用的索引列表：{list(indices)}")
        
        # 再讀取資料
        df = loader.fetch_data(
            index_name='sensor_data-000001',
            start_time='2025-01-01T00:00:00+08:00',
            end_time='2025-01-01T23:59:59+08:00',
            device_id='SPS2021PA000345'
        )
        
        # 將資料存成 JSON 檔案
        output_file = '_data/elastic_data.json'
        df.reset_index().to_json(output_file, orient='records', date_format='iso')
        print(f"資料已儲存至 {output_file}")
        
    except Exception as e:
        print(f"執行時發生錯誤: {str(e)}")

if __name__ == "__main__":
    main() 