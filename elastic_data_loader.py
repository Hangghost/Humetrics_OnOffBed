from elasticsearch import Elasticsearch
from datetime import datetime
import pandas as pd
import pytz
import logging

class ElasticDataLoader:
    def __init__(self, hosts, api_key=None, verify_certs=False):
        """
        初始化 Elasticsearch 連線
        """
        # 基本連線配置
        es_config = {
            "hosts": hosts,
            "request_timeout": 30,  # 使用 request_timeout 替代 timeout
            "retry_on_timeout": True,
            "max_retries": 3,
            "ssl_show_warn": False  # 關閉 SSL 警告
        }
        
        # 如果提供了 API key，使用 API key 認證
        if api_key:
            es_config["api_key"] = api_key
        
        # 如果使用 HTTPS
        if hosts.startswith("https"):
            es_config["verify_certs"] = verify_certs
            if verify_certs:
                es_config["ca_certs"] = "/path/to/ca.crt"  # 請確保路徑正確
        
        self.es = Elasticsearch(**es_config)
        
        # 設定日誌
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def fetch_data(self, start_time, end_time, device_id=None):
        """
        從 Elasticsearch 讀取指定時間範圍的資料
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
                index="sensor_data",
                query=query,
                size=10000,
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

    def get_all_test_data(self):
        """
        測試 Elasticsearch 連線狀態
        """
        try:
            # 測試連線是否成功
            if self.es.ping():
                self.logger.info("成功連線到 Elasticsearch！")
                
                # 獲取叢集資訊
                cluster_info = self.es.info()
                self.logger.info(f"叢集名稱: {cluster_info['cluster_name']}")
                self.logger.info(f"叢集版本: {cluster_info['version']['number']}")
                
                # 獲取索引資訊
                indices = self.es.indices.get(index="sensor_data")
                self.logger.info(f"sensor_data 索引存在: {bool(indices)}")
                
                return True
            else:
                self.logger.error("無法連線到 Elasticsearch")
                return False
                
        except Exception as e:
            self.logger.error(f"連線測試時發生錯誤: {str(e)}")
            return False

def main():
    # Elasticsearch 連線設定
    es_config = {
        'hosts': 'https://192.168.1.68:9200',  # 使用 HTTPS
        'api_key': 'cTItWGFKUUI2MWZPZUVTcFdWOUw6dkg5Nm80WDRTWHFjeWREWTJQLXpZQQ==',
        'verify_certs': False,  # 因為使用 -k 參數，所以這裡設為 False
    }
    
    # 建立 loader
    loader = ElasticDataLoader(**es_config)

    try:
        # 測試連線
        connection_success = loader.get_all_test_data()
        if not connection_success:
            print("連線測試失敗")
    except Exception as e:
        print(f"執行時發生錯誤: {str(e)}")

if __name__ == "__main__":
    main() 