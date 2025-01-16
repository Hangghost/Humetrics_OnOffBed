from elasticsearch import Elasticsearch
from datetime import datetime, timedelta
import pandas as pd
import pytz
import logging
import os
from dotenv import load_dotenv
import argparse

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

    def fetch_data(self, device_id, start_time=None, end_time=None):
        """
        從 Elasticsearch 讀取指定設備的資料
        """
        try:
            # 轉換日期格式為 ISO 格式
            if start_time:
                start_dt = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
                start_iso = start_dt.strftime('%Y-%m-%dT%H:%M:%S+08:00')
            if end_time:
                end_dt = datetime.strptime(end_time, '%Y-%m-%d %H:%M:%S')
                end_iso = end_dt.strftime('%Y-%m-%dT%H:%M:%S+08:00')

            print(f"查詢時間範圍：{start_iso} 到 {end_iso}")  # 除錯用

            # 建立查詢條件
            query = {
                "size": 10000,
                "sort": [{"created_at": "asc"}],
                "query": {
                    "bool": {
                        "must": [
                            {
                                "term": {
                                    "serial_id": device_id
                                }
                            },
                            {
                                "range": {
                                    "created_at": {
                                        "gte": start_iso,
                                        "lte": end_iso
                                    }
                                }
                            }
                        ]
                    }
                }
            }

            # 執行查詢
            response = self.es.search(
                index="sensor_data-*",  # 使用通配符匹配所有 sensor_data 索引
                body=query
            )

            # print("查詢結果：", response)

            # 檢查是否有資料
            if response['hits']['total']['value'] == 0:
                self.logger.warning(f"未找到符合條件的資料：device_id={device_id}, start_time={start_time}, end_time={end_time}")
                return pd.DataFrame()

            # 除錯：印出查詢到的資料數量
            total_hits = response['hits']['total']['value']
            self.logger.info(f"找到 {total_hits} 筆資料")
            
            # 除錯：印出第一筆回應資料
            if response['hits']['hits']:
                self.logger.info(f"第一筆資料結構: {response['hits']['hits'][0]['_source']}")
            
            records = []
            for hit in response['hits']['hits']:
                source = hit['_source']
                records.append({
                    'created_at': source.get('created_at'),
                    'serial_id': source.get('serial_id'),
                    'Channel_1_Raw': source.get('ch0'),
                    'Channel_2_Raw': source.get('ch1'),
                    'Channel_3_Raw': source.get('ch2'),
                    'Channel_4_Raw': source.get('ch3'),
                    'Channel_5_Raw': source.get('ch4'),
                    'Channel_6_Raw': source.get('ch5'),
                    'Angle': source.get('angle'),
                    'timestamp': source.get('timestamp')
                })
            
            # 除錯：印出 DataFrame 的欄位
            df = pd.DataFrame(records)
            self.logger.info(f"DataFrame 欄位: {df.columns.tolist()}")
            
            # 確保 created_at 欄位存在且有值
            if 'created_at' in df.columns and not df['created_at'].isna().all():
                df.set_index('created_at', inplace=True)
            else:
                self.logger.error("找不到有效的 created_at 欄位")
                return pd.DataFrame()  # 返回空的 DataFrame
            
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
        

    def get_serial_id(self):
        """
        獲取所有設備的 serial_id
        """
        response = self.es.search(index="sensor_data", size=10000)
        return [hit['_source']['serial_id'] for hit in response['hits']['hits']]

    def get_device_data_count(self, device_id):
        """
        查詢特定設備的資料總筆數
        """
        try:
            # 建立查詢條件
            query = {
                "query": {
                    "term": {
                        "serial_id": device_id
                    }
                }
            }

            # 執行查詢
            response = self.es.count(
                index="sensor_data-*",
                body=query
            )

            count = response['count']
            print(f"設備 {device_id} 總共有 {count} 筆資料")
            return count

        except Exception as e:
            self.logger.error(f"查詢資料總數時發生錯誤: {str(e)}")
            raise

def main():
    # 建立參數解析器
    parser = argparse.ArgumentParser(description='Elasticsearch 資料讀取工具')
    
    # 添加命令列參數
    parser.add_argument('--device_id', type=str, help='設備 ID')
    parser.add_argument('--start_time', type=str, help='開始時間 (YYYY-MM-DD HH:MM:SS)')
    parser.add_argument('--end_time', type=str, help='結束時間 (YYYY-MM-DD HH:MM:SS)')
    
    # 解析參數
    args = parser.parse_args()
    
    # 設定預設時間範圍（如果沒有指定）
    if not args.start_time:
        end_time = datetime.now()
        start_time = end_time - timedelta(days=7)  # 預設查詢最近7天
        args.start_time = start_time.strftime('%Y-%m-%d %H:%M:%S')
        args.end_time = end_time.strftime('%Y-%m-%d %H:%M:%S')
    
    # 從環境變數讀取設定
    load_dotenv()
    
    # Elasticsearch 連線設定
    es_config = {
        'hosts': os.getenv('ELASTICSEARCH_HOST', 'http://192.168.1.68:9200'),
        'api_key':os.getenv('ELASTICSEARCH_API_KEY'),
        'verify_certs': False,
    }
    
    # 檢查必要的參數
    if not es_config['api_key']:
        print("錯誤：未提供 API Key")
        return
    
    # 建立 loader
    loader = ElasticDataLoader(**es_config)
    
    try:
        # 利用時間區間查詢資料
        data = loader.fetch_data(args.device_id, args.start_time, args.end_time)
        # print(data)

        # 修改 timestamp 的解析方式，加入時區處理
        if not data.empty and 'timestamp' in data.columns:
            # 先解析成 datetime
            data['timestamp'] = pd.to_datetime(data['timestamp'], format='%Y%m%d%H%M%S')
            # 設定為 UTC 時間
            data['timestamp'] = data['timestamp'].dt.tz_localize('UTC')
            # 轉換為亞洲/台北時區
            data['timestamp'] = data['timestamp'].dt.tz_convert('Asia/Taipei')

        # 依照 'timestamp' 欄位排序
        if not data.empty:
            data = data.sort_values(by='timestamp')
            
            # 存成CSV，使用 created_at 作為索引
            data.to_csv(f'{args.device_id}.csv', index=True)
            print(f"資料已成功存成CSV檔案: {args.device_id}.csv")
            
            # 顯示資料時間範圍
            print(f"資料時間範圍：")
            print(f"開始時間：{data['timestamp'].min()}")
            print(f"結束時間：{data['timestamp'].max()}")
        else:
            print("未找到符合條件的資料")

        # ============================================
        # # 先測試連線
        # connection_success = loader.get_all_test_data()
        # if not connection_success:
        #     print("連線測試失敗")
        #     return
        
        # ============================================

        # # 查詢特定設備的資料總共有幾筆
        # if args.device_id:
        #     total_count = loader.get_device_data_count(args.device_id)
        #     print(f"\n設備 {args.device_id} 的資料總數：{total_count} 筆")

        # ============================================
        # # 利用時間區間查詢資料
        # data = loader.fetch_data(args.device_id, args.start_time, args.end_time)

        # # 依照 'timestamp' 欄位排序
        # data = data.sort_values(by='timestamp')

        # # 存成CSV
        # data.to_csv(f'{args.device_id}.csv', index=True)
        # print(f"資料已成功存成CSV檔案: {args.device_id}.csv")

        # ============================================
        # # 獲取所有設備的 serial_id
        # serial_ids = loader.get_serial_id()

        # # 去除重複的 serial_id 並排序   
        # serial_ids = sorted(list(set(serial_ids)))
        
        # print(f"所有設備的 serial_id: {serial_ids}")
        # # 將 serial_id 存成CSV
        # pd.DataFrame(serial_ids, columns=['serial_id']).to_csv('serial_ids.csv', index=False)
        # print(f"serial_id 已成功存成CSV檔案: serial_ids.csv")

        # ============================================
        # # 查詢資料
        # df = loader.fetch_data(
        #     start_time=args.start_time,
        #     end_time=args.end_time,
        #     device_id=args.device_id
        # )
        
        # # 顯示資料
        # if not df.empty:
        #     print(f"\n查詢到 {len(df)} 筆資料")
        #     print("\n資料預覽：")
        #     print(df.head())
            
        #     # 顯示感測器數據的基本統計資訊
        #     print("\n感測器數據統計資訊：")
        #     sensor_columns = ['Channel_1_Raw', 'Channel_2_Raw', 'Channel_3_Raw', 
        #                     'Channel_4_Raw', 'Channel_5_Raw', 'Channel_6_Raw', 'Angle']
        #     print(df[sensor_columns].describe())
            
        #     # 顯示時間範圍
        #     print("\n資料時間範圍：")
        #     print(f"開始時間：{df.index.min()}")
        #     print(f"結束時間：{df.index.max()}")
            
        # else:
        #     print("未找到符合條件的資料")

        # ============================================
    except Exception as e:
        print(f"執行時發生錯誤: {str(e)}")

if __name__ == "__main__":
    main()