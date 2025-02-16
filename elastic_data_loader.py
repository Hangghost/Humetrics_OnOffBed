from elasticsearch import Elasticsearch
from datetime import datetime, timedelta
import pandas as pd
import pytz
import logging
import os
from dotenv import load_dotenv
import argparse
import sys

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

    def fetch_data(self, device_id, start_time=None, end_time=None, limit=4000):
        """
        從 Elasticsearch 讀取指定設備的資料，使用 Scroll API 處理大量資料
        limit: 限制最大讀取筆數，預設4000筆
        """
        try:
            # 轉換日期格式為 ISO 格式
            if start_time:
                start_dt = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
                start_iso = start_dt.strftime('%Y-%m-%dT%H:%M:%S+08:00')
            if end_time:
                end_dt = datetime.strptime(end_time, '%Y-%m-%d %H:%M:%S')
                end_iso = end_dt.strftime('%Y-%m-%dT%H:%M:%S+08:00')

            print(f"查詢時間範圍：{start_iso} 到 {end_iso}")

            # 修正查詢語法
            query = {
                "query": {
                    "bool": {
                        "must": [
                            {"match": {"serial_id": device_id}},
                            {"range": {"created_at": {
                                "gte": start_iso,
                                "lte": end_iso
                            }}}
                        ]
                    }
                },
                "sort": [{"created_at": "asc"}],
                "size": 10
            }

            # 添加查詢結果的除錯資訊
            self.logger.info(f"執行查詢: {query}")
            
            # 修正 search API 的調用方式
            page = self.es.search(
                index="sensor_data",
                body=query,
                scroll='5m'
            )
            
            scroll_id = page['_scroll_id']
            hits = page['hits']['hits']
            
            # 用於儲存所有記錄
            all_records = []
            
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
                        'Channel_1_Raw': source.get('ch0'),
                        'Channel_2_Raw': source.get('ch1'),
                        'Channel_3_Raw': source.get('ch2'),
                        'Channel_4_Raw': source.get('ch3'),
                        'Channel_5_Raw': source.get('ch4'),
                        'Channel_6_Raw': source.get('ch5'),
                        'Angle': source.get('angle'),
                        'timestamp': source.get('timestamp')
                    })
                
                all_records.extend(records)
                
                # 顯示進度
                self.logger.info(f"已獲取 {len(all_records)}/{limit} 筆資料")
                
                # 如果已達到限制，跳出迴圈
                if len(all_records) >= limit:
                    break
                    
                # 獲取下一批資料
                page = self.es.scroll(
                    scroll_id=scroll_id,
                    scroll='5m'
                )
                hits = page['hits']['hits']

            # 清理 scroll
            self.es.clear_scroll(scroll_id=scroll_id)
            
            # 轉換為 DataFrame
            df = pd.DataFrame(all_records)
            
            # 除錯：印出 DataFrame 的欄位
            self.logger.info(f"DataFrame 欄位: {df.columns.tolist()}")
            
            # 確保 created_at 欄位存在且有值
            if 'created_at' in df.columns and not df['created_at'].isna().all():
                df.set_index('created_at', inplace=True)
            else:
                self.logger.error("找不到有效的 created_at 欄位")
                return pd.DataFrame()  # 返回空的 DataFrame
            
            self.logger.info(f"成功讀取總共 {len(all_records)} 筆資料")
            return df
            
        except Exception as e:
            self.logger.error(f"讀取資料時發生錯誤: {str(e)}")
            raise

    def test_data(self):
        """
        測試 Elasticsearch 連線狀態並取得範例資料
        """
        try:
            # 測試連線是否成功
            if self.es.ping():
                self.logger.info("成功連線到 Elasticsearch！")
                
                # 取得一筆範例資料
                query = {
                    "query": {
                        "match_all": {}
                    },
                    "size": 1
                }
                
                response = self.es.search(
                    index="sensor_data-*",
                    body=query
                )
                
                if response['hits']['hits']:
                    sample_doc = response['hits']['hits'][0]['_source']
                    print("\n範例資料結構：")
                    for field, value in sample_doc.items():
                        print(f"{field}: {value}")
                    print("\n索引名稱：", response['hits']['hits'][0]['_index'])
                else:
                    print("找不到範例資料")
                
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
        try:
            query = {
                "query": {
                    "match_all": {}
                },
                "aggs": {
                    "unique_serial_ids": {
                        "terms": {
                            "field": "serial_id.keyword",
                            "size": 10000
                        }
                    }
                },
                "size": 0
            }
            
            # 添加除錯資訊
            self.logger.info(f"執行查詢: {query}")
            
            response = self.es.search(
                index="sensor_data-*",
                body=query
            )
            
            # 添加除錯資訊
            self.logger.info(f"獲得回應: {response}")
            
            if 'aggregations' in response and 'unique_serial_ids' in response['aggregations']:
                buckets = response['aggregations']['unique_serial_ids']['buckets']
                serial_ids = [bucket['key'] for bucket in buckets]
                self.logger.info(f"找到 {len(serial_ids)} 個唯一設備 ID")
                return serial_ids
            else:
                self.logger.error("回應中沒有找到聚合結果")
                return []
            
        except Exception as e:
            self.logger.error(f"獲取 serial_id 時發生錯誤: {str(e)}")
            self.logger.error(f"錯誤詳情: {type(e).__name__}")
            return []

    def get_device_data_count(self, device_id):
        """
        查詢特定設備的資料總筆數
        """
        try:
            # 建立查詢條件
            query = {
                "query": {
                    "match": {
                        "serial_id": device_id
                    }
                }
            }

            # 使用正確的索引名稱
            response = self.es.count(
                index="sensor_data",
                body=query
            )

            count = response['count']
            print(f"設備 {device_id} 總共有 {count} 筆資料")
            
            # 添加除錯資訊
            self.logger.info(f"查詢條件: {query}")
            self.logger.info(f"使用的索引: sensor_data")
            
            return count

        except Exception as e:
            self.logger.error(f"查詢資料總數時發生錯誤: {str(e)}")
            raise

    def check_data_timerange(self):
        """檢查資料的時間範圍"""
        try:
            query = {
                "aggs": {
                    "min_date": { "min": { "field": "created_at" } },
                    "max_date": { "max": { "field": "created_at" } }
                },
                "size": 0
            }
            response = self.es.search(index="sensor_data-*", body=query)
            min_date = response['aggregations']['min_date']['value_as_string']
            max_date = response['aggregations']['max_date']['value_as_string']
            
            print(f"資料庫中最早的資料時間：{min_date}")
            print(f"資料庫中最新的資料時間：{max_date}")
            
        except Exception as e:
            self.logger.error(f"檢查時間範圍時發生錯誤: {str(e)}")

    def get_latest_data_time(self, device_id):
        """
        獲取指定設備的最新資料時間
        """
        try:
            query = {
                "query": {
                    "match": {"serial_id": device_id}
                },
                "sort": [
                    {"created_at": "desc"}
                ],
                "size": 1
            }
            
            response = self.es.search(
                index="sensor_data",
                body=query
            )
            
            if response['hits']['hits']:
                latest_data = response['hits']['hits'][0]['_source']
                latest_time = latest_data.get('created_at')
                print(f"\n設備 {device_id} 的最新資料時間：")
                print(f"ISO 格式：{latest_time}")
                local_time = datetime.strptime(latest_time, '%Y-%m-%dT%H:%M:%S%z').strftime('%Y-%m-%d %H:%M:%S')
                print(f"本地時間：{local_time}")
                return latest_time
            else:
                print(f"\n找不到設備 {device_id} 的資料")
                return None
                
        except Exception as e:
            self.logger.error(f"檢查最新資料時間時發生錯誤: {str(e)}")
            return None

def main():
    # 建立參數解析器
    parser = argparse.ArgumentParser(description='Elasticsearch 資料讀取工具')
    
    # 添加命令列參數
    parser.add_argument('--device_id', type=str, help='設備 ID')
    parser.add_argument('--start_time', type=str, help='開始時間 (YYYY-MM-DD HH:MM:SS)')
    parser.add_argument('--end_time', type=str, help='結束時間 (YYYY-MM-DD HH:MM:SS)')
    parser.add_argument('--limit', type=int, help='限制最大讀取筆數，預設4000筆')
    parser.add_argument('--check_latest', action='store_true', help='檢查設備最新資料時間')
    parser.add_argument('--check_total_count', action='store_true', help='檢查設備資料總數')
    
    # 解析參數
    args = parser.parse_args()
    
    # 設定預設時間範圍（如果沒有指定）
    if not args.start_time:
        # 設定今天的12:00
        end_time = datetime.now().replace(hour=12, minute=0, second=0, microsecond=0)
        start_time = end_time - timedelta(days=1)  # 預設查詢前一天的12:00
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

    # 測試連線是否正確
    connection_success = loader.test_data()
    if not connection_success:
        print("連線測試失敗")
        return
        
    try:
        if args.check_latest and args.device_id:
            loader.get_latest_data_time(args.device_id)
            sys.exit(0)
        # ============================================
        # 查詢特定設備的資料總共有幾筆
        if args.check_total_count and args.device_id:
            total_count = loader.get_device_data_count(args.device_id)
            sys.exit(0)
        
        # ============================================
        # 利用時間區間查詢資料
        data = loader.fetch_data(args.device_id, args.start_time, args.end_time, limit=args.limit)

        # 依照原始 timestamp 欄位排序並存成 CSV
        if not data.empty:
            data = data.sort_values(by='timestamp')
            
            # 將時間格式轉換為較短的格式
            start_short = datetime.strptime(args.start_time, '%Y-%m-%d %H:%M:%S').strftime('%Y%m%d_%H')
            end_short = datetime.strptime(args.end_time, '%Y-%m-%d %H:%M:%S').strftime('%Y%m%d_%H')
            
            # 使用較短的時間格式建立檔名
            filename = f'{args.device_id}_{start_short}_{end_short}.csv'
            data.to_csv(f'_data/elastic_data/{filename}', index=True)
            print(f"資料已成功存成CSV檔案: {filename}")
        else:
            print("未找到符合條件的資料")

        #============================================
        # # 先測試連線
        # connection_success = loader.get_all_test_data()
        # if not connection_success:
        #     print("連線測試失敗")
        #     return
        
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

    # # 基本查詢範例
    # python elastic_data_loader.py --device_id "SPS2021PA000336" --start_time "2025-02-01 00:00:00" --end_time "2025-02-05 00:00:00"

    # # 只查詢最近24小時的資料
    # python elastic_data_loader.py --device_id "SPS2021PA000336"

    # # 指定特定時間範圍的資料
    # python elastic_data_loader.py --device_id "SPS2021PA000336" --start_time "2025-02-01 08:00:00" --end_time "2025-02-02 18:00:00"

    # 先檢查資料時間範圍
    # python elastic_data_loader.py --check_timerange

    # 再用正確的時間範圍查詢
    # python elastic_data_loader.py --device_id "2024HB000052" --start_time "2024-02-03 12:00:00" --end_time "2024-02-08 12:00:00"