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
            "request_timeout": 10,  # 將逾時時間從 30 秒降低到 10 秒
            "retry_on_timeout": True,
            "max_retries": 2,  # 減少重試次數
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
        
        try:
            self.es = Elasticsearch(**es_config)
            
            # 設定日誌
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger(__name__)
            
        except Exception as e:
            print(f"連接 Elasticsearch 時發生錯誤: {str(e)}")
            raise

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
            query_sensor_data = {
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
                "size": 1000
            }

            query_notify_data = {
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
                "size": 1000
            }

            # 添加查詢結果的除錯資訊
            self.logger.info(f"執行查詢: {query_sensor_data}")
            
            page = self.es.search(
                index="sensor_data",
                body=query_sensor_data,
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
            
            # 查詢 notify 資料
            self.logger.info(f"執行 notify 查詢: {query_notify_data}")
            
            notify_page = self.es.search(
                index="notify-*",
                body=query_notify_data
            )
            
            notify_hits = notify_page['hits']['hits']
            notify_dict = {}  # 使用字典來存儲 notify 資料，以 timestamp 為鍵
            
            for hit in notify_hits:
                source = hit['_source']
                timestamp = source.get('timestamp')
                if timestamp:
                    notify_dict[timestamp] = source.get('statusType')
            
            self.logger.info(f"獲取到 {len(notify_dict)} 筆 notify 資料")
            
            # 轉換為 DataFrame
            df = pd.DataFrame(all_records)
            
            # 除錯：印出 DataFrame 的欄位
            self.logger.info(f"DataFrame 欄位: {df.columns.tolist()}")
            
            # 將 notify 的 statusType 併入 sensor data 的 DataFrame
            if 'timestamp' in df.columns:
                # 創建新的 statusType 欄位
                df['statusType'] = df['timestamp'].map(notify_dict)
                self.logger.info(f"已將 {sum(df['statusType'].notna())} 筆 notify 資料併入 sensor data")
            
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
                    # index="notify-*",
                    body=query
                )

                print(f"所有資料: {response}")
                
                if response['hits']['hits']:
                    sample_doc = response['hits']['hits'][0]['_source']
                    print("\n索引名稱：", response['hits']['hits'][0]['_index'])
                    print("\n範例資料結構：")
                    for field, value in sample_doc.items():
                        print(f"{field}: {value}")
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
                "size": 0,
                "aggs": {
                    "unique_serial_ids": {
                        "terms": {
                            "field": "serial_id.keyword",
                            "size": 10000
                        }
                    }
                }
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
        
    def count_serial_id_exists(self):
        """
        計算有 serial_id 欄位的文件數量並嘗試獲取樣本
        返回：文件數量和一個樣本文件（如果存在）
        """
        try:
            # 查詢有 serial_id 欄位的文件數量
            count_query = {
                "query": {
                    "exists": {
                        "field": "serial_id"
                    }
                }
            }
            count_response = self.es.count(
                index="sensor_data-*",
                body=count_query
            )
            count = count_response['count']
            
            # 查詢一個樣本文件以檢查 serial_id 欄位
            sample_query = {
                "query": {
                    "exists": {
                        "field": "serial_id"
                    }
                },
                "size": 1
            }
            sample_response = self.es.search(
                index="sensor_data-*",
                body=sample_query
            )
            
            # 提取樣本數據
            sample_doc = None
            index_name = None
            if sample_response['hits']['hits']:
                sample_doc = sample_response['hits']['hits'][0]['_source']
                index_name = sample_response['hits']['hits'][0]['_index']
            
            print(f"有 serial_id 欄位的文件數量：{count}")
            if sample_doc:
                print(f"樣本文件來自索引：{index_name}")
                print(f"樣本文件 serial_id：{sample_doc.get('serial_id')}")
                print(f"樣本文件結構：")
                for field, value in sample_doc.items():
                    print(f"  {field}: {value}")
            
            # 嘗試用不同方式獲取所有 serial_id
            direct_query = {
                "size": 0,
                "aggs": {
                    "unique_serial_ids": {
                        "terms": {
                            "field": "serial_id",
                            "size": 10000
                        }
                    }
                }
            }
            
            try:
                self.logger.info("嘗試直接用 serial_id 欄位聚合...")
                direct_response = self.es.search(
                    index="sensor_data-*",
                    body=direct_query
                )
                if 'aggregations' in direct_response:
                    buckets = direct_response['aggregations']['unique_serial_ids']['buckets']
                    serial_ids = [bucket['key'] for bucket in buckets]
                    print(f"直接用 serial_id 欄位找到 {len(serial_ids)} 個唯一設備 ID")
            except Exception as e:
                self.logger.info(f"直接用 serial_id 欄位聚合失敗：{str(e)}")
            
            keyword_query = {
                "size": 0,
                "aggs": {
                    "unique_serial_ids": {
                        "terms": {
                            "field": "serial_id.keyword",
                            "size": 10000
                        }
                    }
                }
            }
            
            try:
                self.logger.info("嘗試用 serial_id.keyword 欄位聚合...")
                keyword_response = self.es.search(
                    index="sensor_data-*",
                    body=keyword_query
                )
                if 'aggregations' in keyword_response:
                    buckets = keyword_response['aggregations']['unique_serial_ids']['buckets']
                    serial_ids = [bucket['key'] for bucket in buckets]
                    print(f"用 serial_id.keyword 欄位找到 {len(serial_ids)} 個唯一設備 ID")
                    if len(serial_ids) > 0:
                        print(f"前5個設備 ID: {serial_ids[:5]}")
                        return count, serial_ids
            except Exception as e:
                self.logger.info(f"用 serial_id.keyword 欄位聚合失敗：{str(e)}")
            
            script_query = {
                "size": 0,
                "aggs": {
                    "unique_serial_ids": {
                        "terms": {
                            "script": {
                                "source": "doc['serial_id'].value"
                            },
                            "size": 10000
                        }
                    }
                }
            }
            
            try:
                self.logger.info("嘗試用 script 方式聚合...")
                script_response = self.es.search(
                    index="sensor_data-*", 
                    body=script_query
                )
                if 'aggregations' in script_response:
                    buckets = script_response['aggregations']['unique_serial_ids']['buckets']
                    serial_ids = [bucket['key'] for bucket in buckets]
                    print(f"用 script 方式找到 {len(serial_ids)} 個唯一設備 ID")
                    if len(serial_ids) > 0:
                        print(f"前5個設備 ID: {serial_ids[:5]}")
                        return count, serial_ids
            except Exception as e:
                self.logger.info(f"用 script 方式聚合失敗：{str(e)}")
                
            return count, []
        except Exception as e:
            self.logger.error(f"計算有 serial_id 欄位的文件數量時發生錯誤: {str(e)}")
            return 0, []

    def get_serial_id_paged(self, size=1000, max_pages=10):
        """
        透過分頁查詢來獲取所有設備的 serial_id
        由於無法使用聚合，我們改用簡單的分頁查詢然後手動提取唯一 ID
        """
        try:
            all_serial_ids = set()  # 使用集合來儲存唯一的 serial_id
            
            # 第一次查詢
            self.logger.info(f"開始分頁查詢 serial_id，每頁 {size} 筆...")
            
            for page in range(max_pages):
                from_val = page * size
                
                query = {
                    "query": {
                        "match_all": {}
                    },
                    "_source": ["serial_id"],  # 只獲取 serial_id 欄位
                    "from": from_val,
                    "size": size
                }
                
                self.logger.info(f"查詢第 {page+1} 頁，從 {from_val} 開始...")
                response = self.es.search(
                    index="sensor_data-*",
                    body=query
                )
                
                # 只在第一頁顯示總數
                if page == 0:
                    total_hits = response['hits']['total']['value']
                    if response['hits']['total']['relation'] == 'gte':
                        self.logger.info(f"總文件數超過 {total_hits}，實際可能更多")
                    else:
                        self.logger.info(f"總共有 {total_hits} 份文件")
                
                # 處理結果
                hits = response['hits']['hits']
                if not hits:
                    self.logger.info(f"第 {page+1} 頁沒有更多數據，結束查詢")
                    break
                    
                new_ids_found = 0
                for hit in hits:
                    if 'serial_id' in hit['_source']:
                        serial_id = hit['_source']['serial_id']
                        if serial_id not in all_serial_ids:
                            new_ids_found += 1
                            all_serial_ids.add(serial_id)
                
                self.logger.info(f"第 {page+1} 頁找到 {new_ids_found} 個新設備 ID，累計 {len(all_serial_ids)} 個")
                
                # 如果這頁沒有新 ID，考慮提前結束
                if new_ids_found == 0 and page >= 2:  # 至少查詢3頁
                    self.logger.info("連續沒有發現新 ID，結束查詢")
                    break
            
            serial_ids = list(all_serial_ids)
            self.logger.info(f"總共找到 {len(serial_ids)} 個唯一設備 ID")
            
            return serial_ids
        
        except Exception as e:
            self.logger.error(f"分頁查詢 serial_id 時發生錯誤: {str(e)}")
            return []

    def find_devices_with_bad_sensor_readings(self, hours=1):
        """
        尋找過去N小時內有異常感測器讀數的設備
        異常定義: 任何感測器通道(ch0-ch5)數值低於 -1000000
        
        參數:
            hours: 要查詢的過去小時數，預設為1小時
        
        返回:
            異常設備ID列表與各設備的異常數量
        """
        try:
            # 準備查詢體
            query = {
                "size": 0,
                "runtime_mappings": {
                    "anyBad": {
                        "type": "boolean",
                        "script": {
                            "source": """
                                emit(
                                    doc['ch0'].value <= -1000000 ||
                                    doc['ch1'].value <= -1000000 ||
                                    doc['ch2'].value <= -1000000 ||
                                    doc['ch3'].value <= -1000000 ||
                                    doc['ch4'].value <= -1000000 ||
                                    doc['ch5'].value <= -1000000
                                );
                            """
                        }
                    },
                    "serial_kw": {
                        "type": "keyword",
                        "script": {
                            "source": """
                                if (params['_source'].containsKey('serial_id')) {
                                    emit(params['_source']['serial_id']);
                                }
                            """
                        }
                    }
                },
                "query": {
                    "bool": {
                        "filter": [
                            { "range": { "created_at": { "gte": f"now-{hours}h" } } },
                            { "term":  { "anyBad": True } }
                        ]
                    }
                },
                "aggs": {
                    "by_serial": {
                        "terms": { 
                            "field": "serial_kw", 
                            "size": 10000 
                        }
                    }
                }
            }
            
            # 執行查詢
            self.logger.info(f"查詢過去{hours}小時內有異常感測器讀數的設備...")
            response = self.es.search(
                index="sensor_data-*",
                body=query
            )
            
            # 處理結果
            bad_devices = []
            if 'aggregations' in response and 'by_serial' in response['aggregations']:
                buckets = response['aggregations']['by_serial']['buckets']
                for bucket in buckets:
                    bad_devices.append({
                        'serial_id': bucket['key'],
                        'abnormal_count': bucket['doc_count']
                    })
                
                print(f"找到 {len(bad_devices)} 個有異常感測器讀數的設備")
                
                # 如果有異常設備，顯示前5個
                if bad_devices:
                    print("\n異常設備列表 (前5個):")
                    for i, device in enumerate(bad_devices[:5]):
                        print(f"{i+1}. {device['serial_id']} - 異常記錄數: {device['abnormal_count']}")
                    
                    # 存儲為CSV
                    today = datetime.now().strftime('%Y%m%d_%H%M')
                    filename = f"abnormal_devices_{today}.csv"
                    pd.DataFrame(bad_devices).to_csv(filename, index=False)
                    print(f"\n異常設備列表已儲存為: {filename}")
            
            return bad_devices
                
        except Exception as e:
            self.logger.error(f"查詢異常設備時發生錯誤: {str(e)}")
            return []

def main():
    # 建立參數解析器
    parser = argparse.ArgumentParser(description='Elasticsearch 資料讀取工具')
    
    # 添加命令列參數
    parser.add_argument('--device_id', type=str, help='設備 ID')
    parser.add_argument('--start_time', type=str, help='開始時間 (YYYY-MM-DD HH:MM:SS)')
    parser.add_argument('--end_time', type=str, help='結束時間 (YYYY-MM-DD HH:MM:SS)')
    parser.add_argument('--limit', type=int, default=4000, help='限制最大讀取筆數，預設4000筆')
    parser.add_argument('--check_latest', action='store_true', help='檢查設備最新資料時間')
    parser.add_argument('--check_total_count', action='store_true', help='檢查設備資料總數')
    parser.add_argument('--host', type=str, help='自訂 Elasticsearch 主機位址')
    parser.add_argument('--output', type=str, help='指定輸出檔名（不含副檔名，預設會加上日期）')
    
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
        'hosts': args.host or os.getenv('ELASTICSEARCH_HOST', 'http://192.168.1.68:9200'),
        'api_key': os.getenv('ELASTICSEARCH_API_KEY'),
        'verify_certs': False,
    }
    
    # 檢查必要的參數
    if not es_config['api_key']:
        print("錯誤：未提供 API Key")
        return
    
    print(f"嘗試連接到 Elasticsearch 伺服器: {es_config['hosts']}")
    
    try:
        # 建立 loader
        loader = ElasticDataLoader(**es_config)

        # 測試連線是否正確
        connection_success = loader.test_data()
        if not connection_success:
            print("連線測試失敗")
            return
        
    except Exception as e:
        print(f"發生錯誤: {str(e)}")
        return
    
    try:
        # if args.check_latest and args.device_id:
        #     loader.get_latest_data_time(args.device_id)
        #     sys.exit(0)
        # ============================================
        # # 查詢特定設備的資料總共有幾筆
        # if args.check_total_count and args.device_id:
        #     total_count = loader.get_device_data_count(args.device_id)
        #     sys.exit(0)
        
        # ============================================
        # # 利用時間區間查詢資料
        # data = loader.fetch_data(args.device_id, args.start_time, args.end_time, limit=args.limit)

        # # 依照原始 timestamp 欄位排序並存成 CSV
        # if not data.empty:
        #     data = data.sort_values(by='timestamp')
            
        #     # 將時間格式轉換為較短的格式
        #     start_short = datetime.strptime(args.start_time, '%Y-%m-%d %H:%M:%S').strftime('%Y%m%d_%H')
        #     end_short = datetime.strptime(args.end_time, '%Y-%m-%d %H:%M:%S').strftime('%Y%m%d_%H')
            
        #     # 使用較短的時間格式建立檔名
        #     filename = f'{args.device_id}_{start_short}_{end_short}.csv'
        #     data.to_csv(f'_data/elastic_data/{filename}', index=True)
        #     print(f"資料已成功存成CSV檔案: {filename}")
        # else:
        #     print("未找到符合條件的資料")

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
        # # 計算有 serial_id 欄位的文件數量
        # doc_count, serial_ids = loader.count_serial_id_exists()
        
        # # 如果找到設備 ID，就將其排序並儲存
        # if serial_ids and len(serial_ids) > 0:
        #     # 去除重複的 serial_id 並排序   
        #     serial_ids = sorted(list(set(serial_ids)))
            
        #     print(f"\n所有設備的 serial_id: {serial_ids}")
        #     print(f"找到 {len(serial_ids)} 個唯一設備 ID")
            
        #     # 將 serial_id 存成CSV
        #     pd.DataFrame(serial_ids, columns=['serial_id']).to_csv('serial_ids.csv', index=False)
        #     print(f"serial_id 已成功存成CSV檔案: serial_ids.csv")
        # else:
        #     print("\n找不到任何設備 ID，請檢查 mapping 設定或嘗試其他查詢方式")
        

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

        #============================================
        # 測試連線是否正確
        connection_success = loader.test_data()
        if not connection_success:
            print("連線測試失敗")
            return
            
        try:
            # 嘗試通過分頁查詢獲取設備 ID
            print("\n嘗試通過分頁查詢獲取設備 ID...")
            serial_ids = loader.get_serial_id_paged(size=1000, max_pages=5)  # 限制最多查詢5頁，每頁1000筆
            
            # 如果找到設備 ID，就將其排序並儲存
            if serial_ids and len(serial_ids) > 0:
                # 去除重複的 serial_id 並排序   
                serial_ids = sorted(list(set(serial_ids)))
                
                print(f"\n前20個設備的 serial_id: {serial_ids[:20]}")
                print(f"找到 {len(serial_ids)} 個唯一設備 ID")
                
                # 建立含日期的檔名
                today = datetime.now().strftime('%Y%m%d')
                base_filename = args.output or 'serial_ids'
                filename = f"{base_filename}_{today}.csv"
                
                # 將 serial_id 存成CSV
                pd.DataFrame(serial_ids, columns=['serial_id']).to_csv(filename, index=False)
                print(f"serial_id 已成功存成CSV檔案: {filename}")
            else:
                print("\n找不到任何設備 ID，嘗試其他方法...")
                
                # 計算有 serial_id 欄位的文件數量
                doc_count, serial_ids = loader.count_serial_id_exists()
                
                # 如果找到設備 ID，就將其排序並儲存
                if serial_ids and len(serial_ids) > 0:
                    # 去除重複的 serial_id 並排序   
                    serial_ids = sorted(list(set(serial_ids)))
                    
                    print(f"\n所有設備的 serial_id: {serial_ids}")
                    print(f"找到 {len(serial_ids)} 個唯一設備 ID")
                    
                    # 建立含日期的檔名
                    today = datetime.now().strftime('%Y%m%d')
                    base_filename = args.output or 'serial_ids'
                    filename = f"{base_filename}_{today}.csv"
                    
                    # 將 serial_id 存成CSV
                    pd.DataFrame(serial_ids, columns=['serial_id']).to_csv(filename, index=False)
                    print(f"serial_id 已成功存成CSV檔案: {filename}")
                else:
                    print("\n找不到任何設備 ID，請檢查 mapping 設定或嘗試其他查詢方式")
                
        except Exception as e:
            print(f"處理資料時發生錯誤: {str(e)}")
        
        # #============================================
        # # 尋找過去1小時內有異常感測器讀數的設備
        # loader.find_devices_with_bad_sensor_readings(hours=1)
        
    except Exception as e:
        print(f"執行時發生錯誤: {str(e)}")
        
if __name__ == "__main__":
    main()

    # # 基本查詢範例
    # python elastic_data_loader.py --device_id "SPS2024PA000355" --start_time "2025-02-25 12:00:00" --end_time "2025-02-26 12:00:00" --limit 100
    # python elastic_data_loader.py --device_id "SPS2021PA000336" --start_time "2025-02-01 00:00:00" --end_time "2025-02-05 00:00:00"

    # # 只查詢最近24小時的資料
    # python elastic_data_loader.py --device_id "SPS2021PA000336"

    # 先檢查資料時間範圍
    # python elastic_data_loader.py --device_id "SPS2024PA000355" --check_latest

    # 再用正確的時間範圍查詢
    # python elastic_data_loader.py --device_id "2024HB000052" --start_time "2024-02-03 12:00:00" --end_time "2024-02-08 12:00:00"