#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
檔案清理工具
專門用於刪除指定資料夾中的特定檔案模式
支援精確的檔案名稱匹配，避免誤刪重要檔案
支援自定義檔案名稱模式和副檔名篩選
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import argparse
from datetime import datetime


class FileCleanerTool:
    """檔案清理工具類別"""
    
    def __init__(self, target_directory: str = "./_data/pyqt_viewer"):
        """
        初始化檔案清理工具
        
        Args:
            target_directory: 目標資料夾路徑
        """
        self.target_directory = Path(target_directory)
        self.deleted_files = []
        self.skipped_files = []
        
    def scan_files(self) -> Dict[str, List[str]]:
        """
        掃描目標資料夾中的檔案
        
        Returns:
            包含不同檔案類型的字典
        """
        if not self.target_directory.exists():
            print(f"❌ 目標資料夾不存在: {self.target_directory}")
            return {}
            
        files = {
            'full_data_csv': [],
            'data_csv': [],
            'npz_files': [],
            'other_files': []
        }
        
        # 掃描所有檔案
        for file_path in self.target_directory.glob("*"):
            if file_path.is_file():
                filename = file_path.name
                
                # 分類檔案
                if filename.endswith('_full_data.csv'):
                    files['full_data_csv'].append(str(file_path))
                elif filename.endswith('_data.csv') and not filename.endswith('_full_data.csv'):
                    files['data_csv'].append(str(file_path))
                elif filename.endswith('.npz'):
                    files['npz_files'].append(str(file_path))
                else:
                    files['other_files'].append(str(file_path))
        
        return files
    
    def scan_files_by_extension(self, extension: str) -> List[str]:
        """
        根據副檔名掃描檔案
        
        Args:
            extension: 副檔名（如 '.csv', '.npz', '.txt'）
            
        Returns:
            符合副檔名的檔案列表
        """
        if not self.target_directory.exists():
            print(f"❌ 目標資料夾不存在: {self.target_directory}")
            return []
        
        # 確保副檔名格式正確
        if not extension.startswith('.'):
            extension = '.' + extension
            
        files = []
        for file_path in self.target_directory.glob(f"*{extension}"):
            if file_path.is_file():
                files.append(str(file_path))
        
        return files
    
    def delete_files_by_pattern(
        self, 
        pattern: str, 
        extension: Optional[str] = None,
        exclude_pattern: Optional[str] = None,
        dry_run: bool = True
    ) -> Tuple[int, int]:
        """
        根據自定義模式刪除檔案
        
        Args:
            pattern: 檔案名稱模式 (支援萬用字元 * 和 ?)
            extension: 副檔名篩選 (可選，如 '.csv', '.npz')
            exclude_pattern: 排除模式 (可選，符合此模式的檔案不會被刪除)
            dry_run: 是否為模擬執行
            
        Returns:
            (成功刪除數量, 跳過數量)
        """
        # 獲取檔案列表
        if extension:
            all_files = self.scan_files_by_extension(extension)
            print(f"\n🔍 在 {extension} 檔案中搜尋...")
        else:
            # 掃描所有檔案
            all_files = []
            for file_path in self.target_directory.glob("*"):
                if file_path.is_file():
                    all_files.append(str(file_path))
            print(f"\n🔍 在所有檔案中搜尋...")
        
        print(f"🎯 使用模式: {pattern}")
        if exclude_pattern:
            print(f"❌ 排除模式: {exclude_pattern}")
        if extension:
            print(f"📝 副檔名篩選: {extension}")
        
        print(f"📁 找到 {len(all_files)} 個候選檔案")
        
        deleted_count = 0
        skipped_count = 0
        matched_count = 0
        
        # 將萬用字元模式轉換為正規表達式
        regex_pattern = pattern.replace('*', '.*').replace('?', '.')
        regex_pattern = f"^{regex_pattern}$"
        
        exclude_regex = None
        if exclude_pattern:
            exclude_regex = exclude_pattern.replace('*', '.*').replace('?', '.')
            exclude_regex = f"^{exclude_regex}$"
        
        for file_path in all_files:
            filename = Path(file_path).name
            
            # 檢查檔案名稱是否符合主要模式
            if re.match(regex_pattern, filename):
                matched_count += 1
                
                # 檢查是否符合排除模式
                if exclude_regex and re.match(exclude_regex, filename):
                    print(f"  ⚠️  跳過: {filename} (符合排除模式)")
                    skipped_count += 1
                    continue
                
                # 執行刪除操作
                if dry_run:
                    print(f"  📋 [模擬] 將刪除: {filename}")
                    deleted_count += 1
                else:
                    try:
                        os.remove(file_path)
                        print(f"  ✅ 已刪除: {filename}")
                        self.deleted_files.append(file_path)
                        deleted_count += 1
                    except Exception as e:
                        print(f"  ❌ 刪除失敗: {filename} - {e}")
                        self.skipped_files.append(file_path)
                        skipped_count += 1
            else:
                print(f"  ⏭️  跳過: {filename} (不符合模式)")
                skipped_count += 1
        
        print(f"\n🔍 模式匹配結果:")
        print(f"  🎯 符合主要模式: {matched_count}")
        if exclude_pattern:
            excluded_count = matched_count - deleted_count if dry_run else matched_count - deleted_count
            print(f"  ❌ 被排除: {excluded_count}")
        
        return deleted_count, skipped_count
    
    def delete_full_data_csv(self, dry_run: bool = True) -> Tuple[int, int]:
        """
        刪除 *_full_data.csv 檔案（不會刪除 *_data.csv）
        這是一個快捷方法，等同於使用自定義模式
        
        Args:
            dry_run: 是否為模擬執行（不實際刪除）
            
        Returns:
            (成功刪除數量, 跳過數量)
        """
        print("\n🔧 使用快捷模式：刪除 *_full_data.csv 檔案")
        return self.delete_files_by_pattern(
            pattern="*_full_data.csv",
            extension=".csv",
            exclude_pattern=None,
            dry_run=dry_run
        )
    
    def delete_npz_files_by_pattern(self, pattern: str, dry_run: bool = True) -> Tuple[int, int]:
        """
        根據特定模式刪除 .npz 檔案
        這是一個快捷方法，等同於使用自定義模式
        
        Args:
            pattern: 檔案名稱模式 (支援萬用字元 * 和 ?)
            dry_run: 是否為模擬執行
            
        Returns:
            (成功刪除數量, 跳過數量)
        """
        print("\n🔧 使用快捷模式：刪除 .npz 檔案")
        return self.delete_files_by_pattern(
            pattern=pattern,
            extension=".npz",
            exclude_pattern=None,
            dry_run=dry_run
        )
    
    def list_files_by_type(self):
        """列出所有檔案並按類型分組"""
        files = self.scan_files()
        
        print(f"\n📁 目標資料夾: {self.target_directory}")
        print("=" * 60)
        
        for file_type, file_list in files.items():
            if file_list:
                type_names = {
                    'full_data_csv': '*_full_data.csv 檔案',
                    'data_csv': '*_data.csv 檔案 (非 full_data)',
                    'npz_files': '.npz 檔案',
                    'other_files': '其他檔案'
                }
                
                print(f"\n📂 {type_names[file_type]} ({len(file_list)} 個):")
                for file_path in sorted(file_list):
                    filename = Path(file_path).name
                    file_size = Path(file_path).stat().st_size
                    size_mb = file_size / (1024 * 1024)
                    print(f"  • {filename} ({size_mb:.1f} MB)")
    
    def list_files_by_extension(self, extension: str):
        """根據副檔名列出檔案"""
        files = self.scan_files_by_extension(extension)
        
        print(f"\n📁 目標資料夾: {self.target_directory}")
        print(f"📝 副檔名篩選: {extension}")
        print("=" * 60)
        
        if files:
            print(f"\n📂 {extension} 檔案 ({len(files)} 個):")
            for file_path in sorted(files):
                filename = Path(file_path).name
                file_size = Path(file_path).stat().st_size
                size_mb = file_size / (1024 * 1024)
                print(f"  • {filename} ({size_mb:.1f} MB)")
        else:
            print(f"\n❌ 未找到任何 {extension} 檔案")
    
    def create_backup_log(self):
        """創建刪除記錄檔案"""
        if self.deleted_files:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = f"./_logs/deleted_files_log_{timestamp}.txt"
            
            with open(log_file, 'w', encoding='utf-8') as f:
                f.write(f"檔案刪除記錄 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"已刪除檔案 ({len(self.deleted_files)} 個):\n")
                for file_path in self.deleted_files:
                    f.write(f"  - {file_path}\n")
                
                if self.skipped_files:
                    f.write(f"\n跳過的檔案 ({len(self.skipped_files)} 個):\n")
                    for file_path in self.skipped_files:
                        f.write(f"  - {file_path}\n")
            
            print(f"\n📝 刪除記錄已保存至: {log_file}")


def main():
    """主程式"""
    parser = argparse.ArgumentParser(
        description="檔案清理工具 - 安全刪除指定模式的檔案",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用範例:
  # 列出所有檔案
  python file_cleaner.py --list
  
  # 列出特定副檔名的檔案
  python file_cleaner.py --list-ext .csv
  
  # 模擬刪除所有 *_full_data.csv 檔案（快捷方式）
  python file_cleaner.py --delete-full-data --dry-run
  
  # 自定義刪除：刪除所有 .csv 檔案中包含 'backup' 的檔案
  python file_cleaner.py --delete-files "*backup*" --extension .csv --dry-run
  
  # 自定義刪除：刪除所有 .log 檔案，但排除包含 'important' 的檔案
  python file_cleaner.py --delete-files "*.log" --exclude "*important*" --dry-run
  
  # 自定義刪除：刪除特定模式的 .npz 檔案
  python file_cleaner.py --delete-files "*_processed.npz" --extension .npz --dry-run
  
  # 刪除所有以 'temp_' 開頭的檔案
  python file_cleaner.py --delete-files "temp_*" --dry-run
  
  # 指定不同的目標資料夾
  python file_cleaner.py --directory "./_data/training" --list
        """
    )
    
    parser.add_argument(
        '--directory', '-d',
        default='./_data/pyqt_viewer',
        help='目標資料夾路徑 (預設: ./_data/pyqt_viewer)'
    )
    
    parser.add_argument(
        '--list', '-l',
        action='store_true',
        help='列出所有檔案並按類型分組'
    )
    
    parser.add_argument(
        '--list-ext',
        metavar='EXTENSION',
        help='列出指定副檔名的檔案 (如: .csv, .npz, .txt)'
    )
    
    parser.add_argument(
        '--delete-files',
        metavar='PATTERN',
        help='根據自定義模式刪除檔案 (支援 * 和 ? 萬用字元)'
    )
    
    parser.add_argument(
        '--extension', '-e',
        metavar='EXT',
        help='指定副檔名篩選 (如: .csv, .npz, .txt)'
    )
    
    parser.add_argument(
        '--exclude',
        metavar='PATTERN',
        help='排除模式：符合此模式的檔案不會被刪除'
    )
    
    # 保留原有的快捷方式功能
    parser.add_argument(
        '--delete-full-data',
        action='store_true',
        help='[快捷方式] 刪除所有 *_full_data.csv 檔案'
    )
    
    parser.add_argument(
        '--delete-npz',
        metavar='PATTERN',
        help='[快捷方式] 根據模式刪除 .npz 檔案 (支援 * 和 ? 萬用字元)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='模擬執行，不實際刪除檔案'
    )
    
    args = parser.parse_args()
    
    # 創建檔案清理工具實例
    cleaner = FileCleanerTool(args.directory)
    
    print("🧹 檔案清理工具 v2.0")
    print("=" * 45)
    
    # 執行相應的操作
    if args.list:
        cleaner.list_files_by_type()
    
    elif args.list_ext:
        cleaner.list_files_by_extension(args.list_ext)
    
    elif args.delete_files:
        if args.dry_run:
            print("\n🔍 模擬執行模式 - 不會實際刪除檔案")
        else:
            print("\n⚠️  實際刪除模式 - 檔案將被永久刪除")
            confirm = input("確定要繼續嗎？(y/N): ")
            if confirm.lower() != 'y':
                print("❌ 操作已取消")
                return
        
        deleted, skipped = cleaner.delete_files_by_pattern(
            pattern=args.delete_files,
            extension=args.extension,
            exclude_pattern=args.exclude,
            dry_run=args.dry_run
        )
        
        print(f"\n📊 執行結果:")
        print(f"  ✅ 處理檔案: {deleted}")
        print(f"  ⏭️  跳過檔案: {skipped}")
        
        if not args.dry_run and deleted > 0:
            cleaner.create_backup_log()
    
    elif args.delete_full_data:
        if args.dry_run:
            print("\n🔍 模擬執行模式 - 不會實際刪除檔案")
        else:
            print("\n⚠️  實際刪除模式 - 檔案將被永久刪除")
            confirm = input("確定要繼續嗎？(y/N): ")
            if confirm.lower() != 'y':
                print("❌ 操作已取消")
                return
        
        deleted, skipped = cleaner.delete_full_data_csv(dry_run=args.dry_run)
        
        print(f"\n📊 執行結果:")
        print(f"  ✅ 處理檔案: {deleted}")
        print(f"  ⏭️  跳過檔案: {skipped}")
        
        if not args.dry_run and deleted > 0:
            cleaner.create_backup_log()
    
    elif args.delete_npz:
        if args.dry_run:
            print("\n🔍 模擬執行模式 - 不會實際刪除檔案")
        else:
            print("\n⚠️  實際刪除模式 - 檔案將被永久刪除")
            confirm = input("確定要繼續嗎？(y/N): ")
            if confirm.lower() != 'y':
                print("❌ 操作已取消")
                return
        
        deleted, skipped = cleaner.delete_npz_files_by_pattern(args.delete_npz, dry_run=args.dry_run)
        
        print(f"\n📊 執行結果:")
        print(f"  ✅ 處理檔案: {deleted}")
        print(f"  ⏭️  跳過檔案: {skipped}")
        
        if not args.dry_run and deleted > 0:
            cleaner.create_backup_log()
    
    else:
        print("❓ 請指定要執行的操作。使用 --help 查看所有選項。")
        cleaner.list_files_by_type()


if __name__ == "__main__":
    main() 