# 檔案清理工具使用說明 v2.0

## 概述

這是一個功能強大的檔案清理工具，能夠安全且精確地刪除指定資料夾中的特定檔案模式。工具支援：

1. **自定義檔案名稱模式**：可自由設定要刪除的檔案名稱模式
2. **副檔名篩選**：可指定特定副檔名進行篩選
3. **排除模式**：可設定排除條件，避免誤刪重要檔案
4. **快捷方式**：保留原有的快速刪除功能

## 主要特色

- ✅ **完全自定義**：可自由組合檔案名稱模式、副檔名、排除條件
- ✅ **安全性優先**：多重確認機制，避免誤刪重要檔案
- ✅ **模擬執行模式**：可先預覽將要刪除的檔案
- ✅ **精確模式匹配**：使用正規表達式確保精確匹配
- ✅ **排除機制**：可設定排除模式保護重要檔案
- ✅ **詳細記錄**：自動生成刪除記錄檔案
- ✅ **UTF-8 支援**：完整支援中文檔案名稱
- ✅ **向後兼容**：保留原有的快捷方式功能

## 安裝與執行

### 前置需求
- Python 3.6 或更高版本
- 標準函式庫（無需額外安裝套件）

### 執行方式
```bash
python file_cleaner.py [選項]
```

## 新功能使用範例

### 1. 自定義檔案模式刪除

```bash
# 刪除所有包含 'backup' 的 .csv 檔案
python file_cleaner.py --delete-files "*backup*" --extension .csv --dry-run

# 刪除所有以 'temp_' 開頭的檔案（任何副檔名）
python file_cleaner.py --delete-files "temp_*" --dry-run

# 刪除所有 .log 檔案，但排除包含 'important' 的檔案
python file_cleaner.py --delete-files "*.log" --exclude "*important*" --dry-run

# 刪除特定模式的 .npz 檔案
python file_cleaner.py --delete-files "*_processed.npz" --extension .npz --dry-run
```

### 2. 副檔名篩選

```bash
# 列出所有 .csv 檔案
python file_cleaner.py --list-ext .csv

# 列出所有 .npz 檔案
python file_cleaner.py --list-ext .npz

# 刪除所有 .tmp 檔案
python file_cleaner.py --delete-files "*" --extension .tmp --dry-run
```

### 3. 複雜的篩選條件

```bash
# 刪除所有包含日期的 .csv 檔案，但排除 'full_data' 檔案
python file_cleaner.py --delete-files "*2025*" --extension .csv --exclude "*full_data*" --dry-run

# 刪除所有以設備編號開頭的 .backup 檔案
python file_cleaner.py --delete-files "SPS202?PA*" --extension .backup --dry-run

# 刪除所有臨時檔案，但保留重要的檔案
python file_cleaner.py --delete-files "*temp*" --exclude "*keep*" --exclude "*save*" --dry-run
```

## 傳統快捷方式（向後兼容）

### 1. 查看所有檔案分類
```bash
# 列出預設資料夾中的所有檔案
python file_cleaner.py --list

# 列出指定資料夾中的檔案
python file_cleaner.py --directory "./_data/training" --list
```

### 2. 刪除 *_full_data.csv 檔案

```bash
# 模擬執行（安全預覽）
python file_cleaner.py --delete-full-data --dry-run

# 實際執行刪除
python file_cleaner.py --delete-full-data
```

### 3. 刪除特定模式的 .npz 檔案

```bash
# 模擬刪除特定模式的 .npz 檔案
python file_cleaner.py --delete-npz "*_processed.npz" --dry-run

# 實際執行刪除
python file_cleaner.py --delete-npz "*_processed.npz"
```

## 命令列選項完整清單

| 選項 | 簡寫 | 說明 |
|------|------|------|
| `--directory` | `-d` | 指定目標資料夾路徑（預設：`./_data/pyqt_viewer`） |
| `--list` | `-l` | 列出所有檔案並按類型分組 |
| `--list-ext` | - | 列出指定副檔名的檔案 |
| `--delete-files` | - | **[新功能]** 根據自定義模式刪除檔案 |
| `--extension` | `-e` | **[新功能]** 指定副檔名篩選 |
| `--exclude` | - | **[新功能]** 排除模式：符合此模式的檔案不會被刪除 |
| `--delete-full-data` | - | [快捷方式] 刪除所有 `*_full_data.csv` 檔案 |
| `--delete-npz` | - | [快捷方式] 根據模式刪除 `.npz` 檔案 |
| `--dry-run` | - | 模擬執行，不實際刪除檔案 |
| `--help` | `-h` | 顯示幫助資訊 |

## 萬用字元模式說明

支援以下萬用字元：
- `*`：匹配任意數量的字符
- `?`：匹配單一字符

### 模式範例

| 模式 | 匹配範例 |
|------|----------|
| `*.csv` | 所有 .csv 檔案 |
| `*backup*` | 檔案名稱包含 'backup' 的檔案 |
| `temp_*` | 以 'temp_' 開頭的檔案 |
| `SPS202?PA*` | 以 'SPS202' 開頭，第六位是任意字符，然後是 'PA' 的檔案 |
| `*_full_data.csv` | 以 '_full_data.csv' 結尾的檔案 |
| `log_2025*.txt` | 以 'log_2025' 開頭的 .txt 檔案 |

## 新增排除功能

排除功能讓您可以保護重要檔案不被誤刪：

```bash
# 刪除所有 .csv 檔案，但保留包含 'important' 的檔案
python file_cleaner.py --delete-files "*.csv" --exclude "*important*" --dry-run

# 刪除所有臨時檔案，但保留多個重要模式
python file_cleaner.py --delete-files "*temp*" --exclude "*keep*" --exclude "*save*" --dry-run
```

## 安全機制

### 1. 模擬執行模式
- 使用 `--dry-run` 參數可以預覽將要刪除的檔案
- 不會實際刪除任何檔案，只顯示操作結果

### 2. 確認提示
- 實際刪除前會要求用戶確認
- 輸入 `y` 才會繼續執行

### 3. 精確匹配
- 使用正規表達式確保精確匹配檔案名稱
- 多重檢查機制避免誤刪

### 4. 排除保護
- 支援排除模式，保護重要檔案
- 顯示詳細的匹配和排除統計

### 5. 刪除記錄
- 自動生成時間戳記的刪除記錄檔案
- 記錄所有已刪除和跳過的檔案

## 輸出範例

### 列出特定副檔名檔案
```
🧹 檔案清理工具 v2.0
=============================================

📁 目標資料夾: ./_data/pyqt_viewer
📝 副檔名篩選: .csv
============================================================

📂 .csv 檔案 (150 個):
  • SPS2021PA000003_20250526_04_20250527_04_data.csv (11.8 MB)
  • SPS2021PA000003_20250526_04_20250527_04_full_data.csv (12.5 MB)
  ...
```

### 自定義模式刪除
```
🔍 模擬執行模式 - 不會實際刪除檔案

🔍 在 .csv 檔案中搜尋...
🎯 使用模式: *backup*
❌ 排除模式: *important*
📝 副檔名篩選: .csv
📁 找到 150 個候選檔案

  📋 [模擬] 將刪除: data_backup_20250101.csv
  📋 [模擬] 將刪除: log_backup_file.csv
  ⚠️  跳過: important_backup.csv (符合排除模式)
  ⏭️  跳過: normal_data.csv (不符合模式)

🔍 模式匹配結果:
  🎯 符合主要模式: 15
  ❌ 被排除: 1

📊 執行結果:
  ✅ 處理檔案: 14
  ⏭️  跳過檔案: 136
```

## 常見使用情境

### 情境一：清理備份檔案
```bash
# 清理所有備份檔案，但保留重要備份
python file_cleaner.py --delete-files "*backup*" --exclude "*important*" --dry-run
python file_cleaner.py --delete-files "*backup*" --exclude "*important*"
```

### 情境二：清理特定日期的檔案
```bash
# 清理 2024 年的檔案
python file_cleaner.py --delete-files "*2024*" --dry-run

# 清理特定月份的 .log 檔案
python file_cleaner.py --delete-files "*202401*" --extension .log --dry-run
```

### 情境三：清理臨時檔案
```bash
# 清理所有臨時檔案
python file_cleaner.py --delete-files "temp_*" --dry-run
python file_cleaner.py --delete-files "*tmp*" --dry-run

# 清理特定副檔名的臨時檔案
python file_cleaner.py --delete-files "*" --extension .tmp --dry-run
```

### 情境四：清理特定設備的檔案
```bash
# 清理特定設備的所有檔案
python file_cleaner.py --delete-files "SPS2021PA000003_*" --dry-run

# 清理特定設備的特定類型檔案
python file_cleaner.py --delete-files "SPS2021PA000003_*" --extension .npz --dry-run
```

### 情境五：清理多個資料夾
```bash
# 清理不同資料夾中的檔案
python file_cleaner.py --directory "./_data/training" --delete-files "*temp*" --dry-run
python file_cleaner.py --directory "./_data/experiment" --delete-files "*backup*" --dry-run
```

## 功能比較

| 功能 | 傳統方式 | 新方式 |
|------|----------|--------|
| 刪除 full_data.csv | `--delete-full-data` | `--delete-files "*_full_data.csv" --extension .csv` |
| 刪除特定 .npz | `--delete-npz "*_processed.npz"` | `--delete-files "*_processed.npz" --extension .npz` |
| 自定義模式 | ❌ | ✅ `--delete-files "自定義模式"` |
| 副檔名篩選 | ❌ | ✅ `--extension .副檔名` |
| 排除保護 | ❌ | ✅ `--exclude "排除模式"` |

## 注意事項

1. **備份重要資料**：執行實際刪除前，請確保已備份重要資料
2. **先使用模擬模式**：建議先使用 `--dry-run` 預覽結果
3. **善用排除功能**：使用 `--exclude` 保護重要檔案
4. **檢查刪除記錄**：執行後檢查生成的記錄檔案
5. **權限問題**：確保有足夠權限刪除目標檔案
6. **路徑正確性**：確認目標資料夾路徑正確

## 故障排除

### 問題：找不到符合條件的檔案
- 檢查檔案名稱模式是否正確
- 使用 `--list` 或 `--list-ext` 查看實際檔案
- 確認副檔名格式（需包含點號，如 `.csv`）

### 問題：意外刪除了重要檔案
- 使用 `--exclude` 參數設定排除模式
- 先用 `--dry-run` 預覽結果
- 檢查生成的刪除記錄檔案

### 問題：權限不足
- 確保有檔案刪除權限
- 在 macOS/Linux 上可能需要適當的檔案權限

### 問題：中文檔名顯示異常
- 確保終端支援 UTF-8 編碼
- 工具已內建 UTF-8 支援

## 技術實作細節

- **檔案掃描**：使用 `pathlib.Path.glob()` 進行高效檔案掃描
- **模式匹配**：將萬用字元轉換為正規表達式進行精確匹配
- **排除機制**：支援多重排除條件的組合邏輯
- **安全刪除**：使用 `os.remove()` 進行檔案刪除，包含完整異常處理
- **記錄機制**：自動生成帶時間戳的詳細刪除記錄檔案
- **向後兼容**：保留原有功能作為快捷方式，無縫升級 