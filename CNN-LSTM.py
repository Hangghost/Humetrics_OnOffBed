import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, LSTM, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping
from tensorflow.keras.layers import Input
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
import matplotlib.pyplot as plt
import os

# 設定隨機種子
np.random.seed(1337)

# 設定參數
WINDOW_SIZE = 15  # 15秒的窗口
OVERLAP = 0.8    # 80% 重疊
STEP_SIZE = int(WINDOW_SIZE * (1 - OVERLAP))  # 滑動步長

def create_sequences(df):
    """創建時間序列窗口"""
    sequences = []
    labels = []
    
    # 先檢查數據
    print(f"原始數據長度: {len(df)}")
    
    # 確保所有需要的列都存在
    required_columns = (
        [f'Channel_{i}_Raw' for i in range(1, 7)] +
        [f'Channel_{i}_Noise' for i in range(1, 7)] +
        ['OnBed_Status']
    )
    
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"缺少必要的列: {[col for col in required_columns if col not in df.columns]}")
    
    # 確保數據類型為 float64
    for col in required_columns[:-1]:  # 除了 OnBed_Status
        df[col] = df[col].astype('float64')
    
    # 檢查數值是否有效
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    print(f"清理後數據長度: {len(df)}")
    
    # 計算特徵
    for i in range(0, len(df) - WINDOW_SIZE + 1, STEP_SIZE):
        window = df.iloc[i:(i + WINDOW_SIZE)]
        
        try:
            # 確保數據類型為 float64
            raw_values = window[[f'Channel_{i}_Raw' for i in range(1, 7)]].values.astype('float64')
            noise_values = window[[f'Channel_{i}_Noise' for i in range(1, 7)]].values.astype('float64')
            
            # 使用 np.divide 進行安全的除法運算
            noise_ratios = np.divide(
                noise_values, 
                raw_values, 
                out=np.zeros_like(noise_values, dtype='float64'), 
                where=raw_values!=0
            )
            
            pressure_changes = np.diff(raw_values, axis=0)
            pressure_changes = np.vstack([pressure_changes, pressure_changes[-1]])
            
            pressure_center = np.average(raw_values, axis=1, weights=range(1, 7))
            
            stats_features = np.concatenate([
                np.mean(raw_values, axis=0),
                np.std(raw_values, axis=0),
                np.percentile(raw_values, [25, 50, 75], axis=0).flatten()
            ])
            
            features = np.concatenate([
                raw_values,
                noise_ratios,
                pressure_changes,
                pressure_center.reshape(-1, 1),
                stats_features.reshape(1, -1).repeat(WINDOW_SIZE, axis=0)
            ], axis=1)
            
            sequences.append(features)
            labels.append(window['OnBed_Status'].iloc[-1])
            
        except Exception as e:
            print(f"處理窗口 {i} 時發生錯誤: {e}")
            continue
    
    print(f"生成序列數量: {len(sequences)}")
    
    if len(sequences) == 0:
        raise ValueError("沒有生成任何有效的序列")
    
    return np.array(sequences), np.array(labels)

# 載入數據
try:
    dataset = pd.read_csv("./_data/SPS2021PA000329_20241212_04_20241213_04_data.csv")
    print(f"數據集形狀: {dataset.shape}")
    print(f"數據集列: {dataset.columns.tolist()}")
    X, y = create_sequences(dataset)
    print(f"特徵形狀: {X.shape}")
    print(f"標籤形狀: {y.shape}")
except Exception as e:
    print(f"數據��理錯誤: {e}")
    raise

# 分割訓練和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 創建日誌目錄
log_dir = '_logs/bed_monitor'
os.makedirs(log_dir, exist_ok=True)

# 定義檔案路徑
model_checkpoint_path = os.path.join(log_dir, 'model-{epoch:02d}-{val_accuracy:.4f}.keras')
training_log_path = os.path.join(log_dir, 'training.csv')
final_model_path = os.path.join(log_dir, 'final_model.keras')
training_history_path = os.path.join(log_dir, 'training_history.png')

# 建立模型
input_layer = Input(shape=(WINDOW_SIZE, X_train.shape[2]))
x = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(input_layer)
x = BatchNormalization()(x)
x = MaxPooling1D(pool_size=2)(x)
x = Dropout(0.2)(x)

x = Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(x)
x = BatchNormalization()(x)
x = MaxPooling1D(pool_size=2)(x)
x = Dropout(0.2)(x)

x = LSTM(units=256, return_sequences=True)(x)
x = Dropout(0.3)(x)
x = LSTM(units=128)(x)
x = Dropout(0.3)(x)

output_layer = Dense(1, activation='sigmoid')(x)

model = Model(inputs=input_layer, outputs=output_layer)

# 編譯模型
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# 設置回調
callbacks = [
    EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        restore_best_weights=True
    ),
    ModelCheckpoint(
        filepath=model_checkpoint_path,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max'
    ),
    CSVLogger(training_log_path)
]

# 訓練模型
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.2,
    callbacks=callbacks,
    verbose=1
)

# 評估模型
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"\nTest accuracy: {test_accuracy:.4f}")

# 保存最終模型
model.save(final_model_path)

# 可視化訓練過程
plt.figure(figsize=(12, 4))

# 繪製準確率
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# 繪製損失
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig(training_history_path)
plt.close()