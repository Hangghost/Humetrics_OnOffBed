import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Conv1D, MaxPooling1D, Flatten, LSTM
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping
from tensorflow.keras.layers import Input
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer

# 設定隨機種子
np.random.seed(1337)

# 載入數據
dataset = np.loadtxt("pima-indians-diabetes.csv", delimiter=",")
X = dataset[:,0:8]
Y = dataset[:,8]

# 正規化數據
scaler = Normalizer().fit(X)
X = scaler.transform(X)

# 分割訓練和測試集
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

# 重塑數據
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# 模型參數
lstm_output_size = 70

# 創建模型
cnn = Sequential([
    Input(shape=(8, 1)),  # 明確指定輸入層
    Conv1D(64, 3, padding="same", activation="relu"),
    MaxPooling1D(pool_size=2),
    LSTM(lstm_output_size),
    Dropout(0.1),
    Dense(1, activation="sigmoid")
])

# 編譯模型
cnn.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])

# 確保日誌目錄存在
log_dir = "logs/cnn-lstm"
os.makedirs(log_dir, exist_ok=True)

# 設置回調
checkpointer = ModelCheckpoint(
    filepath=os.path.join(log_dir, "checkpoint-{epoch:02d}.keras"),  # 改為 .keras 格式
    monitor='val_accuracy',
    mode='max',
    save_best_only=True,
    verbose=1
)
csv_logger = CSVLogger(os.path.join(log_dir, 'training.csv'))
early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=10,
    restore_best_weights=True
)

# 訓練模型
history = cnn.fit(
    X_train, y_train,
    epochs=1000,
    validation_data=(X_test, y_test),
    callbacks=[checkpointer, csv_logger, early_stopping]
)

# 保存最終模型
cnn.save(os.path.join(log_dir, "final_model.keras"))  # 改為 .keras 格式

# 評估模型
loss, accuracy = cnn.evaluate(X_test, y_test)
print("\nLoss: %.2f, Accuracy: %.2f%%" % (loss, accuracy*100))