import pandas as pd
import os
import math
import plotly.graph_objects as go
import plotly.io as pio
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import normaltest
import pandas_ta as ta
import tensorflow as tf
from datetime import datetime
import matplotlib.dates as mdates
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, GRU
from tensorflow.keras.layers import Dense, Dropout,Input
from tensorflow.keras.callbacks import EarlyStopping
from keras.models import Model
from sklearn.metrics import mean_absolute_error,r2_score,mean_squared_error, mean_absolute_percentage_error

def clean_data(datasets):
    datasets.columns = datasets.columns.str.strip()
    datasets = datasets.drop(columns=["Volume", "Adj Close"], errors='ignore')
    datasets["Open"] = datasets["Open"].astype(str)
    datasets = datasets[~datasets["Open"].str.contains("Dividend", na=False)]
    datasets = datasets[~datasets["Open"].str.contains("Stock Splits", na=False)]
    datasets["Open"] = datasets["Open"].str.replace(",", "", regex=True).astype(float
    cols_to_convert = ["Open", "High", "Low", "Close"]
    for col in cols_to_convert:
        if col in datasets.columns:
            datasets[col] = datasets[col].astype(str).str.replace(",", "", regex=True).astype(float)
            datasets[col] = np.round(datasets[col])
    datasets = datasets.dropna()
    datasets = datasets[::-1].reset_index(drop=True)

    return datasets
class CompanyAwareNormalizer:
    def __init__(self):
        self.scalers_all = {}
        self.scalers_col = {}  
    def fit_transform_all(self, nama_perusahaan, data):
        if nama_perusahaan not in self.scalers_all:
            self.scalers_all[nama_perusahaan] = MinMaxScaler()
        scaler = self.scalers_all[nama_perusahaan]
        return scaler.fit_transform(data)
    
    def inverse_transform_all(self, nama_perusahaan, scaled_data):
        if nama_perusahaan not in self.scalers_all:
            raise ValueError(f"Scaler untuk perusahaan {nama_perusahaan} tidak ditemukan")        
        return self.scalers_all[nama_perusahaan].inverse_transform(scaled_data)
    
    def fit_transform_column(self, nama_perusahaan, column, data):
        if nama_perusahaan not in self.scalers_col:
            self.scalers_col[nama_perusahaan] = {}            
        if column not in self.scalers_col[nama_perusahaan]:
            self.scalers_col[nama_perusahaan][column] = MinMaxScaler()        
        scaler = self.scalers_col[nama_perusahaan][column]
        return scaler.fit_transform(data.values.reshape(-1, 1)).flatten()
    
    def inverse_transform_column(self, nama_perusahaan, column, scaled_data):
        if nama_perusahaan not in self.scalers_col:
            raise ValueError(f"Scaler untuk perusahaan {nama_perusahaan} tidak ditemukan")            
        if column not in self.scalers_col[nama_perusahaan]:
            raise ValueError(f"Kolom {column} tidak dinormalisasi untuk perusahaan {nama_perusahaan}")        
        scaler = self.scalers_col[nama_perusahaan][column]
        return scaler.inverse_transform(scaled_data.reshape(-1, 1)).flatten()
def indikator(data):
    data = data.copy()
    data[["Open", "High", "Low", "Close"]] = data[["Open", "High", "Low", "Close"]].replace(",", "", regex=True)
    ma = ta.ma(data["Close"])
    rsi = ta.rsi(data["Close"])
    macd = ta.macd(data["Close"]).iloc[:, [0, 1]]
    data_gabung = pd.concat([data,ma, rsi, macd], axis=1)
    data_gabung = data_gabung.dropna()
    return data_gabung

filenames = ["INDF.csv", "ICBP.csv", "KLBF.csv"]
raw_data = [pd.read_csv(file) for file in filenames]
perusahaan = [f.split("/")[-1].split(".")[0] for f in filenames]
cleaned_data = [clean_data(i) for i in raw_data]
data_indikator = [indikator(data) for data in cleaned_data]
data_indikator1 = [data_indikator[i].drop(columns=["Date"]) for i in range(len(data_indikator))]
column = ["Close"]
normalizer = CompanyAwareNormalizer()
scaled_data_all = {}
scaled_data_close = {}
for company, data in zip(perusahaan, data_indikator1):
    scaled_all = normalizer.fit_transform_all(company, data)
    scaled_data_all[company] = scaled_all 
    scaled_data_close[company] = normalizer.fit_transform_column(company, "Close", data["Close"])

onehot_encoder = OneHotEncoder(sparse=False)
perusahaan_ids = np.array(perusahaan).reshape(-1, 1)
onehot_encoder.fit(perusahaan_ids)
encoded_perusahaan = onehot_encoder.transform(perusahaan_ids)

X_train_all, X_val_all, X_test_all = [], [], []
y_train_all, y_val_all, y_test_all = [], [], []

seq_len = 20
n_future = 1
for company, data in scaled_data.items():
    X, y= [], []
    for i in range(seq_len , len(data) - n_future +1):
        X.append(data[i - seq_len:i, 0:data.shape[1]])
        y.append(data[i + n_future - 1:i + n_future, 0])
    X = np.array(X)
    y = np.array(y)
    company_id = onehot_encoder.transform([[company]])
    company_id_repeated = np.repeat(company_id, seq_len, axis=0)
    
    X_combined = []
    for seq in X:
        seq_with_id = np.hstack([seq, company_id_repeated])
        X_combined.append(seq_with_id)    
    X_combined = np.array(X_combined)
    n = len(X_combined)
    train_end = int(0.8 * n)
    val_end = train_end + int(0.1 * n)
    X_train = X_combined[:train_end]
    X_val = X_combined[train_end:val_end]
    X_test = X_combined[val_end:]    
    y_train, y_val, y_test = y[:train_end], y[train_end:val_end], y[val_end:]
    X_train_all.append(X_train)
    X_val_all.append(X_val)
    X_test_all.append(X_test)
    
    y_train_all.append(y_train), y_val_all.append(y_val), y_test_all.append(y_test)
X_train = np.concatenate(X_train_all, axis=0)
X_val = np.concatenate(X_val_all, axis=0)
X_test = np.concatenate(X_test_all, axis=0)

y_train, y_val, y_test = np.concatenate(y_train_all, axis=0), np.concatenate(y_val_all, axis=0), np.concatenate(y_test_all, axis=0)

print(f"Training shape: X = {X_train.shape}, y_Close = {y_train.shape}")
print(f"Validation shape: X = {X_val.shape}, y_Close = {y_val.shape}")
print(f"Test shape: X = {X_test.shape}, y_Close = {y_test.shape}")

list_tanggal = [data_indikator[i].Date for i in range (len(perusahaan))]
list_tanggal = [np.array(list_tanggal[i]).reshape(-1, 1) for i in range (len(perusahaan))]
tanggal_unproses_perusahaan = []
tanggal_unproses_training = []
tanggal_unproses_validasi = []
tanggal_unproses_tes = []

for j in range(len(perusahaan)):  
    tanggal_unproses = []
    
    for i in range(seq_len, len(data_indikator[j]) - n_future + 1):
        tanggal_str = list_tanggal[j][i + n_future - 1, 0]
        if isinstance(tanggal_str, str):  
            tanggal_date = datetime.strptime(tanggal_str, "%d-%b-%y").date()  
        else:
            tanggal_date = tanggal_str  
        tanggal_unproses.append(tanggal_date)  
    n = len(tanggal_unproses)
    train_end = int(0.8 * n)
    val_end = train_end + int(0.1 * n)    

    tanggal_unproses_perusahaan.append(tanggal_unproses)  
    tanggal_unproses_training.append(tanggal_unproses[:train_end])
    tanggal_unproses_validasi.append(tanggal_unproses[train_end:val_end])
    tanggal_unproses_tes.append(tanggal_unproses[val_end:])

for j, perusahaan1 in enumerate(perusahaan):
    print(f"Perusahaan: {perusahaan1}, Tanggal Target: {len(tanggal_unproses_perusahaan[j])}, Training : {len(tanggal_unproses_training[j])}, Validasi : {len(tanggal_unproses_validasi[j])}, Tes : {len(tanggal_unproses_tes[j])}")  

model_jaringan_lstm = Sequential()
model_jaringan_lstm.add(LSTM(150, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
model_jaringan_lstm.add(LSTM(150, activation='relu', return_sequences=False))
model_jaringan_lstm.add(Dropout(0.2))
model_jaringan_lstm.add(Dense(y_train.shape[1]))

model_jaringan_lstm.compile(optimizer='adam', loss='log_cosh')
model_jaringan_lstm.summary()

model_bilstm = Sequential()

model_bilstm.add(
    Bidirectional(
        LSTM(80, activation='relu', return_sequences=True),
        input_shape=(X_train.shape[1], X_train.shape[2])
    )
)
model_bilstm.add(
    Bidirectional(
        LSTM(80, activation='relu', return_sequences=False)
    )
)
model_bilstm.add(Dropout(0.2))
model_bilstm.add(Dense(y_train.shape[1]))
model_bilstm.compile(optimizer='adam', loss='log_cosh')
model_bilstm.summary()

model_gru = Sequential()

model_gru.add(
    GRU(100, activation='relu', 
        input_shape=(X_train.shape[1], X_train.shape[2]), 
        return_sequences=True)
)
model_gru.add(
    GRU(100, activation='relu', return_sequences=False)
)
model_gru.add(Dropout(0.2))
model_gru.add(Dense(y_train.shape[1]))
model_gru.compile(optimizer='adam', loss='log_cosh')
model_gru.summary()

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10, 
    restore_best_weights=True
)
history1 = model_jaringan_lstm.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=150,
    batch_size=32,
    callbacks=[early_stop]
)

history_bilstm = model_bilstm.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=[early_stop]
)
history_gru = model_gru.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=170,
    batch_size=32,
    callbacks=[early_stop]
)

model_jaringan_lstm = tf.keras.models.load_model('E:/Kuliah/Artikel/Models/LSTM/model_lstm.h5')
model_bilstm = tf.keras.models.load_model('E:/Kuliah/Artikel/Models/BiLSTM/model_bilstm.h5')
model_gru = tf.keras.models.load_model('E:/Kuliah/Artikel/Models/GRU/model_gru.h5')

prediksi_lstm_test= [model_jaringan_lstm.predict(X_test_all[i]) for i in range(len(X_test_all))]
prediksi_bilstm_test= [model_bilstm.predict(X_test_all[i]) for i in range(len(X_test_all))]
prediksi_gru_test= [model_gru.predict(X_test_all[i]) for i in range(len(X_test_all))]

prediksi_lstm_test= [normalizer.inverse_transform_column(perusahaan[i], "Close", prediksi_lstm_test[i]) for i in range(len(X_test_all))]
prediksi_bilstm_test= [normalizer.inverse_transform_column(perusahaan[i], "Close", prediksi_bilstm_test[i]) for i in range(len(X_test_all))]
prediksi_gru_test= [normalizer.inverse_transform_column(perusahaan[i], "Close", prediksi_gru_test[i]) for i in range(len(X_test_all))]
asli_test = [normalizer.inverse_transform_column(perusahaan[i], "Close", y_test_all[i]) for i in range(len(X_test_all))]

def smape(y_true, y_pred):
    """Symmetric Mean Absolute Percentage Error."""
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    # hindari pembagian nol
    non_zero = denom != 0
    return 100 * np.mean(np.abs(y_pred[non_zero] - y_true[non_zero]) / denom[non_zero])
results = []

for i, company in enumerate(perusahaan):
    y_pred = normalizer.inverse_transform_column(company, "Close", prediksi_lstm[i]).flatten()
    y_true = normalizer.inverse_transform_column(company, "Close", y_test_all[i]).flatten()
    print(company)
    r2   = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    smp  = smape(y_true, y_pred)
    results.append({
        "Perusahaan": company,
        "R2":   r2,
        "RMSE": rmse,
        "sMAPE (%)": smp
    })
df_results = pd.DataFrame(results)
print(df_results)

results1 = []
for i, company in enumerate(perusahaan):
    y_pred = normalizer.inverse_transform_column(company, "Close", prediksi_bilstm[i]).flatten()
    y_true = normalizer.inverse_transform_column(company, "Close", y_test_all[i]).flatten()
    print(company)
    r2   = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    smp  = smape(y_true, y_pred)
    results1.append({
        "Perusahaan": company,
        "R2":   r2,
        "RMSE": rmse,
        "sMAPE (%)": smp
    })
df_results1 = pd.DataFrame(results1)
print(df_results1)
results2 = []

for i, company in enumerate(perusahaan):
    y_pred = normalizer.inverse_transform_column(company, "Close", prediksi_gru[i]).flatten()
    y_true = normalizer.inverse_transform_column(company, "Close", y_test_all[i]).flatten()
    print(company)
    r2   = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    smp  = smape(y_true, y_pred)
    results2.append({
        "Perusahaan": company,
        "R2":   r2,
        "RMSE": rmse,
        "sMAPE (%)": smp
    })
df_results2 = pd.DataFrame(results2)
print(df_results2)
