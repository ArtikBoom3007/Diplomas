import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import shutil
from progress.bar import IncrementalBar
from pyedflib import highlevel
from scipy.signal import butter, lfilter


# Параметры
archive_folder = "./Data/EF_11k_approx/EDFs"  # Папка с архивами .7z
extract_folder = "./Data/EF_11k_approx/extracted"  # Папка для извлечения данных
edf_folder = os.path.join(extract_folder, "edf_files")  # Папка для EDF-файлов
label_table_path = (
    "Data/EF_11k_approx/Test_Group_List_short.xlsx"  # Путь к таблице с метками
)
threshold = 50  # Порог для определения метки класса
test_size = 0.2  # Доля данных в тестовой выборке
random_state = 42  # Для воспроизводимости

# Шаг 2: Чтение таблицы меток
df_labels = pd.read_excel(label_table_path)
df_labels["class"] = (df_labels["ФВ"] <= threshold).astype(int)  # Определение классов
print(f"Loaded {len(df_labels)} file labels.")

# Шаг 3: Проверка наличия соответствующих файлов EDF
edf_files = [f for f in os.listdir(edf_folder) if f.endswith(".edf")]
df_labels = df_labels[
    (df_labels["Анонимизированный EDF"] + ".edf").isin(edf_files)
].reset_index(drop=True)
print(f"Found {len(df_labels)} matching EDF files.")

# Шаг 4: Разбиение на train-test
train_files, test_files = train_test_split(
    df_labels,
    test_size=test_size,
    stratify=df_labels["class"],
    random_state=random_state,
)

# Шаг 5: Организация папок для train-test
train_folder = os.path.join(extract_folder, "train")
test_folder = os.path.join(extract_folder, "test")
os.makedirs(train_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)


def copy_files(file_list, target_folder):
    for _, row in file_list.iterrows():
        src_path = os.path.join(edf_folder, (row["Анонимизированный EDF"] + ".edf"))
        dst_path = os.path.join(target_folder, (row["Анонимизированный EDF"] + ".edf"))
        shutil.copy(src_path, dst_path)


# copy_files(train_files, train_folder)
# copy_files(test_files, test_folder)

# vad_folder = os.path.join(extract_folder, "vad")
# os.makedirs(vad_folder, exist_ok=True)
# bar = IncrementalBar("VAD", max=len(train_files) + len(test_files))


# def make_vad_distrib(file_list, target_folder):
#     for _, row in file_list.iterrows():
#         src_path = os.path.join(edf_folder, (row["Анонимизированный EDF"] + ".edf"))
#         dst_path = os.path.join(target_folder, (row["Анонимизированный EDF"] + ".edf"))
#         if row["class"] == 0:
#             shutil.copy(src_path, dst_path)
#         bar.next()


# bar.finish()


# make_vad_distrib(train_files, vad_folder)
# make_vad_distrib(test_files, vad_folder)

# signals, signal_headers, header = highlevel.read_edf(os.path.join(amy_path, "Amy1.edf"), ch_names=['ECG I'])
# print(signal_headers)
# print(header)

signal_len = 1000


def crop(data: list):
    parts = []
    for record in data:
        for i in range(int(record.shape[1] / signal_len)):
            parts.append(np.array(record[:, i * signal_len : (i + 1) * signal_len]))
    return parts

def highpass(highcut, order, fs):
    nyq = 0.5 * fs
    high = highcut / nyq
    b, a = butter(order, high, btype='low')
    return b, a

make_filter = True
highcut = 60
def final_filter(data, fs, order=4):
    if make_filter:
        b, a = highpass(highcut, order, fs)
        x = lfilter(b, a, data, axis=1)
        return x
    return data


X_train = []
y_train = []

bar = IncrementalBar("Train", max=len(train_files))

for _, row in train_files.iterrows():
    name = row["Анонимизированный EDF"] + ".edf"
    full_pth = os.path.join(train_folder, name)
    signals, signal_headers, _ = highlevel.read_edf(full_pth)
    cropped_value = crop([signals])

    null_filtered = []
    for value in cropped_value:
        value = np.array([value[i] - np.min(value[i], axis=0) for i in range(8)])
        maxes = np.array([np.max(value[i]).round(3) for i in range(8)])
        maxes[maxes < 0.1] = 0
        if 0.0 not in maxes:
            null_filtered.append(value)
    X_train.extend(null_filtered)
    y_train.extend(np.ones(len(null_filtered), dtype=np.int8) * row["class"])
    bar.next()
bar.finish()

X_test = []
y_test = []


bar = IncrementalBar("Test", max=len(test_files))

for _, row in test_files.iterrows():
    name = row["Анонимизированный EDF"] + ".edf"
    full_pth = os.path.join(test_folder, name)
    signals, signal_headers, _ = highlevel.read_edf(full_pth)
    cropped_value = crop([signals])
    null_filtered = []
    for value in cropped_value:
        value = np.array([value[i] - np.min(value[i], axis=0) for i in range(8)])
        maxes = np.array([np.max(value[i], axis=0).round(3) for i in range(8)])
        maxes[maxes < 0.1] = 0
        if 0.0 not in maxes:
            null_filtered.append(value)
    X_test.extend(null_filtered)
    y_test.extend(np.ones(len(null_filtered), dtype=np.int8) * row["class"])
    bar.next()
bar.finish()

print(f"Train files: {len(train_files)}")
print(f"Test files: {len(test_files)}")
print(f"Train-test split saved in {extract_folder}")

import pickle
import gc

gc.collect()

X_train = np.array(X_train, dtype=np.float16)
X_test = np.array(X_test, dtype=np.float16)
y_train = np.array(y_train, dtype=np.int8)
y_test = np.array(y_test, dtype=np.int8)

print(X_train.shape)

# Сохранение данных в файлы numpy
with open("./Data/dumped/X_train_fraction.pkl", "wb") as f:
    pickle.dump(X_train, f)
with open("./Data/dumped/y_train_fraction.pkl", "wb") as f:
    pickle.dump(y_train, f)
with open("./Data/dumped/X_test_fraction.pkl", "wb") as f:
    pickle.dump(X_test, f)
with open("./Data/dumped/y_test_fraction.pkl", "wb") as f:
    pickle.dump(y_test, f)
