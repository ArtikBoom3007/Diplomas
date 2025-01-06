import os
import pandas as pd
import numpy as np
import json
import pickle
import gc
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
test_size = 0.5  # Доля данных в тестовой выборке
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


# Создание подвыборок для каждой конфигурации
split_data = {"train": {}, "test": {}}

train_files, test_files = train_test_split(
    df_labels,
    test_size=test_size,
    stratify=df_labels["class"],
    random_state=random_state,
)

# Фильтруем классы для текущей конфигурации
# train_1 = train_files[train_files["class"] == 0].sample(n=count_class_0, random_state=random_state)
train_1 = train_files.copy()
one_class_count = train_files[train_files["class"] == 1].shape[0]
train_2_first = train_files[train_files["class"] == 0].sample(
    n=one_class_count, random_state=random_state
)
train_2_second = train_files[train_files["class"] == 1].copy()
train_2 = pd.concat([train_2_first, train_2_second])
split_data["train"]["train_1"] = train_1
split_data["train"]["train_2"] = train_2

# test_0 = test_files[test_files["class"] == 0].sample(n=count_class_0, random_state=random_state)
test_1 = test_files.copy()
one_class_count = test_files[test_files["class"] == 1].shape[0]
test_2_first = test_files[test_files["class"] == 0].sample(
    n=one_class_count, random_state=random_state
)
test_2_second = test_files[test_files["class"] == 1].copy()
test_2 = pd.concat([test_2_first, test_2_second])
split_data["test"]["test_1"] = test_1
split_data["test"]["test_2"] = test_2

# Гарантируем отсутствие пересечений
for train_key in [k for k in split_data["train"].keys() if "train" in k]:
    for test_key in [k for k in split_data["test"].keys() if "test" in k]:
        train_filenames = set(split_data["train"][train_key]["Анонимизированный EDF"])
        test_filenames = set(split_data["test"][test_key]["Анонимизированный EDF"])
        if not train_filenames.isdisjoint(test_filenames):
            raise ValueError(
                f"Train and Test sets overlap in {train_key} and {test_key}."
            )


# Шаг 5: Организация папок для train-test
train_folder = os.path.join(extract_folder, "train")
test_folder = os.path.join(extract_folder, "test")
os.makedirs(train_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)


def copy_files(file_list, target_folder):
    bar = IncrementalBar(f"Copying files", max=len(file_list))
    for _, row in file_list.iterrows():
        src_path = os.path.join(edf_folder, (row["Анонимизированный EDF"] + ".edf"))
        dst_path = os.path.join(target_folder, (row["Анонимизированный EDF"] + ".edf"))
        shutil.copy(src_path, dst_path)
        bar.next()
    bar.finish()


print("Copying files to dst folders!")
copy_files(train_files, train_folder)
copy_files(test_files, test_folder)

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

signal_len = 5000


def crop(data: list):
    parts = []
    for record in data:
        record = record[:, :5000]
        for i in range(int(record.shape[1] / signal_len)):
            parts.append(np.array(record[:, i * signal_len : (i + 1) * signal_len]))
    return parts


def highpass(highcut, order, fs):
    nyq = 0.5 * fs
    high = highcut / nyq
    b, a = butter(order, high, btype="low")
    return b, a


make_filter = True
highcut = 60


def final_filter(data, fs, order=4):
    if make_filter:
        b, a = highpass(highcut, order, fs)
        x = lfilter(b, a, data, axis=1)
        return x
    return data


for config_name, temp_train_files in split_data["train"].items():

    X_train = []
    y_train = []
    y_meta = []

    bar = IncrementalBar(f"Train config {config_name}", max=len(temp_train_files))

    train_itoname_map = {}
    badass_data_names = []

    for _, row in temp_train_files.iterrows():
        name = row["Анонимизированный EDF"] + ".edf"
        full_pth = os.path.join(train_folder, name)
        signals, signal_headers, meta = highlevel.read_edf(full_pth)
        cropped_value = crop([signals])

        null_filtered = []
        for value in cropped_value:
            value = np.array([value[i] - np.min(value[i], axis=0) for i in range(8)])
            maxes = np.array([np.max(value[i]).round(3) for i in range(8)])
            maxes[maxes < 0.1] = 0
            if 0.0 not in maxes:
                null_filtered.append(value)
            else:
                badass_data_names.append(name)
        if null_filtered:
            X_train.extend(null_filtered)
            train_itoname_map[int(len(X_train) - 1)] = name
            y_train.extend(np.ones(len(null_filtered), dtype=np.int8) * row["class"])
            y_meta.extend([meta])
        bar.next()
    bar.finish()

    train_itoname_map["badass_data"] = badass_data_names
    with open(f"Data/dumped/train_itoname_{config_name}.json", "w") as outfile:
        json.dump(train_itoname_map, outfile)

    print(f"Train files: {len(train_files)}")

    unique_train, counts_train = np.unique(y_train, return_counts=True)
    train_dict = {
        int(key): int(value)
        for key, value in dict(zip(unique_train, counts_train)).items()
    }

    print(f"Train for config {config_name} class distribution: {train_dict}")
    gc.collect()

    y_train = [y_train, y_meta]

    X_train = np.array(X_train, dtype=np.float64)
    y_train = np.array(y_train)

    with open(f"./Data/dumped/X_train_fraction_{config_name}.pkl", "wb") as f:
        pickle.dump(X_train, f)
    with open(f"./Data/dumped/y_train_fraction_{config_name}.pkl", "wb") as f:
        pickle.dump(y_train, f)


for config_name, temp_test_files in split_data["test"].items():

    X_test = []
    y_test = []
    y_meta = []

    test_itoname_map = {}
    badass_data_names = []

    bar = IncrementalBar(f"Test config {config_name}", max=len(temp_test_files))

    for _, row in temp_test_files.iterrows():
        name = row["Анонимизированный EDF"] + ".edf"
        full_pth = os.path.join(test_folder, name)
        signals, signal_headers, meta = highlevel.read_edf(full_pth)
        cropped_value = crop([signals])
        null_filtered = []
        for value in cropped_value:
            value = np.array([value[i] - np.min(value[i], axis=0) for i in range(8)])
            maxes = np.array([np.max(value[i], axis=0).round(3) for i in range(8)])
            maxes[maxes < 0.1] = 0
            if 0.0 not in maxes:
                null_filtered.append(value)
            else:
                badass_data_names.append(name)
        if null_filtered:
            X_test.extend(null_filtered)
            test_itoname_map[int(len(X_test) - 1)] = name
            y_test.extend(np.ones(len(null_filtered), dtype=np.int8) * row["class"])
            y_meta.extend([meta])
        bar.next()
    bar.finish()

    test_itoname_map["badass_data"] = badass_data_names
    with open(f"Data/dumped/test_itoname_{config_name}.json", "w") as outfile:
        json.dump(test_itoname_map, outfile)

    print(f"Test files: {len(test_files)}")
    unique_test, counts_test = np.unique(y_test, return_counts=True)
    test_dict = {
        int(key): int(value)
        for key, value in dict(zip(unique_test, counts_test)).items()
    }
    print(f"Test for config {config_name} class distribution: {train_dict}")
    gc.collect()
    y_test = [y_test, y_meta]

    X_test = np.array(X_test, dtype=np.float64)
    y_test = np.array(y_test)

    with open(f"./Data/dumped/X_test_fraction_{config_name}.pkl", "wb") as f:
        pickle.dump(X_test, f)
    with open(f"./Data/dumped/y_test_fraction_{config_name}.pkl", "wb") as f:
        pickle.dump(y_test, f)


print(f"Train-test files saved in {extract_folder}")
print(f"Dumped Train-test split saved in Data/dumped/")
