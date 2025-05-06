import os
import pandas as pd
import numpy as np
import json
import pickle
import gc
from sklearn.model_selection import train_test_split
import shutil
from progress.bar import IncrementalBar
from pathlib import Path
from pyedflib import highlevel
from scipy.signal import butter, lfilter


# Параметры
threshold = 50  # Порог для определения метки класса
test_size = 0.5  # Доля данных в тестовой выборке
random_state = 42  # Для воспроизводимости

# Шаг 2: Чтение таблицы меток
sdla30_filenames = [x.__str__() for x in Path("./PulmHypert").glob("SDLA30*.edf")]
labels = [0] * len(sdla30_filenames)
sdla50_filenames = [x.__str__() for x in Path("./PulmHypert").glob("SDLA50*.edf")]
labels.extend([1] * len(sdla50_filenames))

df_labels = pd.DataFrame({"filename" : sdla30_filenames + sdla50_filenames, "class": labels})
print(f"Loaded {len(df_labels)} file labels."
      f"There are class 0: {len(sdla30_filenames)} and class 1: {len(sdla50_filenames)}")


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
        train_filenames = set(split_data["train"][train_key]["filename"])
        test_filenames = set(split_data["test"][test_key]["filename"])
        if not train_filenames.isdisjoint(test_filenames):
            raise ValueError(
                f"Train and Test sets overlap in {train_key} and {test_key}."
            )

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
        name = row["filename"]
        full_pth = os.path.join(os.getcwd(), name)
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
    with open(f"./dumped/pulm_train_itoname_{config_name}.json", "w") as outfile:
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

    with open(f"./dumped/X_train_pulm_hypertension_{config_name}.pkl", "wb") as f:
        pickle.dump(X_train, f)
    with open(f"./dumped/y_train_pulm_hypertension_{config_name}.pkl", "wb") as f:
        pickle.dump(y_train, f)


for config_name, temp_test_files in split_data["test"].items():

    X_test = []
    y_test = []
    y_meta = []

    test_itoname_map = {}
    badass_data_names = []

    bar = IncrementalBar(f"Test config {config_name}", max=len(temp_test_files))

    for _, row in temp_test_files.iterrows():
        name = row["filename"]
        full_pth = os.path.join(os.getcwd(), name)
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
    with open(f"./dumped/pulm_test_itoname_{config_name}.json", "w") as outfile:
        json.dump(test_itoname_map, outfile)

    print(f"Test files: {len(test_files)}")
    unique_test, counts_test = np.unique(y_test, return_counts=True)
    test_dict = {
        int(key): int(value)
        for key, value in dict(zip(unique_test, counts_test)).items()
    }
    print(f"Test for config {config_name} class distribution: {test_dict}")
    gc.collect()
    y_test = [y_test, y_meta]

    X_test = np.array(X_test, dtype=np.float64)
    y_test = np.array(y_test)

    with open(f"./dumped/X_test_pulm_hypertension_{config_name}.pkl", "wb") as f:
        pickle.dump(X_test, f)
    with open(f"./dumped/y_test_pulm_hypertension_{config_name}.pkl", "wb") as f:
        pickle.dump(y_test, f)


print(f"Dumped Train-test split saved in Data/dumped/")
