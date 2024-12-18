import numpy as np
from collections import Counter
import os
import sys
from sklearn.model_selection import train_test_split
import pickle
import random

# Подгружаем пути к директориям с нашими алгоритмами:
script_path = os.path.join(os.getcwd(), "../Scripts/")
print(script_path)
sys.path.append(script_path)
import data_generator as dgen

balance_by_class = 0
balance_by_train_test = 1
balance_by_real_distrib = 2

RANDOM_SEED = 42

real_norm_size = 40
real_amy_size = 8

def split(X, y, strategy : int, norm_len=None, amyc_len=None,test_size = 0.2):
    match strategy:
        case 0:
            class_counts = Counter(y[0])
            min_class_size = min(class_counts.values())  # Находим минимальное количество классов для балансировки

            # Создаем равномерную выборку (по минимальному количеству примеров каждого класса)
            balanced_signals = []
            balanced_labels = [[], []]
            for label in class_counts.keys():
                indices = np.where(y[0] == label)[0]
                selected_indices = np.random.choice(indices, min_class_size, replace=False)
                balanced_signals.extend([X[i] for i in selected_indices])
                balanced_labels[0].extend([y[0][i] for i in selected_indices])
                balanced_labels[1].extend([y[1][i] for i in selected_indices])

            # Создаем структуру для одновременного хранения сигналов, меток и метаданных
            data = list(zip(balanced_signals, balanced_labels[0], balanced_labels[1]))

            # Преобразуем в массив объектов, если необходимо
            data = np.array(data, dtype=object)

            # Разделяем данные на train и test с сохранением структуры
            train_data, test_data = train_test_split(
                data, test_size=test_size, stratify=balanced_labels[0], random_state=42
            )

            # Разворачиваем обратно сигналы, метки и метаданные
            X_train, y_train_labels, y_train_metadata = zip(*train_data)
            X_test, y_test_labels, y_test_metadata = zip(*test_data)

            y_train = [y_train_labels, y_train_metadata]
            y_test = [y_test_labels, y_test_metadata]

            # Проверим распределение классов в обучающей и тестовой выборке
            unique_train, counts_train = np.unique(y_train_labels, return_counts=True)
            unique_test, counts_test = np.unique(y_test_labels, return_counts=True)

            print(f"Train class distribution: {dict(zip(unique_train, counts_train))}")
            print(f"Test class distribution: {dict(zip(unique_test, counts_test))}")

        case 1:
            data = list(zip(X, y[0], y[1]))

            # Преобразуем в массив объектов, если необходимо
            data = np.array(data, dtype=object)

            # Разделяем данные на train и test с сохранением структуры
            train_data, test_data = train_test_split(
                data, test_size=test_size, stratify=y[0], random_state=42
            )

            # Разворачиваем обратно сигналы, метки и метаданные
            X_train, y_train_labels, y_train_metadata = list(list(zip(*train_data)))
            X_test, y_test_labels, y_test_metadata = list(zip(*test_data))

            y_train = [y_train_labels, y_train_metadata]
            y_test = [y_test_labels, y_test_metadata]

            # Проверим распределение классов в обучающей и тестовой выборке
            unique_train, counts_train = np.unique(y_train[0], return_counts=True)
            unique_test, counts_test = np.unique(y_test[0], return_counts=True)

            print(f"Train class distribution: {dict(zip(unique_train, counts_train))}")
            print(f"Test class distribution: {dict(zip(unique_test, counts_test))}")
        case 2:
            norm_val, norm_train = train_test_split(X[:norm_len], random_state=42, train_size=real_norm_size/ 2 / norm_len)
            amyc_val, amyc_train = train_test_split(X[norm_len:norm_len+amyc_len], random_state=42, train_size=real_norm_size /2 /amyc_len)
            amy_val, amy_train = train_test_split(X[norm_len+amyc_len:], random_state=42, train_size=real_amy_size / (len(X) - norm_len - amyc_len))
            
            metadata = list(zip(y[0], y[1]))
            norm_meta_val, norm_meta_train = train_test_split(metadata[:norm_len], random_state=42, train_size=real_norm_size/ 2 / norm_len)
            amyc_meta_val, amyc_meta_train = train_test_split(metadata[norm_len:norm_len+amyc_len], random_state=42, train_size=real_norm_size /2 /amyc_len)
            amy_meta_val, amy_meta_train = train_test_split(metadata[norm_len+amyc_len:], random_state=42, train_size=real_amy_size / (len(X) - norm_len - amyc_len))
            
            norm_meta_train = list(list(zip(*norm_meta_train)))
            amyc_meta_train = list(list(zip(*amyc_meta_train)))
            amy_meta_train = list(list(zip(*amy_meta_train)))
            
            norm_meta_val = list(list(zip(*norm_meta_val)))
            amyc_meta_val = list(list(zip(*amyc_meta_val)))
            amy_meta_val = list(list(zip(*amy_meta_val)))

            
            X_train = np.concatenate([norm_train, amyc_train, amy_train])
            X_test = np.concatenate([norm_val, amyc_val, amy_val])

            y_train = np.concatenate([norm_meta_train, amyc_meta_train, amy_meta_train], axis=1)
            y_test = np.concatenate([norm_meta_val, amyc_meta_val, amy_meta_val], axis=1)
            
            # Проверим распределение классов в обучающей и тестовой выборке
            unique_train, counts_train = np.unique(y_train[0], return_counts=True)
            unique_test, counts_test = np.unique(y_test[0], return_counts=True)

            print(f"Train class distribution: {dict(zip(unique_train, counts_train))}")
            print(f"Test class distribution: {dict(zip(unique_test, counts_test))}")
        case 3:

            norm_val, norm_train = train_test_split(X[:norm_len], random_state=42, train_size=real_norm_size/ 2 / norm_len)
            amyc_val, amyc_train = train_test_split(X[norm_len:norm_len+amyc_len], random_state=42, train_size=real_norm_size /2 /amyc_len)
            amy_val, amy_train = train_test_split(X[norm_len+amyc_len:], random_state=42, train_size=real_amy_size / (len(X) - norm_len - amyc_len))
            
            # max_train_size = min(len(norm_train), len(amyc_train), len(amy_train))
            max_train_size = len(amy_train)
            random.seed(RANDOM_SEED)
            norm_train = random.choices(norm_train, k=int(max_train_size / 2))
            random.seed(RANDOM_SEED)
            amyc_train = random.choices(amyc_train, k=int(max_train_size / 2))
            random.seed(RANDOM_SEED)
            amy_train = random.choices(amy_train, k=max_train_size)
            
            metadata = list(zip(y[0], y[1]))
            norm_meta_val, norm_meta_train = train_test_split(metadata[:norm_len], random_state=42, train_size=real_norm_size/ 2 / norm_len)
            amyc_meta_val, amyc_meta_train = train_test_split(metadata[norm_len:norm_len+amyc_len], random_state=42, train_size=real_norm_size /2 /amyc_len)
            amy_meta_val, amy_meta_train = train_test_split(metadata[norm_len+amyc_len:], random_state=42, train_size=real_amy_size / (len(X) - norm_len - amyc_len))
            
            random.seed(RANDOM_SEED)
            norm_meta_train = random.choices(norm_meta_train, k=int(max_train_size/2))
            random.seed(RANDOM_SEED)
            amyc_meta_train = random.choices(amyc_meta_train, k=int(max_train_size/2))
            random.seed(RANDOM_SEED)
            amy_meta_train = random.choices(amy_meta_train, k=max_train_size)
            
            norm_meta_train = list(list(zip(*norm_meta_train)))
            amyc_meta_train = list(list(zip(*amyc_meta_train)))
            amy_meta_train = list(list(zip(*amy_meta_train)))
            
            norm_meta_val = list(list(zip(*norm_meta_val)))
            amyc_meta_val = list(list(zip(*amyc_meta_val)))
            amy_meta_val = list(list(zip(*amy_meta_val)))
            
            X_train = np.concatenate([norm_train, amyc_train, amy_train])
            X_test = np.concatenate([norm_val, amyc_val, amy_val])

            y_train = np.concatenate([norm_meta_train, amyc_meta_train, amy_meta_train], axis=1)
            y_test = np.concatenate([norm_meta_val, amyc_meta_val, amy_meta_val], axis=1)
            
            # Проверим распределение классов в обучающей и тестовой выборке
            unique_train, counts_train = np.unique(y_train[0], return_counts=True)
            unique_test, counts_test = np.unique(y_test[0], return_counts=True)

            print(f"Train class distribution: {dict(zip(unique_train, counts_train))}")
            print(f"Test class distribution: {dict(zip(unique_test, counts_test))}")
            

    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    
    print(X_train.shape)

    # Сохранение данных в файлы numpy
    with open('dumped/X_train.pkl', 'wb') as f:
        pickle.dump(X_train, f)
    with open('dumped/y_train.pkl', 'wb') as f:
        pickle.dump(y_train, f)
    with open('dumped/X_test.pkl', 'wb') as f:
        pickle.dump(X_test, f)
    with open('dumped/y_test.pkl', 'wb') as f:
        pickle.dump(y_test, f)


    print("Данные успешно сохранены!")

if __name__ == "__main__":
    args = {}
    filter = input("Enable filtering: (1/0) ")
    if filter:
        args["filter"] = bool(filter)
        if bool(filter):
            hc = input("Enter highcut: (default 56) :")
            if hc:
                args["hcut"] = int(hc)
            else:
                args["hcut"] = int(56)
    

    # channels = input("Input channels divided by comma: ")
    # ch = [part for part in channels.split(',') if part.strip()]

    # if channels:
    #     args["channel"] = ch\
    

    dgen.init(**args)

    amy, amyc, norm, amy_h, amyc_h, norm_h = dgen.read_data_with_meta()

    strat = input("strategy 0 - balance by class \n"
                  "strategy 1 - balance by train test set\n"
                  "strategy 2 - balance by real distribution in val, and rest in train\n"
                  "strategy 3 - balance by real distribution in val, and equal classes in train\n"
                  "Enter balance strategy: ")
    if strat == '':
        strat = 0

    test_sz = input("Enter test size: (default 0.2): ")

    tts_args = {}
    
    X = norm + amyc + amy
    y = np.concatenate([np.zeros(len(norm) + len(amyc)), np.ones(len(amy))])
    meta = np.concatenate([norm_h, amyc_h, amy_h])
    y = np.stack([y, meta], axis=0)
    
    if strat == '2' or strat == '3':
        tts_args["norm_len"] = len(norm)
        tts_args["amyc_len"] = len(amyc)
    
    tts_args["X"] = X
    tts_args["y"] = y
    tts_args["strategy"] = int(strat)
    if test_sz:
        tts_args["test_size"] = test_sz

    split(**tts_args)

