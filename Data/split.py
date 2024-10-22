import numpy as np
from collections import Counter
import os
import sys
from sklearn.model_selection import train_test_split
import pickle

# Подгружаем пути к директориям с нашими алгоритмами:
script_path = os.path.join(os.getcwd(), "../Scripts/")
print(script_path)
sys.path.append(script_path)
import data_generator as dgen

balance_by_class = 0
balance_by_train_test = 1

def split(X, y, strategy : int, test_size = 0.2):
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
            X_train, y_train_labels, y_train_metadata = zip(*train_data)
            X_test, y_test_labels, y_test_metadata = zip(*test_data)

            y_train = [y_train_labels, y_train_metadata]
            y_test = [y_test_labels, y_test_metadata]

            # Проверим распределение классов в обучающей и тестовой выборке
            unique_train, counts_train = np.unique(y_train[0], return_counts=True)
            unique_test, counts_test = np.unique(y_test[0], return_counts=True)

            print(f"Train class distribution: {dict(zip(unique_train, counts_train))}")
            print(f"Test class distribution: {dict(zip(unique_test, counts_test))}")



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
        args["filter"] = filter
        if bool(filter):
            hc = input("Enter highcut: (default 56) :")
            if hc:
                args["highcut"] = hc
    

    # channels = input("Input channels divided by comma: ")
    # ch = [part for part in channels.split(',') if part.strip()]

    # if channels:
    #     args["channel"] = ch

    dgen.init(*args)

    amy, amyc, norm, amy_h, amyc_h, norm_h = dgen.read_data_with_meta()

    strat = input("strategy 0 - balance by class \nstrategy 1 - balance by train test set\nEnter balance strategy: ")
    if strat == '':
        strat = 0

    test_sz = input("Enter test size: (default 0.2): ")

    tts_args = {}

    X = norm + amyc + amy
    y = np.concatenate([np.zeros(len(norm) + len(amyc)), np.ones(len(amy))])
    meta = np.concatenate([norm_h, amyc_h, amy_h])
    y = np.stack([y, meta], axis=0)
    
    tts_args["X"] = X
    tts_args["y"] = y
    tts_args["strategy"] = int(strat)
    if test_sz:
        tts_args["test_size"] = test_sz

    split(**tts_args)

