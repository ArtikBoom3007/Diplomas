import numpy as np
from collections import Counter
import os
import sys
from sklearn.model_selection import train_test_split

# Подгружаем пути к директориям с нашими алгоритмами:
script_path = os.path.join(os.getcwd(), "../Scripts/")
sys.path.append(script_path)
import data_generator as dgen

balance_by_class = 0
balance_by_train_test = 1

def split(X, y, strategy : int, test_size = 0.2):
    match strategy:
        case 0:
            class_counts = Counter(y)
            min_class_size = min(class_counts.values())  # Находим минимальное количество классов для балансировки

            # Создаем равномерную выборку (по минимальному количеству примеров каждого класса)
            balanced_signals = []
            balanced_labels = []
            for label in class_counts.keys():
                indices = np.where(y == label)[0]
                selected_indices = np.random.choice(indices, min_class_size, replace=False)
                balanced_signals.extend([X[i] for i in selected_indices])
                balanced_labels.extend([y[i] for i in selected_indices])

            # Преобразуем списки обратно в numpy объекты для сохранения
            balanced_signals = np.array(balanced_signals, dtype=object)
            balanced_labels = np.array(balanced_labels)

            # Разделение на обучающую и тестовую выборки
            X_train, X_test, y_train, y_test = train_test_split(
                balanced_signals, balanced_labels, test_size=test_size, stratify=balanced_labels, random_state=42
            )

            # Проверим распределение классов в обучающей и тестовой выборке
            unique_train, counts_train = np.unique(y_train, return_counts=True)
            unique_test, counts_test = np.unique(y_test, return_counts=True)

            print(f"Train class distribution: {dict(zip(unique_train, counts_train))}")
            print(f"Test class distribution: {dict(zip(unique_test, counts_test))}")

        case 1:
            # Стратифицированное разбиение на обучающую и тестовую выборки
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)

            # Проверим распределение классов в обучающей и тестовой выборке
            unique_train, counts_train = np.unique(y_train, return_counts=True)
            unique_test, counts_test = np.unique(y_test, return_counts=True)

            print(f"Train class distribution: {dict(zip(unique_train, counts_train))}")
            print(f"Test class distribution: {dict(zip(unique_test, counts_test))}")

            X_train = np.array(X_train, dtype=object)
            X_test = np.array(X_test, dtype=object)



    # Сохранение данных в файлы numpy
    np.save('dumped/X_train.npy', X_train)
    np.save('dumped/y_train.npy', y_train)
    np.save('dumped/X_test.npy', X_test)
    np.save('dumped/y_test.npy', y_test)

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
    

    channels = input("Input channels divided by comma: ")
    ch = [part for part in channels.split(',') if part.strip()]

    if channels:
        args["channel"] = ch

    dgen.init(*args)

    amy, amyc, norm = dgen.read_data()

    strat = input("strategy 0 - balance by class \nstrategy 1 - balance by train test set\nEnter balance strategy: ")
    test_sz = input("Enter test size: (default 0.2): ")

    tts_args = {}

    X = norm + amyc + amy
    y = np.concatenate([np.zeros(len(norm) + len(amyc)), np.ones(len(amy))])
    
    tts_args["X"] = X
    tts_args["y"] = y
    tts_args["strategy"] = int(strat)
    if test_sz:
        tts_args["test_size"] = test_sz

    split(**tts_args)

    

