import os
import numpy as np
import pandas as pd
from pyedflib import EdfReader, highlevel
from datetime import date, timedelta, datetime
import pickle

def extract_info_from_edf(meta):
    try:
        # signals, signal_headers, meta = highlevel.read_edf(file_path)

        age = None
        sex = None

        if sex is None:
            sex = meta["sex"][0]

        birth_date = datetime.strptime(meta["birthdate"], "%d %b %Y")
        age = (meta["startdate"] - birth_date) / timedelta(days=365.2425)
        return sex, age
    except Exception as e:
        print(f"Ошибка при чтении пациента: {e}")
        return None, None

def is_amyloidosis_from_label(label):
    return int(label) == 1

def collect_statistics(X, y):
    sexes = []
    ages = []
    amyloidosis_count = 0

    for i, patient in enumerate(X):
        sex, age = extract_info_from_edf(y[1][i])
        if sex:
            sexes.append(sex)
        if age:
            ages.append(age)
        if is_amyloidosis_from_label(y[0][i]):
            amyloidosis_count += 1

    num_males = sexes.count("M")
    ages = np.array(ages)
    stats = {
        "Число пациентов (мужчин)": f"{len(sexes)}({num_males})",
        "Средний возраст (mean ± std)": [f"{ages.mean():.1f} ± {ages.std():.1f}" if len(ages) > 0 else "—"],
        "Число с амилоидозом": [amyloidosis_count]
    }

    return pd.DataFrame(stats)

# Пример использования
if __name__ == "__main__":
    import glob

    root_dir = "/home/kravchenko.artem/Projects/Diplomas/Data"  # Здесь должна быть общая папка, содержащая папки с файлами
    # edf_files_1 = glob.glob(os.path.join(root_dir, "Amy/Amy", "*.edf"), recursive=True)
    # edf_files_2 = glob.glob(os.path.join(root_dir, "AMY_add/AMY/2", "*.edf"), recursive=True)
    # edf_files_3 = glob.glob(os.path.join(root_dir, "AmyC/AmyC", "*.edf"), recursive=True)

    # all_edf = []
    # all_edf.extend(edf_files_1)
    # all_edf.extend(edf_files_2)
    # all_edf.extend(edf_files_3)
    
    with open("./dumped/X_train.pkl", "rb") as f:
        f.seek(0)
        X_train = pickle.load(f)
    with open("./dumped/y_train.pkl", "rb") as f:
        f.seek(0)
        y_train = pickle.load(f)
    with open("./dumped/X_test.pkl", "rb") as f:
        f.seek(0)
        X_test = pickle.load(f)
    with open("./dumped/y_test.pkl", "rb") as f:
        f.seek(0)
        y_test = pickle.load(f)

    df = collect_statistics(X_test, y_test)
    print(df.to_markdown(index=False))
