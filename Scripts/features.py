import numpy as np
import pandas as pd
import neurokit2 as nk
from shapely.geometry import Polygon
import math
import datetime

SAMPLING_RATE = 500 

def calculate_area(points):
    polygon = Polygon(points)
    return polygon.area

def loop(df_term):
    points_frontal = list(zip(df_term['y'], df_term['z']))
    points_sagittal = list(zip(df_term['x'], df_term['z']))
    points_axial = list(zip(df_term['y'], df_term['x']))
    area_frontal = calculate_area(points_frontal)
    area_sagittal = calculate_area(points_sagittal)
    area_axial = calculate_area(points_axial)
    return area_frontal, area_sagittal, area_axial

def find_qrst_angle(mean_qrs, mean_t):
    mean_qrs = np.array(mean_qrs)
    mean_t = np.array(mean_t)
    dot_product = np.dot(mean_qrs, mean_t)
    norm_qrs = np.linalg.norm(mean_qrs)
    norm_t = np.linalg.norm(mean_t)
    angle_radians = np.arccos(dot_product / (norm_qrs * norm_t))
    return np.degrees(angle_radians)

def eccentricity_loop(x, y):
    cov_matrix = np.cov(x, y)
    eigenvalues, _ = np.linalg.eig(cov_matrix)
    return np.sqrt(1 - (min(eigenvalues) / max(eigenvalues)))

def calculate_velocity(df_term):
    dx = np.diff(df_term['x'])
    dy = np.diff(df_term['y'])
    dz = np.diff(df_term['z'])
    dt = np.diff(df_term['time'])
    velocity = np.sqrt(dx**2 + dy**2 + dz**2) / dt
    return np.mean(velocity), np.max(velocity)

def calculate_wave_durations(waves_peak):
    durations = {}
    if 'ECG_R_Peaks' in waves_peak:
        r_peaks = waves_peak['ECG_R_Peaks']
        rr_intervals = np.diff(r_peaks) / SAMPLING_RATE
        durations['RR_interval_median'] = np.median(rr_intervals)
    if 'ECG_T_Offsets' in waves_peak and 'ECG_Q_Peaks' in waves_peak:
        q_peaks = waves_peak['ECG_Q_Peaks']
        t_offsets = waves_peak['ECG_T_Offsets']
        if len(q_peaks) == len(t_offsets):
            qrs_duration = [(t - q) / SAMPLING_RATE for t, q in zip(t_offsets, q_peaks)]
            durations['QRS_duration'] = np.median(qrs_duration) 
    return durations

def make_vecg(df_term):
    try:
        DI = df_term.get('ECG I')
        DII = df_term.get('ECG II')
        V1 = df_term.get('ECG V1')
        V2 = df_term.get('ECG V2')
        V3 = df_term.get('ECG V3')
        V4 = df_term.get('ECG V4')
        V5 = df_term.get('ECG V5')
        V6 = df_term.get('ECG V6')

        if DI is None or DII is None or any(v is None for v in [V1, V2, V3, V4, V5, V6]):
            raise KeyError("Недостаточно отведений для расчета ВЭКГ")

        df_term['x'] = -(-0.172*V1 - 0.074*V2 + 0.122*V3 + 0.231*V4 + 0.239*V5 + 0.194*V6 + 0.156*DI - 0.01*DII)
        df_term['y'] = (0.057*V1 - 0.019*V2 - 0.106*V3 - 0.022*V4 + 0.041*V5 + 0.048*V6 - 0.227*DI + 0.887*DII)
        df_term['z'] = -(-0.229*V1 - 0.31*V2 - 0.246*V3 - 0.063*V4 + 0.055*V5 + 0.108*V6 + 0.022*DI + 0.102*DII)

    except KeyError as e:
        print(f"Ошибка: отсутствует сигнал для вычисления координат ВЭКГ: {e}")
        return df_term

    return df_term

def extract_features(x_data, y_metadata):
    """
    Основная функция для извлечения признаков из массива сигналов.
    
    :param x_data: массив сигналов (например, x_train)
    :param y_labels: метки классов
    :param y_metadata: метаинформация
    :return: DataFrame с извлеченными признаками
    """
    features_list = []

    for i, signal in enumerate(x_data):
        try:
            metadata = y_metadata[i]
            male = 1 if metadata.get('sex') == 'Male' else 0
            female = 1 if metadata.get('sex') == 'Female' else 0
            today = metadata.get('startdate')
            born = datetime.datetime.strptime(metadata.get('birthdate'), "%d %b %Y")
            age = today.year - born.year - ((today.month, today.day) < (born.month, born.day))
            admin_code = metadata.get('admincode', 'Unknown')

            # Инициализация признаков
            features = {
                'Male': male,
                'Female': female,
                'Age': age,
                'Admin_Code': admin_code,
                'Square_QRS_frontal': 0,
                'Square_QRS_sagittal': 0,
                'Square_QRS_axial': 0,
                'Frontal_Angle_QRST': 0,
                'Eccentricity_QRS': 0,
                'Mean_Velocity': 0,
                'Max_Velocity': 0
            }

            # Преобразование сигнала в DataFrame
            df_term = pd.DataFrame(signal.transpose(), columns=['ECG I', 'ECG II', 'ECG V1', 'ECG V2', 'ECG V3', 'ECG V4', 'ECG V5', 'ECG V6'])
            df_term['time'] = np.arange(len(df_term)) * (1 / SAMPLING_RATE)

            # Преобразование ЭКГ в ВЭКГ
            df_term = make_vecg(df_term)

            # Очистка сигнала и обнаружение пиков
            cleaned_signal = nk.ecg_clean(df_term['ECG I'], sampling_rate=SAMPLING_RATE)
            _, rpeaks = nk.ecg_peaks(cleaned_signal, sampling_rate=SAMPLING_RATE)
            _, waves_peak = nk.ecg_delineate(cleaned_signal, rpeaks, sampling_rate=SAMPLING_RATE, method="peak")

            # Рассчитать площади петель и углы
            start = rpeaks['ECG_R_Peaks'][0]
            areas_qrs = loop(df_term.iloc[start:, :])
            features['Square_QRS_frontal'] = areas_qrs[0]
            features['Square_QRS_sagittal'] = areas_qrs[1]
            features['Square_QRS_axial'] = areas_qrs[2]

            # Извлечение QRS и T векторов
            #t_offsets = waves_peak.get('ECG_T_Offsets', [])

            start_qrs = max(0, rpeaks['ECG_R_Peaks'][0] - 30)  # Используем 30 семплов перед R-пиком
            end_qrs = rpeaks['ECG_R_Peaks'][0]
            start_t = rpeaks['ECG_R_Peaks'][0]
            end_t = start_t + 100
            
            mean_qrs = [
                df_term['x'][start_qrs:end_qrs].mean(),
                df_term['y'][start_qrs:end_qrs].mean(),
                df_term['z'][start_qrs:end_qrs].mean()
            ]

            mean_t = [
                df_term['x'][start_t:end_t].mean(),
                df_term['y'][start_t:end_t].mean(),
                df_term['z'][start_t:end_t].mean()
            ]

            cos_theta = np.dot(mean_qrs, mean_t) / (np.linalg.norm(mean_qrs) * np.linalg.norm(mean_t))
            features['Frontal_Angle_QRST'] = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))

            # Дополнительные признаки
            features['Eccentricity_QRS'] = eccentricity_loop(df_term['x'], df_term['y'])
            mean_velocity, max_velocity = calculate_velocity(df_term)
            features['Mean_Velocity'] = mean_velocity
            features['Max_Velocity'] = max_velocity

            durations = calculate_wave_durations(waves_peak)
            features.update(durations)

            features_list.append(features)
        except Exception as e:
            print(f"Ошибка при обработке сигнала {i}: {e}")

    return pd.DataFrame(features_list)