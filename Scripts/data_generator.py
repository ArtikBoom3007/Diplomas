import os
import sys
import inspect

from pyedflib import highlevel
from scipy.signal import butter, lfilter
import numpy as np

current_module = sys.modules[__name__]
module_path = os.path.dirname(inspect.getfile(current_module))

def init(siglen = 750, filter=False, hcut=56, channel : list = ['ECG I']):
    global signal_len, make_filter, highcut, ecg_channel
    signal_len = siglen
    make_filter = filter
    highcut = hcut
    ecg_channel = channel

def normalize(Y):
    X = Y.copy()
    X -= X.mean()
    return X

def highpass(highcut, order, fs):
    nyq = 0.5 * fs
    high = highcut / nyq
    b, a = butter(order, high, btype='low')
    return b, a

def final_filter(data, fs, order=4):
    if make_filter:
        b, a = highpass(highcut, order, fs)
        x = lfilter(b, a, data)
        return x
    return data

def read_data():
    global sample_rate, amy_path, amyc_path, norm_path
    
    # If this module starts from different places. Make path consistency.
    path = os.path.join(module_path, "../Data")

    amy_path = os.path.join(path, "Amy/Amy")
    amyc_path = os.path.join(path, "AmyC/AmyC")
    norm_path = os.path.join(path, "AMY_add/AMY/2")

    amy = []
    amyc = []
    norm = []

    #signals, signal_headers, header = highlevel.read_edf(os.path.join(amy_path, "Amy1.edf"), ch_names=['ECG I'])
    # print(signal_headers)
    # print(header)

    for name in os.listdir(amy_path):
        signals, signal_headers, _ = highlevel.read_edf(os.path.join(amy_path, name), ch_names=ecg_channel)

        signal = signals[0]
        amy.append(final_filter(signal, signal_headers[0]["sample_frequency"]))  

    for name in os.listdir(amyc_path):
        signals, signal_headers, _ = highlevel.read_edf(os.path.join(amyc_path, name), ch_names=ecg_channel)

        signal = signals[0]
        amyc.append(final_filter(signal, signal_headers[0]["sample_frequency"]))  

    for name in os.listdir(norm_path):
        signals, signal_headers, _ = highlevel.read_edf(os.path.join(norm_path, name), ch_names=ecg_channel)

        signal = signals[0]
        norm.append(final_filter(signal, signal_headers[0]["sample_frequency"]))  

    sample_rate = signal_headers[0]["sample_frequency"]
    return amy, amyc, norm

def crop(data : list):
    parts = []
    for record in data:
        for i in range(int(record.shape[0] / signal_len) - 1):
            parts.append(np.array(normalize(record[i*signal_len:(i+1) * signal_len])))
    return parts