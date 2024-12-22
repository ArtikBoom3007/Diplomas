from pyedflib import highlevel
from sys import argv
from pathlib import Path
import json
import matplotlib.pyplot as plt


def make_vecg(signal):
    """
    Преобразует сигналы ЭКГ в ВЭКГ и добавляет координаты x, y, z.
    """
    DI = signal[0]
    DII = signal[1]
    V1 = signal[2]
    V2 = signal[3]
    V3 = signal[4]
    V4 = signal[5]
    V5 = signal[6]
    V6 = signal[7]

#координаты x, y, z для ВЭКГ
    x = -(-0.172*V1 - 0.074*V2 + 0.122*V3 + 0.231*V4 + 0.239*V5 + 0.194*V6 + 0.156*DI - 0.01*DII)
    y = (0.057*V1 - 0.019*V2 - 0.106*V3 - 0.022*V4 + 0.041*V5 + 0.048*V6 - 0.227*DI + 0.887*DII)
    z = -(-0.229*V1 - 0.31*V2 - 0.246*V3 - 0.063*V4 + 0.055*V5 + 0.108*V6 + 0.022*DI + 0.102*DII)
    
    return x, y, z

if __name__ == "__main__":
    if len(argv) < 2:
        raise ValueError("Path to file does not specified!")
    filepath = argv[1]
    signals, signal_headers, _ = highlevel.read_edf(filepath)
    x, y, z = make_vecg(signals)


    json_obj = {"x": list(x), "y": list(y), "z": list(z)}
    file = Path("./output_json.json")
    with file.open("w", encoding ="utf-8") as f:
        json.dump(json_obj, f)

    # fig = plt.figure()
    # ax = plt.axes(projection='3d')
    # ax.plot3D(x, y, z, 'green')
    # plt.show()