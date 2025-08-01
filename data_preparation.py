import os
import numpy as np
from os import listdir
from os.path import join

def full_fname2_str(data_dir, fname, sep_char):
    fnametostr = ''.join(fname).replace(data_dir, '')
    ind = int(fnametostr.index(sep_char))
    label = int(fnametostr[ind + 1:ind + 3])
    return label

def read_and_save_data():
    data_dir = 'MSRAction3DSkeleton(20joints)'
    print(f"Reading data from {data_dir}")
    data, labels = [], []

    files = [join(data_dir, f) for f in sorted(listdir(data_dir)) if f.endswith('.txt')]
    for file in files:
        action = np.loadtxt(file)[:, :3].flatten()
        label = full_fname2_str(data_dir, file, 'a')
        frame_size = len(action) // 60
        action = action.reshape(frame_size, 60)
        data.append(action)
        labels.append(label - 1)

    data = np.array(data, dtype=object)
    labels = np.array(labels)

    os.makedirs('model', exist_ok=True)
    np.save("model/data.txt", data)
    np.save("model/labels.txt", labels)
    print("✅ Data saved successfully.")

if __name__ == "__main__":
    read_and_save_data()