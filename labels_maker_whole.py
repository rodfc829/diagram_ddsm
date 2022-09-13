"""Algoritmo construído para a criação do arquivo .csv contendo os rótulos das imagens inteiras tamanho 256x256"""


import os
import pandas as pd
import numpy as np
from keras.preprocessing.image import load_img, img_to_array


def scan_folder_whole(parent, names, pathologs, tag='BENIGN'):
    for file_name in os.listdir(parent):
        if file_name.endswith('png'):
            name_image = file_name.replace("-", "_")
            img = load_img("".join((parent, "/", file_name)), color_mode='rgb')
            img_array = img_to_array(img)
            X.append(img_array)
            names.append(name_image)
            pathologs.append(tag)
        else:
            current_path = "".join((parent, "/", file_name))
            if os.path.isdir(current_path):
                scan_folder_whole(current_path, names, pathologs, tag)


def convert_inputs(parent):
    X = []
    list_names = []
    for file_name in os.listdir(parent):
        img = load_img("".join((parent, "/", file_name)), color_mode='rgb')
        img_array = img_to_array(img)
        X.append(img_array)
        list_names.append(file_name)
    return np.asarray(X), list_names


inputdir1 = '/path/to/folder/containing/benign/images'
inputdir2 = '/path/to/folder/containing/malignant/images'
inputdir3 = '/path/to/folder/containing/normal/images'
outdir = '/path/to/exit/directory/ddsm_labels_whole.csv'
X = []
list_names = []
list_pathologies = []

scan_folder_whole(inputdir1, list_names, list_pathologies, 'BENIGN')
scan_folder_whole(inputdir2, list_names, list_pathologies, 'MALIGNANT')
scan_folder_whole(inputdir3, list_names, list_pathologies, 'NORMAL')

df = pd.DataFrame(list(zip(list_names, X, list_pathologies)), columns=['NAME', 'DATA', 'PATHOLOGY'])
df.to_csv(outdir, index=False)
