"""Algoritmo construído para a criação do arquivo .csv contendo os rótulos dos recortes de tamanho 32x32 da imagens inteiras"""

import os
import pandas as pd


def scan_folder_ROIs(parent, names, pathologs, l_types):
    for file_name in os.listdir(parent):
        if file_name.endswith('ics'):
            calendar = open("".join((parent, "/", file_name)))
            calendar_lines = calendar.readlines()
            for line in calendar_lines[-4:]:
                print(line[-12:-1], line[-12:-1] == 'NON_OVERLAY')
                if line[-12:-1] == 'NON_OVERLAY':  # caso seja uma imagem sem nenhuma anomalia
                    print("".join((parent, "/", file_name[:-3].replace("-", "_"), line.split()[0], '.png')))
                    name_image = "".join((file_name[:-3].replace("-", "_"), line.split()[0], '.png'))
                    names.append(name_image)
                    pathologs.append('NORMAL')
                    l_types.append('NONE')
                else:
                    print("".join((parent, "/", file_name[:-3].replace("-", "_"), line.split()[0], '.overlay')))
                    overlay = open("".join((parent, "/", file_name[:-3].replace("-", "_"), line.split()[0], '.overlay')))
                    overlay_lines = overlay.readlines()
                    quant_abnorms = int(overlay_lines[0].split()[1])
                    quant_outlines = []
                    lesions = []
                    im_pathol = []
                    for i in overlay_lines:
                        if i.split() and len(lesions) <= quant_abnorms and i.split()[0] == 'LESION_TYPE':
                            lesions.append(i.split()[1])
                        elif i.split() and i.split()[0] == 'PATHOLOGY':
                            im_pathol.append(i.split()[1])
                        elif i.split() and i.split()[0] == 'TOTAL_OUTLINES':
                            quant_outlines.append(int(i.split()[1]))
                    for i in range(quant_abnorms):
                        for j in range(quant_outlines[i]):
                            name_image = "".join((file_name[:-3].replace("-", "_"), line.split()[0], '-', calendar_lines[3].split()[1], '-', calendar_lines[3].split()[2], '-', calendar_lines[3].split()[3], '-', str(i+1), '-', str(j+1), '.png'))
                            names.append(name_image)
                            pathologs.append(im_pathol[i])
                            l_types.append(lesions[i])

        else:
            current_path = "".join((parent, "/", file_name))
            if os.path.isdir(current_path):
                scan_folder_ROIs(current_path, names, pathologs, l_types)


inputdir1 = '/path/to/folder/containing/benign/images'
inputdir2 = '/path/to/folder/containing/malignant/images'
inputdir3 = '/path/to/folder/containing/normal/images'
outdir = '/path/to/exit/directory/ddsm_labels_patches.csv'
list_names = []
list_pathologies = []
list_lesions = []

scan_folder_ROIs(inputdir1, list_names, list_pathologies, list_lesions)
scan_folder_ROIs(inputdir2, list_names, list_pathologies, list_lesions)
scan_folder_ROIs(inputdir3, list_names, list_pathologies, list_lesions)

df = pd.DataFrame(list(zip(list_names, list_pathologies, list_lesions)), columns=['NAME', 'PATHOLOGY', 'LESION'])
df.to_csv(outdir, index=False)
