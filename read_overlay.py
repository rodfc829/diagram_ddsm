"""Algoritmo construído para efetivar a criação dos recortes 32x32 (regiões de interesse) das imagens inteiras"""

import os
from PIL import Image


def find_square(xmin, xmax, ymin, ymax, im):
    width = xmax - xmin
    height = ymax - ymin
    if width > height:
        new_ymin = ymin - (width - height)//2 + 1
        new_ymax = ymax + (width - height)//2 + 1
        if new_ymin < 0:
            dist = 0 - new_ymin
            new_ymin = 0
            new_ymax += dist
        if new_ymax > im.size[1]:
            dist = new_ymax - im.size[1]
            new_ymax = im.size[1]
            new_ymin -= dist
        return xmin, min(xmax, im.size[0]), new_ymin, new_ymax
    elif width < height:
        new_xmin = xmin - (height - width)//2 + 1
        new_xmax = xmax + (height - width)//2 + 1
        flag = False
        if new_xmin < 0:
            dist = 0 - new_xmin
            new_xmin = 0
            new_xmax += dist
            flag = True
        if new_xmax > im.size[0]:
            if flag:
                new_xmax = im.size[0]
            else:
                dist = new_xmax - im.size[0]
                new_xmax = im.size[0]
                new_xmin = max(new_xmin - dist, 0)
        return new_xmin, new_xmax, ymin, ymax
    else:
        return xmin, xmax, ymin, ymax


def image_travel(commands, im):
    xatual = int(commands[0])
    yatual = int(commands[1])
    xpos = [xatual]
    ypos = [yatual]

    for i in commands[2:]:
        if i == '#':
            break
        if int(i) == 0:
            yatual -= 1
        elif int(i) == 1:
            xatual += 1
            yatual -= 1
        elif int(i) == 2:
            xatual += 1
        elif int(i) == 3:
            xatual += 1
            yatual += 1
        elif int(i) == 4:
            yatual += 1
        elif int(i) == 5:
            xatual -= 1
            yatual += 1
        elif int(i) == 6:
            xatual -= 1
        elif int(i) == 7:
            xatual -= 1
            yatual -= 1
        else:
            raise Exception("Slice feito de maneira inadequada. Verificar")

        if xatual > im.size[0]:
            xatual = im.size[0]
        if yatual > im.size[1]:
            yatual = im.size[1]
        if xatual < 0:
            xatual = 0
        if yatual < 0:
            yatual = 0
        xpos.append(xatual)
        ypos.append(yatual)

    xmin, ymin = min(xpos), min(ypos)
    xmax, ymax = max(xpos), max(ypos)

    return find_square(xmin, xmax, ymin, ymax, im)


def scan_folder(parent):
    for file_name in os.listdir(parent):
        # FILE NAME É SÓ O NOME DO ARQUIVO SEM O PATH INTEIRO
        if file_name.endswith('ics'):  # encontra o calendar
            calendar = open("".join((parent, "/", file_name)))  # abre o calendar
            calendar_lines = calendar.readlines()  # pega as linhas do calendar
            for line in calendar_lines[-4:]:  # acessa as últimas quatro linhas do calendário
                if line[-12:-1] != 'NON_OVERLAY':  # acessa aquelas que tem overlay
                    image = Image.open("".join((parent, "/", file_name[:-3].replace("-", "_"), line.split()[0], '.png')))  # abre a imagem
                    overlay = open("".join((parent, "/", file_name[:-3].replace("-", "_"), line.split()[0], '.overlay')))  # abre o arquivo overlay
                    overlay_lines = overlay.readlines()  # separa o overlay em linhas
                    quant_abnorms = overlay_lines[0].split()[1]  # quantidade de anormalidades
                    quant_outlines = []  # quantidade de outlines para cada anormalidade
                    for i in overlay_lines:
                        if i.split() and i.split()[0] == 'TOTAL_OUTLINES':
                            quant_outlines.append(int(i.split()[1]))  # preenche a lista dos outlines
                    idx = 1
                    for ab in range(int(quant_abnorms)):
                        current_interval = overlay_lines[idx:idx+6+2*quant_outlines[ab]]
                        flag = False
                        if current_interval[-1] == 'BOUNDARY\n' or current_interval[-1] == '\tCORE\n':
                            current_interval = overlay_lines[idx:idx+7+2*quant_outlines[ab]]
                            flag = True
                        for i in range(int(quant_outlines[ab])):  # acessa os overlays por quantidade por imagens
                            xmin, xmax, ymin, ymax = image_travel(current_interval[-1-2*i].split(), image)  # encontra o quadrado
                            if xmin < 0 or xmax < 0 or ymin < 0 or ymax < 0:
                                raise Exception("Valor negativo encontrado no crop. Verificar")
                            if xmax > image.size[0] or ymax > image.size[1]:
                                raise Exception("Valor maior que as dimensões da imagem encontrado no crop. Verificar")
                            im_crop = image.crop((xmin, ymin, xmax, ymax))  # cropa a ROI
                            im_crop.save("".join((outdir, "/", file_name[:-3].replace("-", "_"), line.split()[0], '-', calendar_lines[3].split()[1], '-', calendar_lines[3].split()[2], '-', calendar_lines[3].split()[3], '-', str(ab+1), '-', str(i+1), '.png')))  # salva a ROI
                        if flag:
                            idx += 7 + 2 * quant_outlines[ab]
                        else:
                            idx += 6 + 2*quant_outlines[ab]
        else:
            current_path = "".join((parent, "/", file_name))
            if os.path.isdir(current_path):
                # if we're checking a sub-directory, recursively call this method
                scan_folder(current_path)


inputdir = '/path/to/folder/with/images/containing/ROIs'
outdir = '/path/to/exit/directory'
scan_folder(inputdir)
