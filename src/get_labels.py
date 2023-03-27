# %%

import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
import ipywidgets as widgets
from imantics import Polygons, Annotation
from datetime import date
import json

#  functions


def get_scale_and_offset(sh_img, sh_labeler):
    """
    calculate the scale depending on the offset
    """
    sh_labeler = [752.9, 856.7]
    # swap, as img is currently stored as (y, x)
    sh_img = [sh_img[1], sh_img[0]]
    print(sh_img)

    scale_x = sh_labeler[0] / sh_img[0]
    scale_y = sh_labeler[1] / sh_img[1]
    print(scale_x, scale_y)
    print(sh_labeler)

    offset_x = 0
    offset_y = 0
    scale = 1

    if scale_x < scale_y:
        print('scale x')
        scale = scale_x
        offset_x = (sh_labeler[0] - sh_img[0] * scale) / 2
        offset_y = (sh_labeler[1] - sh_img[1] * scale) / 2
        print(offset_x)
        print(offset_y)
    else:
        print('scale y')
        scale = scale_y

        offset_x = (sh_labeler[0] - sh_img[0] * scale) / 2
        offset_y = (sh_labeler[1] - sh_img[1] * scale) / 2

        print(offset_x)
        print(offset_y)

    return scale, offset_x, offset_y


def get_lines(points):
    """
    get all seperate lines from the string with pointsinfo
    """
    lines = []
    line = []
    x_old, y_old = 0, 0

    for point_string in points:
        # check if end of line
        if point_string == ' ':
            lines.append(line.copy())
            line = []
        elif len(point_string) > 2:
            # get point
            x, y = [int(float(point))
                    for point in point_string.split(',')]

            # only add new point
            if (x != x_old) or (y != y_old):
                line.append([x, y])

            # overwrite old point
            x_old, y_old = x, y

    return lines


def get_xy(point, scale, offset_x, offset_y):
    """first remove offset, then scale down"""
    x = int((point[0] - offset_x) / scale)
    y = int((point[1] - offset_y) / scale)
    return x, y


def transform_lines(lines, scale, offset_x, offset_y):
    """transfrom all lines on the image"""
    new_lines = []

    for line in lines:
        if len(line) > 1:
            new_line = []
            x_old, y_old = 0, 0
            # exclude points
            for point in line:
                x, y = get_xy(point, scale, offset_x, offset_y)
                if (x != x_old) or (y != y_old):
                    new_line.append([x, y])

            new_lines.append(new_line)

    return new_lines


def draw_lines(img, lines):
    """draw all lines on the image"""
    # make drawable
    draw = ImageDraw.Draw(img)

    x_old, y_old = 0, 0
    for line in lines:
        # exclude points
        if len(line) > 1:
            for i, point in enumerate(line):
                x, y = point[0], point[1]
                # start from second
                if i > 0:
                    draw.line((x, y, x_old, y_old),width=3, fill=(0,255,0))
                # reset old
                x_old, y_old = x, y

    arr_img = np.array(img)
    return arr_img


def tlbr2bbox(top, left, bottom, right, op=int):
    """
    tlbr = [top, left, bottom, right]
    to ->
    bbox = [x(left), y(top), width, height]
    """
    x = op(left)
    y = op(top)
    width = op(right - left)
    height = op(bottom - top)

    return [x, y, width, height]


def split_labels_by_classes(labels):
    """get the dataset dirstribution from the labels"""
    class_split = {}
    classnames = get_classnames(labels)

    for cur_class in classnames:
        classlabels = []

        for label in labels:
            classname = label.split('.')[0].split('_')[1]
            if classname == cur_class:
                classlabels.append(label)

        class_split[cur_class] = classlabels

    return class_split


def get_classnames(labels):
    """get the name of the classes from the list of labels"""
    return list(set([label.split('.')[0].split('_')[1] for label in labels]))


def get_patnum(label):
    if label.split('.')[0].split('_')[3] == 'patient':
        patnum = int(label.split('.')[0].split('_')[4])
    else:
        patnum = int(label.split('.')[0].split('_')[3])

    return patnum


def dis_labels(labels, split):
    """calculate the dataset dirtribution from the split"""
    # prepare dis
    dis = {}
    for splitname in split.keys():
        dis[splitname] = {}
        dis[splitname]['label'] = []
        dis[splitname]['idx'] = []

    idx_count = 0

    # go trough classes
    classlabels = split_labels_by_classes(labels)
    for classname in classlabels.keys():
        cur_list = classlabels[classname]
        patlist = list(set([get_patnum(label) for label in cur_list]))
        patlen = len(patlist)

        cur_pat = 0
        for i, splitname in enumerate(split.keys()):
            anz_pats = int(patlen * split[splitname])
            final_pat = cur_pat + anz_pats
            if i == len(split.keys()) - 1:
                final_pat = patlen

            for label in cur_list:
                patnum = get_patnum(label)

                if patnum in patlist[cur_pat:final_pat]:
                    dis[splitname]['label'].append(label)
                    dis[splitname]['idx'].append(idx_count)
                    idx_count += 1

            cur_pat = final_pat

    return dis


def make_empty_coco(mode, classnames):
    des = f'{mode}-Radiomics detection in coco-format'
    today = date.today()
    today_str = str(today.year) + str(today.month) + str(today.day)

    cat_dict_list = [{
        "id": i,
        "name": catname,
    } for i, catname in enumerate(classnames)]

    coco = {
        "infos": {
            "description": des,
            "version": "0.01",
            "year": today.year,
            "contributor": "Nikolas Wilhelm",
            "date_created": today_str
        },
        "licences": [
            {
                "id": 1,
                "name": "todo"
            },
        ],
        "categories": cat_dict_list,
        "images": [],
        "annotations": [],
    }
    return coco


def create_cocos(labels, split):
    """
    create datasets in coco format
    """
    dis = dis_labels(labels, split)
    classnames = get_classnames(labels)

    # create a json file for train / val / test
    for dataset in dis.keys():
        coco = make_empty_coco(dataset, classnames)

        for idx, label in zip(dis[dataset]['idx'], dis[dataset]['label']):
            img_dict, ann_dict = perform_file_dict(label, idx, classnames)

            # append the dictionaries to the coco bunch
            coco['images'].append(img_dict)
            coco['annotations'].append(ann_dict)

        # save the coco
        local_path = os.getcwd()
        add = "../" if local_path[-3:] == "src" else ""
        save_file = f'{add}{dataset}.json'

        print(f'Saving to: {save_file}')
        with open(save_file, 'w') as fp:
            json.dump(coco, fp, indent=2)


def perform_file_dict(label, id_label, classnames):
    """get the coco infos for a single file"""
    if label.split('.')[0].split('_')[3] == 'patient':
        _, classname, _, patname, patnum, plane, slicenum = label.split('.')[
            0].split('_')
    else:
        _, classname, patname, patnum, plane, slicenum = label.split('.')[
            0].split('_')

    imgpath = f'{path}/{classname}/{patname}_{patnum}_{plane}_{slicenum}.png'
    img = Image.open(imgpath)
    arr_img = np.array(img)
    sh_img = arr_img.shape[:2]

    # get the polynom
    poly = label_to_poly(label, sh_img)

    # get the bounding box
    bbox, area = poly_2_bbox(poly)

    # get the category id
    cat = classnames.index(classname)

    # build the image dictionary
    img_dict = {
        "id": id_label,
        "file_name": imgpath,
        "height": sh_img[0],
        "width": sh_img[1],
    }

    # build the annotation dictionary
    ann_dict = {
        "id": id_label,
        "image_id": id_label,
        "category_id": cat,
        "iscrowd": 0,
        "area": area,
        "bbox": bbox,
        "segmentation": poly.segmentation,
    }

    return img_dict, ann_dict


def poly_2_bbox(poly):
    """take min/max values and construct a bounding box"""
    # get x and y separated
    x = [x_val for x_val in poly.segmentation[0][::2]]
    y = [y_val for y_val in poly.segmentation[0][1::2]]

    # minmax vals
    left, right = min(x), max(x)
    top, bottom = min(y), max(y)

    # get the bounding box
    bbox = tlbr2bbox(top, left, bottom, right)

    # calculate area
    area = (bottom - top) * (right - left)

    return bbox, area


def label_to_poly(label, sh_img):
    """extract polygon from the label file"""
    filename = f'{labelpath}/{label}'
    with open(filename) as f:
        label_content = f.read()

    # fit images
    sh_labeler = [int(dim)
                  for dim in label_content.split('///')[1].split(',')[:2]]
    scale, offset_x, offset_y = get_scale_and_offset(sh_img, sh_labeler)

    # get coordinates and lines
    points = label_content.split('///')[0].split(';')
    lines = get_lines(points)
    lines = transform_lines(lines, scale, offset_x, offset_y)
    poly = Polygons.create(lines)
    return poly


def get_transformed_lines(arr_img, complete_labelname):
    """
    take image and labelname and calculate the transformed lines
    """
    sh_img = arr_img.shape[:2]

    # read file
    with open(complete_labelname) as f:
        label_content = f.read()

    # fit images
    sh_labeler = [int(dim)
                  for dim in label_content.split('///')[1].split(',')[:2]]
    scale, offset_x, offset_y = get_scale_and_offset(sh_img, sh_labeler)

    # get coordinates and lines
    points = label_content.split('///')[0].split(';')
    lines = get_lines(points)
    lines = transform_lines(lines, scale, offset_x, offset_y)
    return lines


def update(idx=10):
    """Perform label annotation for a single file"""

    # get the content of the file
    labelfiles = os.listdir(labelpath)
    label = labelfiles[idx]
    filename = f'{labelpath}/{label}'

    imgname = filename.split('_')[5].replace('.png.txt', '')
    if len(filename.split('_')) == 7:
        imgname = f'{imgname}_1'

    print(len(filename.split('_')))

    imgpath = f'{path}/{path_img}/{imgname}.png'
    print(imgpath)
    img = Image.open(imgpath)
    arr_img = np.array(img)
    lines = get_transformed_lines(arr_img, filename)

    # draw the lines on the image
    arr_img = draw_lines(img, lines)
    plt.figure(figsize=(16, 16))
    plt.imshow(arr_img)


# %%
if __name__ == '__main__':
    #  select general path
    path = '../data'
    # get the labelname of the folder
    labelname = 'labels_Basti'
    # get the path to the image folder
    path_img = 'osteosarcoma_data_final_unlabelled'

    # define the required datasplit for training
    split = {
        'train': 0.7,
        'valid': 0.2,
        'test': 0.1
    }

    labelpath = f'{path}/{labelname}'
    labelfiles = os.listdir(labelpath)

    # create the coco files depending on the split
    create_cocos(labelfiles, split)
    dis = dis_labels(labelfiles, split)

    # explore dataset
    idx = widgets.IntSlider(min=0, max=len(labelfiles)-1, value=3, step=1)
    widgets.interactive(update, idx=idx)

    # %%
