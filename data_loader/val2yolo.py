import os
import cv2
import numpy as np
import shutil
from tqdm import tqdm
import yaml


def xywh2xxyy(box):
    x1 = box[0]
    y1 = box[1]
    x2 = box[0] + box[2]
    y2 = box[1] + box[3]
    return (x1, x2, y1, y2)


def convert(size, box):
    dw = 1.0 / (size[0])
    dh = 1.0 / (size[1])
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


def wider2face(path, ignore_small=0):
    data = {}
    with open("{}/label.txt".format(path), "r") as f:
        lines = f.readlines()
        for line in tqdm(lines):
            line = line.strip()
            if "#" in line:
                path_img = "{}/images/{}".format(path, line[2:])
                img = cv2.imread(path_img)
                height, width, _ = img.shape
                data[path_img] = list()
            else:
                box = np.array(line.split()[0:4], dtype=np.float32)  # (x1,y1,w,h)
                if box[2] < ignore_small or box[3] < ignore_small:
                    continue
                box = convert((width, height), xywh2xxyy(box))
                label = "0 {} {} {} {} -1 -1 -1 -1 -1 -1 -1 -1 -1 -1".format(
                    round(box[0], 4),
                    round(box[1], 4),
                    round(box[2], 4),
                    round(box[3], 4),
                )
                data[path_img].append(label)
    return data


if __name__ == "__main__":
    path2val = yaml.load(
        open("../config/data.widerface.yaml"), Loader=yaml.FullLoader
    ).get("ori_val")
    datas = wider2face("../" + path2val)
    save_path_img = "../data/widerface/val/images"
    save_path_txt = "../data/widerface/val/labels"
    if not os.path.exists(save_path_img):
        os.makedirs(save_path_img)
    if not os.path.exists(save_path_txt):
        os.makedirs(save_path_txt)
    for idx, data in enumerate(datas.keys()):
        pict_name = os.path.basename(data)
        out_img = "{}/{}".format(save_path_img, pict_name)
        out_txt = "{}/{}.txt".format(save_path_txt, os.path.splitext(pict_name)[0])
        shutil.copyfile(data, out_img)
        labels = datas[data]
        f = open(out_txt, "w")
        for label in labels:
            f.write(label + "\n")
        f.close()
