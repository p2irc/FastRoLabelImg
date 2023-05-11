import os
import pandas as pd
import numpy as np
import json
import mmcv
import itertools
import shutil
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import xml.etree.ElementTree as ET

from collections import defaultdict
from PIL import Image

import DOTA_devkit.polyiou as polyiou


def TuplePoly2Poly(poly):
    outpoly = [
        poly[0][0],
        poly[0][1],
        poly[1][0],
        poly[1][1],
        poly[2][0],
        poly[2][1],
        poly[3][0],
        poly[3][1],
    ]
    return outpoly

def dist(a, b):
    return np.linalg.norm(a - b)

def show_four_corners(image, corners, size=(5, 5)):
    fig, ax = plt.subplots(1, 1, figsize=size)
    ax.imshow(image)
    colors = ["yellow", "red", "green", "blue"]

    for i, c in enumerate(corners):
        circ = patches.Circle((c[0], c[1]), 5, color=colors[i])
        ax.add_patch(circ)

    plt.tight_layout()
    plt.show()

def angle_with_x_axis(a, b):
    angle1 = np.rad2deg(np.arctan2(a[1] - b[1], a[0] - b[0]))
    angle2 = np.rad2deg(np.arctan2(b[1] - a[1], b[0] - a[0]))
    angle = max(angle1, angle2)
    return angle

def calculate_angle(b):
    b = b.reshape((-1, 2))
    d1 = np.linalg.norm(b[0] - b[1])
    d2 = np.linalg.norm(b[1] - b[2])
    if d1 > d2:
        return angle_with_x_axis(b[0], b[1])
    return angle_with_x_axis(b[1], b[2])

def calculate_center(b):
    b = b.reshape((-1, 2))
    cx = (b[0][0] + b [2][0]) * 0.5
    cy = (b[0][1] + b [2][1]) * 0.5

    return cx, cy

def correct_dimension(b):
    b = b.reshape((-1, 2))
    d1 = np.linalg.norm(b[0] - b[1])
    d2 = np.linalg.norm(b[1] - b[2])
    if d1 < d2:
        b = np.roll(b, -1, axis=0)

    return b.flatten()

def xywh_to_xyxy(boxes, r_numpy=False):
    boxes = boxes if isinstance(boxes, np.ndarray) else np.array(boxes)
    boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
    boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

    return boxes if r_numpy else boxes.tolist()


def xyxy_to_poly(boxes, r_numpy=False):
    boxes = boxes if isinstance(boxes, np.ndarray) else np.array(boxes)
    poly_boxes = [[b[0], b[1], b[2], b[1], b[2], b[3], b[0], b[3]] for b in boxes]

    return np.array(poly_boxes) if r_numpy else poly_boxes


def cwha_to_xyp(x_c, y_c, w, h, a):
    """Returns four corners of the box
       First determine four axis aligned corner points
       (-w/2, -h/2)-----------(w/2, -h/2)
                   |         |
                   |    c    |
                   |         |
       (-w/2, h/2) -----------(w/2, h/2)

       Then, apply angle on it to get final corner points

       src * H = dst
    ...

    Parameters
    ----------
    x_c : number
          x coordinate of center
    y_c : number
          y coordinate of center
    w   : number
          width of the box
    h   : number
          height of the box
    a   : number
          angle of rotation in degrees

    Returns
    -------
    numpy array
        four corners of the box
    """
    w = w / 2
    h = h / 2
    c = np.array([[x_c, y_c]])
    a = math.radians(a)

    rotation = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)],])

    axis_aligned_box = np.array(
        [
            [-w, -h],  # bottom-left
            [w, -h],  # bottom-right
            [w, h],  # top-right
            [-w, h],  # top-left
        ]
    )

    return c + axis_aligned_box.dot(rotation.T)


def cwha_to_xyxy(x_c, y_c, w, h, a):
    poly = cwha_to_xyp(x_c, y_c, w, h, a)

    xmin = max(int(poly[:, 0].min()), 0)
    xmax = int(poly[:, 0].max())

    ymin = max(int(poly[:, 1].min()), 0)
    ymax = int(poly[:, 1].max())

    return [xmin, ymin, xmax, ymax]


def get_poly_iou(box1, box2):
    box1 = np.array(box1).astype(float).tolist()
    box2 = np.array(box2).astype(float).tolist()

    return polyiou.iou_poly(polyiou.VectorDouble(box1), polyiou.VectorDouble(box2))


def show_poly_anno(filename, b_boxes=[], title=None, sep=False, f_size=(10, 10), isSave=False, path=None):
    im = np.array(Image.open(filename), dtype=np.uint8)
    ln = len(b_boxes)
    fig, ax = plt.subplots(1, ln if sep else 1, figsize=f_size)

    if sep:
        for i in range(ln):
            ax[i].imshow(im)
            ax[i].set_title(title[i] if title else "")
    else:
        ax.imshow(im)
        ax.set_title(title[0] if title else "")

    color = ["w", "r"]

    for b_no, boxes in enumerate(b_boxes):
        boxes = boxes if isinstance(boxes, np.ndarray) else np.array(boxes)
        for box in boxes:
            box_reshaped = box[:8].reshape((-1, 2))
            poly = patches.Polygon(
                box_reshaped, linewidth=1, edgecolor=color[b_no], facecolor="none",
            )

            if sep:
                ax[b_no].add_patch(poly)
            else:
                ax.add_patch(poly)

    if isSave:
        create_folder(path)
        plt.gca().set_axis_off()
        plt.savefig(os.path.join(path, filename.split('/')[-1]), format="png", transparent = True, bbox_inches = 'tight', pad_inches = 0)
    else:
        plt.show()

    plt.close()

def create_folder(path):
    if not os.path.exists(path):
        os.mkdir(path)

def remove_folder(path):
    if os.path.exists(path):
        shutil.rmtree(path)

def draw_plot_xy(x, y, title):
    plt.figure()
    plt.plot(x, y)
    plt.title(title)
    plt.show()


# gt_boxes numpy, pred_boxes numpy
def match_detections(gt_boxes, pred_boxes, iou_thresh, show_plot = False, filename=""):
    matched = {}
    not_matched = {}
    true_positive = {}
    used = {}

    for g_idx, gt_box in enumerate(gt_boxes):
#         print("Trying to match: ")
#         print(g_idx, gt_box)
        mx_iou = 0
        idx = -1
        for p_idx, pred_box in enumerate(pred_boxes):
            if p_idx in used.keys():
                continue

            iou = get_poly_iou(gt_box, pred_box[:-1])
            if iou > iou_thresh:

                if p_idx in true_positive.keys():
                    true_positive[p_idx].append({'indx': p_idx, 'pred_box': pred_box, 'gt_box': gt_box})
                else:
                    true_positive[p_idx] = [{'indx': p_idx, 'pred_box': pred_box, 'gt_box': gt_box}]

                if iou > mx_iou:
                    matched[g_idx] = {'indx': p_idx, 'pred_box': pred_box, 'gt_box': gt_box}

                    mx_iou = iou
                    idx = p_idx

            used[idx] = True

        if idx == -1:
#             print("No Match")
            not_matched[g_idx] = {'indx': g_idx, 'gt_box': gt_box}
        else:
#             print("Found Match")
#             print(matched[g_idx])
            a = 0

    t_p = len(matched)
    f_p = pred_boxes.shape[0] - len(true_positive.keys())
    f_n = len(not_matched)
    ignored = len(true_positive.keys()) - len(matched)

    print("-------------------------------")
    print("-------------------------------")
    print("Total Ground Truth Boxes: ", gt_boxes.shape[0])
    print("Total Prediction Boxes: ", pred_boxes.shape[0])
    print("")

    print("Detections - True Positive: ", t_p)
    print("Detections - False Positive: ", f_p)
    print("Detections - False Negative ", f_n)
    print("Extra True Positive (Ignored as not max iou): ", ignored)


#     for key in true_positive.keys():
#         print("Found ", len(true_positive[key])," for ", key)
#         print(true_positive[key])
#         print("")

    if show_plot:
        l1 = []
        l2 = []
        for key in matched.keys():
            l1.append(matched[key]['gt_box'])
            l2.append(matched[key]['pred_box'])

        show_poly_anno(filename, [l1, l2], ["Matched Boxes: White: GT, Red: Pred"], False, (10, 10))
        show_poly_anno(filename, [gt_boxes, l2], ["GT Boxes", "Matched Pred Boxes"], True, (20, 15))

        l1 = []
        for key in not_matched.keys():
            l1.append(not_matched[key]['gt_box'])

        l2 = []
        for p_idx, box in enumerate(pred_boxes):
            if p_idx not in true_positive:
                l2.append(box)
                for g_idx, gt_box in enumerate(gt_boxes):
                    iou = get_poly_iou(gt_box, box[:-1])
                    if iou > iou_thresh:
                        print([*used])
                        print(p_idx, g_idx)
                        show_poly_anno(filename, [[gt_box], [box]], ["GT", "PRED"], True, (20, 15))
                        print("Something is wrong")

        show_poly_anno(filename, [l1, l2], ["False Negatives", "False Positives"], True, (20, 15))

        if len(true_positive.keys()) - len(matched) > 0:
            l1 = []
            l2 = []

            l3 = []
            for key in matched.keys():
                l3.append(matched[key]['indx'])

            for key in true_positive.keys():
                if key in l3:
                    l1.append(true_positive[key][0]['pred_box'])
                else:
                    print(true_positive[key])
                    l2.append(true_positive[key][0]['pred_box'])

            show_poly_anno(filename, [l1, l2], ["Ignored Boxes: White: GT, Red: Ignored"], False, (10, 10))


    return t_p, f_p, f_n, ignored


def get_rgb_image(image):
    if len(image.shape) > 2 and image.shape[2] == 4:
        return color.rgba2rgb(image)
    return image


def display_image(image, isGray=False, isLarge=True, size=20):
    if isLarge:
        plt.figure(figsize=(size, size))
    else:
        plt.figure()

    if isGray:
        plt.imshow(image, cmap="gray")
    else:
        plt.imshow(image)

    plt.show()



def read_xml(xml_file: str, isSave=False, path=None, size=(15, 15)):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    list_with_all_boxes = []

    filename = root.find("filename").text

    if ".png" not in filename:
        filename = filename + ".png"

    for boxes in root.iter("object"):

        xmin, ymin, xmax, ymax = None, None, None, None

        for box in boxes.findall("bndbox"):
            xmin = int(box.find("xmin").text)
            ymin = int(box.find("ymin").text)
            xmax = int(box.find("xmax").text)
            ymax = int(box.find("ymax").text)

            list_with_all_boxes.append([xmin, ymin, xmax, ymax])

    xyps = xyxy_to_poly(list_with_all_boxes, r_numpy=True)

    if isSave:
        image = get_rgb_image(
            img_as_float(io.imread(os.path.join(path_images, filename)))
        )

        show_poly_anno(
            os.path.join(path_images, filename),
            [np.array(xyps)],
            ["Rotated"],
            False,
            size,
            True,
            path,
        )

    return filename, np.array(list_with_all_boxes), xyps


def read_xml_rotated(xml_file: str, isSave=False, path=None, size=(15, 15)):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    list_with_all_boxes = []
    labels = []

    filename = root.find("filename").text

    if ".png" not in filename:
        filename = filename + ".png"

    for boxes in root.iter("object"):

        cx, cy, w, h, a = None, None, None, None, None
        label = boxes.find('name').text

        for box in boxes.findall("robndbox"):
            labels.append(label)

            cx = float(box.find("cx").text)
            cy = float(box.find("cy").text)
            w = float(box.find("w").text)
            h = float(box.find("h").text)
            a = float(box.find("angle").text)

            angle_deg = math.degrees(a)

            if w < h:
                temp = w
                w = h
                h = temp
                angle_deg = (angle_deg + 90) % 180

            list_with_single_boxes = [cx, cy, w, h, angle_deg]
            list_with_all_boxes.append(list_with_single_boxes)

    xyps = []
    for cwha in list_with_all_boxes:
        xyp = cwha_to_xyp(*cwha).flatten()
        xyps.append(xyp)

    xyps = np.array(xyps)

    if isSave:
        image = get_rgb_image(
            img_as_float(io.imread(os.path.join(path_images, filename)))
        )

        show_poly_anno(
            os.path.join(path_images, filename),
            [np.array(xyps)],
            ["Rotated"],
            False,
            size,
            True,
            path,
        )

    return filename, np.array(list_with_all_boxes), xyps, labels
