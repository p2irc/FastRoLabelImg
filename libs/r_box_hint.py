from skimage.util import img_as_float
from skimage import io
from skimage.measure import label, regionprops

import skimage.filters as filt
import skimage.color as color
import skimage.morphology as morph

import os
import math
import numpy as np
import pandas as pd

import torchvision
import torchvision.transforms as T
import torch

from PIL import Image
import mmcv

from Evaluator import Evaluator
from LabelManager import LabelManager
import utils

# import sys
# sys.path.append("/home/badhon/Documents/thesis/AerialDetection")
import DOTA_devkit.polyiou as polyiou
from mmdet.apis import (
    init_detector,
    inference_detector,
    show_result,
    draw_poly_detections,
)

def get_rgb_image(image):
    if len(image.shape) > 2 and image.shape[2] == 4:
        return color.rgba2rgb(image)
    return image

def dist(plot1, plot2):
    return math.sqrt((plot1[0] - plot2[0]) ** 2 + (plot1[1] - plot2[1]) ** 2)

def line(p1, p2):
    A = p1[1] - p2[1]
    B = p2[0] - p1[0]
    C = p1[0] * p2[1] - p2[0] * p1[1]
    return A, B, -C


def intersection(L1, L2):
    D = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        return x, y
    else:
        return False


def get_intersection(a1, a2, b1, b2):
    L1 = line(a1, a2)
    L2 = line(b1, b2)

    R = intersection(L1, L2)
    if R:
        return R
    else:
        print("No single intersection point detected")
        return None


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

# def detect_orientation_image_processing(image, box):
#     img = image[box[1] : box[3], box[0] : box[2]]
#     img_g = color.rgb2gray(img)
#     img_g = filt.gaussian(img_g, 1)
#
#     img_g = img_g > filt.threshold_otsu(img_g)
#     img_g = morph.remove_small_objects(img_g)
#     img_g = morph.remove_small_holes(img_g)
#     img_g = morph.binary_opening(img_g, morph.disk(1))
#
#     label_img = label(img_g)
#
#     regions = regionprops(label_img)
#
#     prop = None
#     mn = 0
#     for props in regions:
#         if props.area > mn:
#             mn = props.area
#             prop = props
#
#     h, w = img_g.shape
#     if mn < 15 or w < 5 or h < 5:
#         print("Too small")
#         return []
#
#     y0, x0 = prop.centroid
#
#     x1 = x0 + math.cos(prop.orientation) * 0.5 * prop.major_axis_length
#     y1 = y0 - math.sin(prop.orientation) * 0.5 * prop.major_axis_length
#     x2 = x0 - math.sin(prop.orientation) * 0.5 * prop.minor_axis_length
#     y2 = y0 - math.cos(prop.orientation) * 0.5 * prop.minor_axis_length
#
#     intersections = []
#     intersections.append(get_intersection([x0, y0], [x1, y1], [0, 0], [w, 0]))
#     intersections.append(get_intersection([x0, y0], [x1, y1], [0, 0], [0, h]))
#     intersections.append(get_intersection([x0, y0], [x1, y1], [w, 0], [w, h]))
#     intersections.append(get_intersection([x0, y0], [x1, y1], [0, h], [w, h]))
#
#     points = []
#     for i in intersections:
#         if i[0] >= 0 and i[0] - w <= 0.0001 and i[1] >= 0 and i[1] - h <= 0.0001:
#             points.append(i)
#
#     if len(points) == 2:
#         b_width = dist(points[0], points[1])
#         b_height = prop.minor_axis_length + 2
#
#         x_center = box[0] + 0.5 * abs(points[0][0] + points[1][0])
#         y_center = box[1] + 0.5 * abs(points[0][1] + points[1][1])
#
#         maj_ang = np.rad2deg(prop.orientation)
#         maj_ang = 180 - maj_ang if maj_ang > 0 else -maj_ang
#
#         return [x_center, y_center, b_width, b_height, maj_ang]
#     else:
#         print("NOT FOUND")
#         return []


def center_region_based_thresh(img):
    h, w = img.shape
    w_l = int(w * 0.45)
    w_r = int(w * 0.55)

    h_t = int(h * 0.45)
    h_b = int(h * 0.55)

    thresh = np.mean(img[h_t : h_b + 1, w_l : w_r + 1])
    thresh_otsu = filt.threshold_otsu(img)

    if thresh > thresh_otsu:
        b = img > thresh_otsu
        # img = img > thresh_otsu
        # b = img < filt.threshold_otsu(img)
    else:
        # img = img < thresh_otsu
        # b = img > filt.threshold_otsu(img)
        b = img < thresh_otsu

    return b


def detect_orientation_image_processing(image, box, center_thresh=True, axis_correction=False, weighted_area=True):
    img = image[box[1] : box[3], box[0] : box[2]]
    img_g = color.rgb2gray(img)
    img_g = filt.gaussian(img_g, 1)

    if center_thresh:
        img_g = center_region_based_thresh(img_g)
    else:
        img_g = img_g > filt.threshold_otsu(img_g)

    img_g = morph.remove_small_objects(img_g)
    img_g = morph.remove_small_holes(img_g)
    img_g = morph.binary_opening(img_g, morph.disk(1))

    label_img = label(img_g)
    regions = regionprops(label_img)

    prop = None
    mn_area = 0
    weighted_mn_area = 0
    h, w = img_g.shape

    for props in regions:
        area = props.area
        weighted = area

        if weighted_area:
            r, c = props.centroid
            d = dist([w / 2, h / 2], [c, r])
            weighted = area / d

        if weighted > weighted_mn_area:
            weighted_mn_area = weighted
            mn_area = area
            prop = props


    if mn_area < 15 or w < 5 or h < 5:
        print("Too small")
        return []

    y0, x0 = h / 2, w / 2

    x1 = x0 + math.cos(prop.orientation) * 0.5 * prop.major_axis_length
    y1 = y0 - math.sin(prop.orientation) * 0.5 * prop.major_axis_length
    x2 = x0 - math.sin(prop.orientation) * 0.5 * prop.minor_axis_length
    y2 = y0 - math.cos(prop.orientation) * 0.5 * prop.minor_axis_length

    ang_min = None
    ang_maj = None

    if axis_correction:
        if w > h:
            ptr1 = get_intersection([x0, y0], [x1, y1], [0, 0], [w, 0])
            ptr2 = get_intersection([x0, y0], [x2, y2], [0, 0], [w, 0])
        else:
            ptr1 = get_intersection([x0, y0], [x1, y1], [0, 0], [0, h])
            ptr2 = get_intersection([x0, y0], [x2, y2], [0, 0], [0, h])

        if ptr1 and ptr2:
            ang_maj = get_angle([x0, y0], ptr1)
            ang_min = get_angle([x0, y0], ptr2)

            ang_maj = 180 + ang_maj if ang_maj < 0 else ang_maj
            ang_min = 180 + ang_min if ang_min < 0 else ang_min

            ang_maj = abs(90 - ang_maj)
            ang_min = abs(90 - ang_min)

            if ang_maj > ang_min:
                x1 = x2
                y1 = y2

    intersections = []
    intersections.append(get_intersection([x0, y0], [x1, y1], [0, 0], [w, 0]))
    intersections.append(get_intersection([x0, y0], [x1, y1], [0, 0], [0, h]))
    intersections.append(get_intersection([x0, y0], [x1, y1], [w, 0], [w, h]))
    intersections.append(get_intersection([x0, y0], [x1, y1], [0, h], [w, h]))

    # TypeError: 'NoneType' object is not subscriptabl
    points = []
    for i in intersections:
        if i is not None and i[0] >= 0 and i[0] - w <= 0.0001 and i[1] >= 0 and i[1] - h <= 0.0001:
            points.append(i)

    if len(points) == 2:
        b_width = dist(points[0], points[1])

        offset = 0
        if ang_maj and ang_min:
            b_height = (
                prop.minor_axis_length + offset
                if ang_maj <= ang_min
                else prop.major_axis_length + offset
            )
        else:
            b_height = prop.minor_axis_length + offset

        x_center = box[0] + 0.5 * abs(points[0][0] + points[1][0])
        y_center = box[1] + 0.5 * abs(points[0][1] + points[1][1])

        maj_ang = np.rad2deg(prop.orientation)
        if ang_maj and ang_min:
            maj_ang = maj_ang if ang_maj <= ang_min else maj_ang + 90
        maj_ang = 180 - maj_ang if maj_ang > 0 else -maj_ang

        return [x_center, y_center, b_width, b_height, maj_ang]
    else:
        print("NOT FOUND")
        return []


def get_outer_rect(points):
    points = np.array(points).astype(int)
    points[points < 0] = 0

    xmin = min(points[:, 0])
    xmax = max(points[:, 0])

    ymin = min(points[:, 1])
    ymax = max(points[:, 1])

    bbox = [xmin, ymin, xmax, ymax]
    return bbox

def generate_bbox_hint(filename, cx, cy, points, method = None):
    print("Before generating hint: ")
    print("Image Path: ", filename)
    print("Center: ", cx, cy)
    print("points: ", points)

    bbox = get_outer_rect(points)

    if method == "mask_rcnn":
        image = Image.open(filename)
        cwha = detect_orientation_mask_rcnn(image, bbox)
    elif method == "image_processing":
        image = get_rgb_image(img_as_float(io.imread(filename)))
        cwha = detect_orientation_image_processing(image, bbox)

    obj = {'status': False}

    if len(cwha) > 0:
        xyp = cwha_to_xyp(*cwha).flatten()
        # xyp[xyp < 0] = 0

        obj = {
            'status': True,
            'cx': cwha[0],
            'cy': cwha[1],
            'direction': math.radians(cwha[-1]),
            'points': list(xyp.reshape((-1, 2))),
            'isRotated': True
        }

    print("After hint generation")
    print(obj)

    return obj




# -------------------------------------------------------------------------------------

mask_rcnn_model = None

def load_model():
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True).cuda()
    model.eval()
    return model

def get_prediction(img, threshold):
    transform = T.Compose([T.ToTensor()])
    img = transform(img).unsqueeze(dim=0).cuda()

    global mask_rcnn_model
    if mask_rcnn_model is None:
        mask_rcnn_model = load_model()

    with torch.no_grad():
        pred = mask_rcnn_model(img)

    pred_score = list(pred[0]["scores"].detach().cpu().numpy())

    pred_t = [pred_score.index(x) for x in pred_score if x > threshold]
    pred_t = pred_t[-1] if len(pred_t) > 0 else 0

    masks = (pred[0]["masks"] > 0.5).squeeze().detach().cpu().numpy()
    masks = masks[: pred_t + 1]
    return masks

def instance_segmentation(image, threshold=0.3):
    masks = get_prediction(image.copy(), threshold)

    mx_area = 0
    mask = None

    for i in range(len(masks)):
        if len(masks[i].shape) < 2:
            continue

        label_img = label(masks[i])
        regions = regionprops(label_img)

        for region in regions:
            if region.area > mx_area:
                mx_area = region.area
                mask = region.filled_image

    return mask

def detect_orientation_mask_rcnn(image, box, axis_correction=False, weighted_area=True):
    img = image.crop((box[0], box[1], box[2], box[3]))
    mask = instance_segmentation(img)

    if mask is None:
        return []

    label_img = label(mask)
    regions = regionprops(label_img)

    prop = None
    mn_area = 0
    weighted_mn_area = 0
    w, h = img.size

    for props in regions:
        area = props.area
        weighted = area

        if weighted_area:
            r, c = props.centroid
            d = dist([w / 2, h / 2], [c, r])
            weighted = area / d

        if weighted > weighted_mn_area:
            weighted_mn_area = weighted
            mn_area = area
            prop = props

    if mn_area < 15 or w < 5 or h < 5:
        print("Too small")
        return []

    y0, x0 = h / 2, w / 2
    x1 = x0 + math.cos(prop.orientation) * 0.5 * prop.major_axis_length
    y1 = y0 - math.sin(prop.orientation) * 0.5 * prop.major_axis_length
    x2 = x0 - math.sin(prop.orientation) * 0.5 * prop.minor_axis_length
    y2 = y0 - math.cos(prop.orientation) * 0.5 * prop.minor_axis_length

    ang_min = None
    ang_maj = None

    if axis_correction:
        if w > h:
            ptr1 = get_intersection([x0, y0], [x1, y1], [0, 0], [w, 0])
            ptr2 = get_intersection([x0, y0], [x2, y2], [0, 0], [w, 0])
        else:
            ptr1 = get_intersection([x0, y0], [x1, y1], [0, 0], [0, h])
            ptr2 = get_intersection([x0, y0], [x2, y2], [0, 0], [0, h])

        if ptr1 and ptr2:
            ang_maj = get_angle([x0, y0], ptr1)
            ang_min = get_angle([x0, y0], ptr2)

            ang_maj = 180 + ang_maj if ang_maj < 0 else ang_maj
            ang_min = 180 + ang_min if ang_min < 0 else ang_min

            ang_maj = abs(90 - ang_maj)
            ang_min = abs(90 - ang_min)

            if ang_maj > ang_min:
                x1 = x2
                y1 = y2

    intersections = []
    intersections.append(get_intersection([x0, y0], [x1, y1], [0, 0], [w, 0]))
    intersections.append(get_intersection([x0, y0], [x1, y1], [0, 0], [0, h]))
    intersections.append(get_intersection([x0, y0], [x1, y1], [w, 0], [w, h]))
    intersections.append(get_intersection([x0, y0], [x1, y1], [0, h], [w, h]))

    points = []
    for i in intersections:
        if i[0] >= 0 and i[0] - w <= 0.0001 and i[1] >= 0 and i[1] - h <= 0.0001:
            points.append(i)

    if len(points) == 2:
        b_width = dist(points[0], points[1])

        offset = 0
        if ang_maj and ang_min:
            b_height = (
                prop.minor_axis_length + offset
                if ang_maj <= ang_min
                else prop.major_axis_length + offset
            )
        else:
            b_height = prop.minor_axis_length + offset

        x_center = box[0] + 0.5 * abs(points[0][0] + points[1][0])
        y_center = box[1] + 0.5 * abs(points[0][1] + points[1][1])

        maj_ang = np.rad2deg(prop.orientation)
        if ang_maj and ang_min:
            maj_ang = maj_ang if ang_maj <= ang_min else maj_ang + 90
        maj_ang = 180 - maj_ang if maj_ang > 0 else -maj_ang

        return [x_center, y_center, b_width, b_height, maj_ang]
    else:
        print("NOT FOUND")
        return []



# ------------------------

def py_cpu_nms_poly_fast_np(dets, thresh = 0.3):
    obbs = dets[:, 0:-1]
    x1 = np.min(obbs[:, 0::2], axis=1)
    y1 = np.min(obbs[:, 1::2], axis=1)
    x2 = np.max(obbs[:, 0::2], axis=1)
    y2 = np.max(obbs[:, 1::2], axis=1)
    scores = dets[:, 8]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    polys = []
    for i in range(len(dets)):
        tm_polygon = polyiou.VectorDouble(
            [
                dets[i][0],
                dets[i][1],
                dets[i][2],
                dets[i][3],
                dets[i][4],
                dets[i][5],
                dets[i][6],
                dets[i][7],
            ]
        )
        polys.append(tm_polygon)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        ovr = []
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        hbb_inter = w * h
        hbb_ovr = hbb_inter / (areas[i] + areas[order[1:]] - hbb_inter)
        h_inds = np.where(hbb_ovr > 0)[0]
        tmp_order = order[h_inds + 1]
        for j in range(tmp_order.size):
            iou = polyiou.iou_poly(polys[i], polys[tmp_order[j]])
            hbb_ovr[h_inds[j]] = iou

        try:
            if math.isnan(ovr[0]):
                pdb.set_trace()
        except:
            pass
        inds = np.where(hbb_ovr <= thresh)[0]
        order = order[inds + 1]
    return keep


def match_detections(gt_boxes, pred_boxes, iou_thresh, show_plot=False, filename=""):
    matched = {}
    not_matched = {}
    true_positive = {}
    used = {}

    for g_idx, gt_box in enumerate(gt_boxes[:]):
        mx_iou = 0
        idx = -1
        for p_idx, pred_box in enumerate(pred_boxes):
            if p_idx in used.keys():
                continue

            iou = utils.get_poly_iou(
                gt_box, pred_box[:-1] if len(pred_box) > 8 else pred_box
            )
            print(iou)

            if iou > iou_thresh:
                if p_idx in true_positive.keys():
                    true_positive[p_idx].append(
                        {"indx": p_idx, "pred_box": pred_box, "gt_box": gt_box}
                    )
                else:
                    true_positive[p_idx] = [
                        {"indx": p_idx, "pred_box": pred_box, "gt_box": gt_box}
                    ]

                if iou > mx_iou:
                    matched[g_idx] = {
                        "indx": p_idx,
                        "pred_box": pred_box,
                        "gt_box": gt_box,
                    }

                    mx_iou = iou
                    idx = p_idx

            used[idx] = True

        if idx == -1:
            pass
#             print("No Match")
        else:
            a = 0

    t_p = len(matched)
    f_p = pred_boxes.shape[0] - len(true_positive.keys())
    f_n = len(not_matched)
    ignored = len(true_positive.keys()) - len(matched)

    if show_plot:

        print("T_P", t_p)
        print("f_p", f_p)
        print("f_n", f_n)
        print("ignored", ignored)

        l1 = []
        l2 = []
        for key in matched.keys():
            l1.append(matched[key]["gt_box"])
            l2.append(matched[key]["pred_box"])

        utils.show_poly_anno(
            filename, [l1, l2], ["Matched Boxes: White: GT, Red: Pred"], False, (10, 10)
        )
        utils.show_poly_anno(
            filename, [gt_boxes, l2], ["GT Boxes", "Matched Pred Boxes"], True, (20, 15)
        )

    return matched


class Dota_Evaluator:
    def __init__(self, path_config, path_work_dir, epoch):
        self._path_config = path_config
        self._path_work_dir = path_work_dir
        self.build_detector_from_epoch(epoch)

    def build_detector_from_epoch(self, epoch):
        self._epoch = epoch
        self._model = init_detector(
            self._path_config,
            os.path.join(self._path_work_dir, "epoch_" + str(epoch) + ".pth"),
            device="cuda:0",
        )

    def predict_single_image(
        self, image, confidence_score, nms_thresh, box=None, enable_nms = True, display_prediction=False, path_image=None
    ):
        dets = inference_detector(self._model, image)

        total_dets = None
        for det in dets:
            total_dets = (
                det if total_dets is None else np.concatenate((total_dets, det))
            )

        if enable_nms:
            keep = py_cpu_nms_poly_fast_np(total_dets, nms_thresh)
            total_dets = total_dets[keep]

        if box is not None:
            for i in range(len(total_dets)):
                total_dets[i] = [
                    box[0] + total_dets[i, 0],
                    box[1] + total_dets[i, 1],
                    box[0] + total_dets[i, 2],
                    box[1] + total_dets[i, 3],
                    box[0] + total_dets[i, 4],
                    box[1] + total_dets[i, 5],
                    box[0] + total_dets[i, 6],
                    box[1] + total_dets[i, 7],
                    total_dets[i, 8],
                ]

        if display_prediction:
            utils.show_poly_anno(
                path_image if path_image else image,
                [np.array([]), total_dets],
                ["Axis Aligned", "Rotated"],
                False,
                (20, 20),
            )

        return total_dets


pretrained_evaluator = None
learning_evaluator = None
learning_thread = None
tuned_nms_thresh = 0.1
tuned_confidence_score = None
tuned_iou_thresh_box_match = 0.25


def set_learning_thread(thread):
    global learning_thread
    learning_thread = thread

def init_evaluator(method):
    if method == "pretrained_dota":
        print("pretrained_dota...................")
        global pretrained_evaluator

        if pretrained_evaluator:
            return pretrained_evaluator
        else:
            path_config = "pretrained_models/faster_rcnn_RoITrans_r50_fpn_1x_dota1_5/faster_rcnn_RoITrans_r50_fpn_1x_dota1_5.py"
            path_work_dir = "pretrained_models/faster_rcnn_RoITrans_r50_fpn_1x_dota1_5"
            epoch = 12

            pretrained_evaluator = Dota_Evaluator(path_config, path_work_dir, epoch)
            return pretrained_evaluator
    else:
        path_config = "pretrained_models/online_learning/config/faster_rcnn_RoITrans_r50_fpn_1x_dota1_5_active_learning.py"
        path_work_dir = "pretrained_models/online_learning/model"
        epoch = 0

        global learning_evaluator, learning_thread

        if learning_thread and learning_thread.is_alive():
            print("Keep using old model-----------------------------------------------")
            return learning_evaluator
        else:
            if os.path.exists(os.path.join(path_work_dir, "latest.pth")):
                files = os.listdir(path_work_dir)
                mx_epoch = 0
                for file in files:
                    if "epoch" in file:
                        mx_epoch = max(mx_epoch, int(file[:-4].split("_")[1]))
                epoch = mx_epoch

            if epoch == 0:
                return None
            else:
                print("Loading New Model--------------------------------------")
                print(epoch)
                learning_evaluator = Dota_Evaluator(path_config, path_work_dir, epoch)
                return learning_evaluator


def detect_orientation_based_on_detections(filename, mode="global", method="online_learning", bboxes = np.array([])):
    image = mmcv.imread(filename)
    img_h, img_w, _ = image.shape

    evaluator = init_evaluator(method)

    if evaluator is None:
        return [{'status': False}]

    global tuned_confidence_score, tuned_nms_thresh

    total_dets = evaluator.predict_single_image(
        np.array(image), tuned_confidence_score, tuned_nms_thresh
    )

    proposals = []

    if mode == "global":
        for sug_box in total_dets:
            sug_box = np.array(sug_box)[:-1]
            sug_box = utils.correct_dimension(sug_box)
            cx, cy = utils.calculate_center(sug_box)
            proposals.append({
                'status': True,
                'cx': cx,
                'cy': cy,
                'direction': math.radians(utils.calculate_angle(sug_box)),
                'points': list(sug_box.reshape((-1, 2))),
                'isRotated': True,
                'label': "wheat_head"
            })
    else:
        xyp_bboxes = utils.xyxy_to_poly(bboxes, r_numpy=True)

        global tuned_iou_thresh_box_match

        for box, xyp_box in zip(bboxes, xyp_bboxes):
            res = match_detections(np.array([xyp_box]), total_dets, tuned_iou_thresh_box_match)
            sug_box = []

            if not bool(res):
                print("Detecting with image processing")
                res = detect_orientation_image_processing(image, box)
                if res:
                    sug_box = cwha_to_xyp(*res).flatten()
            else:
                sug_box = np.array(res[0]['pred_box'])[:-1]

            if len(sug_box) > 0:
                sug_box = utils.correct_dimension(sug_box)
                cx, cy = utils.calculate_center(sug_box)
                angleDeg = utils.calculate_angle(sug_box)

                print(angleDeg)
                print(math.radians(angleDeg))

                proposals.append({
                    'status': True,
                    'cx': cx,
                    'cy': cy,
                    'direction': math.radians(angleDeg),
                    'points': list(sug_box.reshape((-1, 2))),
                    'isRotated': True
                })
            else:
                proposals.append({'status': False})

    return proposals
