# Evaluator combines the ability to both predict and calculate tp, fp, precision, recall and ap.
# If det boxes are already calculated we can use methods from utils.evaluation and pass gt boxes and det boxes
import mmcv
import os
import cv2
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import DOTA_devkit.polyiou as polyiou
from IPython.display import display, HTML

import utils
import evaluation

# import sys
# sys.path.append("/home/badhon/Documents/thesis/AerialDetection")
import DOTA_devkit.polyiou as polyiou
from mmdet.apis import init_detector, inference_detector, show_result, draw_poly_detections



class Evaluator:
    def __init__(self, path_config, path_work_dir, epoch):
        self._path_config = path_config
        self._path_work_dir = path_work_dir
        self.build_detector_from_epoch(epoch)

    def build_detector_from_epoch(self, epoch):
        self._epoch = epoch
        self._model =  init_detector(self._path_config, os.path.join(self._path_work_dir, 'epoch_' + str(epoch) + '.pth'), device='cuda:0')

    def get_model(self):
        return self._model

    def predict_single_image(self, path_image, confidence_score=None, display_prediction=False, r_dict=False, save_prediction=False, save_path=None):
        dets = np.array(inference_detector(self._model, path_image))

        detections = None
        for det in dets:
            detections = (
                det if detections is None else np.concatenate((detections, det))
            )
        # keep = self.py_cpu_nms_poly_fast_np(detections, 0.1)
        # detections = detections[keep]

        isDet = detections.shape[0] > 0

        if isDet and detections.shape[1] <= 5:
            confidence = np.array([detections[:,4]])
            bboxes = utils.xyxy_to_poly(detections[:,:4], True)
            detections = np.concatenate((bboxes, confidence.T), axis=1)

        if isDet and confidence_score is not None:
            detections = detections[detections[:,-1] > confidence_score]

        if display_prediction:
            utils.show_poly_anno(
                path_image, [detections], ["Detections: " + path_image.split('/')[-1]]
            )

        if save_prediction:
            if not save_path:
                print("Save path is not defined.")
            else:
                utils.show_poly_anno(
                    path_image, [detections], ["Detections: " + path_image.split('/')[-1]], isSave=True, path=save_path, f_size=(20, 20)
                )

        if r_dict:
            detections = {
                path_image: detections
            }

        return detections

    def file_range(self, path_folder, limit):
        filenames = os.listdir(path_folder)
        start = limit[0] if limit else 0
        end = limit[1] if limit else len(filenames)
        return filenames[start:end]

    def predict_image_folder(self, path_folder, limit=None, confidence_score=None, display_prediction=False, save_prediction=False, save_path=None):
        filenames = self.file_range(path_folder, limit)

        all_dets = {}
        for path_image in filenames:
            all_dets[path_image] = self.predict_single_image(os.path.join(path_folder, path_image), confidence_score, display_prediction, save_prediction=save_prediction, save_path=save_path)
        return all_dets

    def evaluate_single_image(self, path_image, gt_boxes, thresh=0.5, use_07_metric=False, r_metrics = None):
        all_gts = {
            path_image: gt_boxes
        }
        all_dets = self.predict_single_image(path_image, r_dict=True)
        res = evaluation.voc_eval(all_gts, all_dets, thresh, use_07_metric)

        if r_metrics:
            res = [res['metric'] for metric in r_metrics]
        return res

    def evaluate_image_folder(self, path_folder, limit=None, all_gts={}, thresh=0.5, confidence_score=None, use_07_metric=False, r_metrics = None):
        filenames = self.file_range(path_folder, limit)

        for key in list(all_gts.keys()):
            if key not in filenames:
                all_gts.pop(key, None)

        all_dets = self.predict_image_folder(path_folder, limit, confidence_score)

        res = evaluation.voc_eval(all_gts, all_dets, thresh, use_07_metric)

        if r_metrics:
            res = [res['metric'] for metric in r_metrics]
        return res

    def print_summary(self, res={}, exclude=['rec', 'prec']):
        for key in res.keys():
            if key in exclude:
                continue
            print(key + ": ", res[key])
