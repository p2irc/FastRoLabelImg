import numpy as np
import DOTA_devkit.polyiou as polyiou
import utils

# i want to give all gt boxes and detections (det box + confidence) and threshold to get the ap value
# gt boxes could be mapped to image filename or image ids
# {
#     file_name: [[x1, y1, x2, y2, x3, y3, x4, y4]]
#     file_name: [[x1, y1, x2, y2, x3, y3, x4, y4]]
# }
# detections could be mapped to image filename or image ids
# {
#     file_name: [[x1, y1, x2, y2, x3, y3, x4, y4, confidence]]
#     file_name: [[x1, y1, x2, y2, x3, y3, x4, y4, confidence]]
# }


def calcoverlaps(BBGT_keep, bb):
    overlaps = []
    for index, GT in enumerate(BBGT_keep):
        overlap = polyiou.iou_poly(polyiou.VectorDouble(BBGT_keep[index]), polyiou.VectorDouble(bb))
        overlaps.append(overlap)
    return overlaps


def voc_eval(gts, detections, ovthresh=0.5, use_07_metric=False):
    class_recs = {}

    n_gts = 0
    for key in gts.keys():
        n_gts += len(gts[key])
        class_recs[key] = {'bbox': gts[key], 'det': [False] * len(gts[key])}


    image_ids = []
    confidence = []
    BB = []

    for key in detections.keys():
        for det in detections[key]:
            det = np.array(det)
            confidence.append(float(det[-1]))
            BB.append([float(z) for z in det[:-1]])
            image_ids.append(key)

    BB = np.array(BB)
    confidence = np.array(confidence)

#     class_recs = {
#         image: {
#             'bbox': [[x1, y1, x2, y2, x3, y3, x4, y4]], n X 8
#             'det': [False] n
#         }
#     }

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)

    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]

    # Check each detection and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)

    pred_rot = []
    gt_rot = []
    bbs = []
    gtbbs = []

    for d in range(nd):
        # R is of one particular image
        R = class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = np.array(R['bbox']).astype(float) # will make it a numpy ndarray if 2d list

        ## compute det bb with each BBGT

        if BBGT.size > 0:
            # compute overlaps
            # intersection

            # 1. calculate the overlaps between hbbs, if the iou between hbbs are 0, the iou between obbs are 0, too.
            # pdb.set_trace()
            BBGT_xmin =  np.min(BBGT[:, 0::2], axis=1)
            BBGT_ymin = np.min(BBGT[:, 1::2], axis=1)
            BBGT_xmax = np.max(BBGT[:, 0::2], axis=1)
            BBGT_ymax = np.max(BBGT[:, 1::2], axis=1)
            bb_xmin = np.min(bb[0::2])
            bb_ymin = np.min(bb[1::2])
            bb_xmax = np.max(bb[0::2])
            bb_ymax = np.max(bb[1::2])

            ixmin = np.maximum(BBGT_xmin, bb_xmin)
            iymin = np.maximum(BBGT_ymin, bb_ymin)
            ixmax = np.minimum(BBGT_xmax, bb_xmax)
            iymax = np.minimum(BBGT_ymax, bb_ymax)
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # union
            uni = ((bb_xmax - bb_xmin + 1.) * (bb_ymax - bb_ymin + 1.) +
                   (BBGT_xmax - BBGT_xmin + 1.) *
                   (BBGT_ymax - BBGT_ymin + 1.) - inters)

            overlaps = inters / uni

            BBGT_keep_mask = overlaps > 0
            BBGT_keep = BBGT[BBGT_keep_mask, :]
            BBGT_keep_index = np.where(overlaps > 0)[0]


            if len(BBGT_keep) > 0:
                overlaps = calcoverlaps(BBGT_keep, bb)
                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)
                gtbb = BBGT_keep[jmax]
                jmax = BBGT_keep_index[jmax]

        if ovmax > ovthresh:
            if not R['det'][jmax]:
                tp[d] = 1.
                R['det'][jmax] = 1
                pred_rot.append(utils.calculate_angle(bb))
                gt_rot.append(utils.calculate_angle(gtbb))

                bbs.append(bb)
                gtbbs.append(gtbb)
            else:
                fp[d] = 1.
        else:
            fp[d] = 1.

    # compute precision recall

    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    fn = n_gts - tp

    rec = tp / float(n_gts)
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)

    ap = voc_ap(rec, prec, use_07_metric)

    res = {
        'tp': int(tp[-1]),
        'fp': int(fp[-1]),
        'fn': int(fn[-1]),
        'gt': int(n_gts),
        'rec': rec,
        'prec': prec,
        'ap': ap,
        'pred_rot': pred_rot,
        'gt_rot': gt_rot,
        'gt_match': gtbbs,
        'pred_match': bbs
    }

    return res


def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap
