from mmengine.evaluator import Evaluator
import numpy as np
import sys

'''
precision = TP / (TP + FP)[预测为P的总数]
recall = TP / (TP + FN) [P的总数]
'''


def iou(boxA, boxB):
    #计算重合部分的上下左右4个边的值，注意最大最小函数的使用
    left_max = max(boxA[0], boxB[0])
    top_max = max(boxA[1], boxB[1])
    right_min = min(boxA[2], boxB[2])
    bottom_min = min(boxA[3], boxB[3])
    # 计算重合部分面积
    inter = max(0, (right_min-left_max) * max(0, (bottom_min-top_max)))
    Sa = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    Sb = (boxB[2]-boxB[0]) * (boxB[3]-boxB[2])
    # 计算所有区域的面积并计算IOU值，如果python是2，则增加浮点化操作
    uniou = Sa+Sb-inter
    iou =inter / uniou
    return iou


# 类别评测过程中的AP求解过程，det_boxes:包含全部图像中所有类别的预测框，其中一个预测框包含
# [left, top, right, bottom, score, NameofImage]
# gt_boxes:包含全部图像中所有类别的标签，其中一个标签的内容为[left, top, right, bottom, 0]。
# 需要注意的是，最后一位0代表该标签有没有被匹配过，如果匹配过这将会置为1，其他预测框再去匹配则为误检框。
# 通过计算IoU计算TP和FP

def cal_mAP(classes, det_boxes, gt_boxes, num_pos, cfg):
    for c in classes:
        # 通过类别作为关键字，得到每个类别的预测、标签及总的标签数
        dects = det_boxes[c]
        gt_class = gt_boxes[c]
        npos = num_pos[c]
        # 利用得分作为关键字，对预测框按照得分从高到低进行排序
        dects = sorted(dects, key=lambda conf: conf[5], reverse=True)
        # 设置两个与预测边框长度相同的列表，标记是True Psoitive还是False Positive
        TP = np.zeros(len(dects))
        FP = np.zeros(len(dects))
        # 对某一个类别的预测框进行遍历
        for d in range(len(dects)):
            # 将IOU默认置为最低
            iouMax = sys.float_info.min
            # 遍历与预测框同一图像中的同一类别的标签，计算IOU
            if dects[d][-1] in gt_class:
                for j in range(len(gt_class[dects[d][-1]])):
                    iou = Evaluator.iou(dects[d][:4], gt_class[dects[d][-1]][j][:4])
                    if iou > iouMax:
                        iouMax = iou
                        jmax = j  # 记录与预测有最大IOU的标签

                # 如果最大IOU大于阈值，并且没有匹配过，这赋予TP
                if iou >= cfg['iouThreshold']:
                    if gt_class[dects[d][-1]][jmax][4] == 0:
                        TP[d] = 1
                        gt_class[dects[d][-1]][jmax][4] = 1
                    # 如果被匹配，赋予FP
                    else:
                        FP[d] = 1
                # 如果最大IOU没有超过阈值，赋予FP
                else:
                    FP[d] = 1
            # 如果对于图像没有该类别的标签，赋予FP
            else:
                FP[d] = 1
        # 利用Numpy中的cumsum()函数，计算累计的FP与TP
        acc_FP = np.cumsum(FP)
        acc_TP = np.cumsum(TP)
        # 得到没个点的recall
        rec = acc_TP / npos
        prec = np.divide(acc_TP, (acc_FP + acc_TP))
        # 利用Recall与precision进一步计算得到Ap
        [ap, mpre, mrec, ii] = Evaluator.CalculateAveragePrecision(rec, prec)