import numpy as np

'''
1、按置信度进行排序
2、计算IoU
3、IoU大于阈值则删去
'''


def nms(boxes, iou_thres):  # 去除冗余框
    """ 非极大值抑制 """
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    scores = boxes[:, 4]
    areas = (x2-x1) * (y2-y1)
    keep = []
    # 按置信度进行排序
    index = np.argsort(scores)[::-1]  # 排序返回对应下标
    while index.size > 1:
        # 置信度最高的框
        i = index[0]
        keep.append(index[0])
        # 计算交集左下角与右上角坐标
        inter_x1 = np.maximum(x1[i], x1[index[1:]])
        inter_y1 = np.maximum(y1[i], y1[index[1:]])
        inter_x2 = np.minimum(x2[i], x2[index[1:]])
        inter_y2 = np.minimum(y2[i], y2[index[1:]])
        # 计算交集的面积
        inter_area = np.maximum(inter_x2-inter_x1, 0) * np.maximum(inter_y2-inter_y1, 0)
        # 计算当前框与其余框的iou
        iou = inter_area / (areas[index[1:]] + areas[i] - inter_area)
        ids = np.where(iou < iou_thres)[0]
        index = index[ids+1]
    return boxes[keep]
