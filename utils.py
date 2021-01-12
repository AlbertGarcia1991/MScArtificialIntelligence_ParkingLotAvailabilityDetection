########################################################################################################################
#                               MSc Artificial Intelligence - City, University of London                               #
#                                    INM705 - Garcia Plaza, Albert / Bohkary, Syed                                     #
########################################################################################################################

import random
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision


def parse_model_cfg(path):
    """
    Parse the model .config file, where the number of layers, their types and parameters are specified.
    Arguments of the function are:
        :param path: directory to the .cfg file.
    And the output:
        :return mdefs: list of dictionaries. Each item on the array is a layer, and this item is a dictionary containing
            all the layer's properties (type, size, batch_size, and so on depending on each layer kind).
    """
    # Open the .cfg file, parse and post-process its lines
    with open(path, 'r') as f:
        lines = f.read().split('\n')
    lines = [x for x in lines if x and not x.startswith('#')]  # omit comment and black lines
    lines = [x.rstrip().lstrip() for x in lines]  # get rid of fringe whitespaces

    mdefs = []  # module definitions
    for line in lines:
        # All layer block starts with '[' symbol, where the layer type is contained between these square brackets
        #   (e.g. [convolutional] or [route]). Then, all layer's parameters are listed below line by line.
        if line.startswith('['):
            mdefs.append({})  # each new bloc, create a dictionary to store its parameters
            mdefs[-1]['type'] = line[1:-1].rstrip()  # save the layer's 'type' given between the square brackets
            if mdefs[-1]['type'] == 'convolutional':
                mdefs[-1]['batch_normalize'] = 0  # if convolutional, pre-populate its batch normalization as all False

        # When the line is not containing the layer's type, it is containing the layer's parameters.
        else:
            key, val = line.split("=")  # all parameters are specified following the format: "param_name = param_value"
            key = key.rstrip()
            # If the layer's parameter is 'anchors' (on YOLO layers), store the parameter's values as numpy array.
            if key == 'anchors':
                mdefs[-1][key] = np.array([float(x) for x in val.split(',')]).reshape((-1, 2))
            # Is the layer's parameter is either 'layers' or 'mask', store the parameter's values as Python list.
            elif key in ['layers', 'mask']:
                mdefs[-1][key] = [int(x) for x in val.split(',')]
            # For other layer's parameters store as number if it is numeric (int or float), otherwise as string    .
            else:
                val = val.strip()
                if val.isnumeric():  # return int or float
                    mdefs[-1][key] = int(val) if (int(val) - float(val)) == 0 else float(val)
                else:
                    mdefs[-1][key] = val  # return string

    return mdefs


def parse_data_cfg(path):
    with open(path, 'r') as f:
        lines = f.readlines()

    options = dict()
    for line in lines:
        line = line.strip()
        if line == '' or line.startswith('#'):
            continue
        else:
            key, val = line.split('=')
            options[key.strip()] = val.strip()

    return options


def load_classes(path):
    """Loads *.names file at 'path'"""
    with open(path, 'r') as f:
        names = f.read().split('\n')

    return list(filter(None, names))  # filter removes empty strings (such as last line)


def xyxy2xywh(x):
    """
    Convert the given image from xyxy to xywh format. Arguments of the function are:
    :param x: input image in xyxy format to be converted.
    And the output:
    :return y: output image converted to xywh format.
    """
    # Depending on the input datatype (Torch tensor or Numpy array), initialise the output tensor with zeros.
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)

    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height

    return y


def xywh2xyxy(x):
    """
    Convert the given image from xyxy to xywh format. Arguments of the function are:
    :param x: input image in xyxy format to be converted.
    And the output:
    :return y: output image converted to xywh format.
    """
    # Depending on the input datatype (Torch tensor or Numpy array), initialise the output tensor with zeros.
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)

    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y

    return y


def ap_per_class(tp, conf, pred_cls, target_cls):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:    True positives (nparray, nx1 or nx10).
        conf:  Objectness value from 0-1 (nparray).
        pred_cls: Predicted object classes (nparray).
        target_cls: True object classes (nparray).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls)

    # Create Precision-Recall curve and compute AP for each class
    pr_score = 0.1  # score to evaluate P and R https://github.com/ultralytics/yolov3/issues/898
    s = [len(unique_classes), tp.shape[1]]  # number class, number iou thresholds (i.e. 10 for mAP0.5...0.95)
    ap, p, r = np.zeros(s), np.zeros(s), np.zeros(s)
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_gt = (target_cls == c).sum()  # Number of ground truth objects
        n_p = i.sum()  # Number of predicted objects

        if n_p == 0 or n_gt == 0:
            continue
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum(0)
            tpc = tp[i].cumsum(0)

            # Recall
            recall = tpc / (n_gt + 1e-16)  # recall curve
            r[ci] = np.interp(-pr_score, -conf[i], recall[:, 0])  # r at pr_score, negative x, xp because xp decreases

            # Precision
            precision = tpc / (tpc + fpc)  # precision curve
            p[ci] = np.interp(-pr_score, -conf[i], precision[:, 0])  # p at pr_score

            # AP from recall-precision curve
            for j in range(tp.shape[1]):
                ap[ci, j] = compute_ap(recall[:, j], precision[:, j])

    # Compute F1 score (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + 1e-16)

    return p, r, ap, f1, unique_classes.astype('int32')


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.], recall, [min(recall[-1] + 1E-3, 1.)]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = 'interp'  # methods: 'continuous', 'interp'
    if method == 'interp':
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap


def bbox_iou(box1, box2, x1y1x2y2=True, iou_method=None):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.t()

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
    union = (w1 * h1 + 1e-16) + w2 * h2 - inter

    iou = inter / union  # iou
    if iou_method:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height

        if iou_method == 'g':
            c_area = cw * ch + 1e-16  # convex area
            return iou - (c_area - union) / c_area  # GIoU

        else:
            # convex diagonal squared
            c2 = cw ** 2 + ch ** 2 + 1e-16
            # center-point distance squared
            rho2 = ((b2_x1 + b2_x2) - (b1_x1 + b1_x2)) ** 2 / 4 + ((b2_y1 + b2_y2) - (b1_y1 + b1_y2)) ** 2 / 4
            if iou_method == 'd':
                return iou - rho2 / c2  # DIoU

            elif iou_method == 'c':
                v = (4 / np.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / (1 - iou + v)
                return iou - (rho2 / c2 + v * alpha)  # CIoU

    return iou


def box_iou(boxes1, boxes2):
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        boxes1 (Tensor[N, 4])
        boxes2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(boxes1.t())
    area2 = box_area(boxes2.t())

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    inter = (rb - lt).clamp(0).prod(2)  # [N,M]
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)


def wh_iou(wh1, wh2):
    """
    Returns the IoU metric between to square boxes with shared centre-point.
    Arguments:
    :param wh1: box 1.
    :param wh2: box 2.
    Outputs:
    :return: intersection over union (IoU) metric between the two boxes.
    """
    wh1 = wh1[:, None]  # [N,1,2]
    wh2 = wh2[None]  # [1,M,2]
    inter = torch.min(wh1, wh2).prod(2)  # [N,M]
    union = (wh1.prod(2) + wh2.prod(2) - inter)
    iou = inter / union
    return iou


def compute_loss(p, targets, model, iou_thres, iou_method=None):
    """
    Compute the loss comparing the predicted bounding boxes and the labeled ones (ground truth).
    Arguments:
        :param p: predicted bounding boxes for all images in the batch.
        :param targets: ground truth bounding boxes of all images in the batch.
        :param model: Darknet network.
    Outputs:
        :return loss:
        :return loss_items:
    """
    ft = torch.cuda.FloatTensor
    lcls, lbox, lobj = ft([0]), ft([0]), ft([0])  # initialise the cls, box and obj losses Torch's variables
    tcls, tbox, indices, anchor_vec = build_targets(model, targets, iou_thres)  # get the ground truth bounding boxes

    # Define computing algorithm for class and objectness losses.
    BCEcls = nn.BCEWithLogitsLoss(pos_weight=ft([1.0]), reduction='mean')
    BCEobj = nn.BCEWithLogitsLoss(pos_weight=ft([1.0]), reduction='mean')

    # Compute losses
    np, ng = 0, 0  # number grid points, targets
    for i, pi in enumerate(p):  # layer index, layer predictions
        b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
        tobj = torch.zeros_like(pi[..., 0])  # target obj
        np += tobj.numel()

        # Compute losses
        nb = len(b)
        if nb:  # number of targets
            ng += nb
            ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

            # GIoU
            pxy = torch.sigmoid(ps[:, 0:2])  # pxy = pxy * s - (s - 1) / 2,  s = 1.5  (scale_xy)
            pwh = torch.exp(ps[:, 2:4]).clamp(max=1E3) * anchor_vec[i]
            pbox = torch.cat((pxy, pwh), 1)  # predicted box
            giou = bbox_iou(pbox.t(), tbox[i], x1y1x2y2=False, iou_method=iou_method)  # giou computation
            lbox += (1.0 - giou).mean()  # giou loss
            tobj[b, a, gj, gi] = 1.0

            if model.nc > 1:  # cls loss (only if multiple classes)
                t = torch.full_like(ps[:, 5:], 0.0)  # targets
                t[range(nb), tcls[i]] = 1.0
                lcls += BCEcls(ps[:, 5:], t)  # BCE

        lobj += BCEobj(pi[..., 4], tobj)  # obj loss

    lbox *= 3.54
    lobj *= 64.3
    lcls *= 37.4

    loss = lbox + lobj + lcls
    loss_items = torch.cat((lbox, lobj, lcls, loss)).detach()

    return loss, loss_items


def build_targets(model, targets, iou_thres):
    """
    Build bounding boxes of the ground truth.
    Arguments:
        :param model: Darknet model.
        :param targets: Predicted targets during the forward pass.
        :param iou_thres: minimum IoU to be accepted as truth bounding boxes.
    Outputs:
        :return tcls:
        :return tbox:
        :return indices:
        :return av:
     """
    nt = targets.shape[0]  # number of predicted objects
    tcls, tbox, indices, av = [], [], [], []  # output variables initialisation

    # Detections at each YOLO layer
    for i in model.yolo_layers:
        # Get number of grid points and anchors for the current YOLO layer.
        ng, anchor_vec = model.module_list[i].ng, model.module_list[i].anchor_vec

        t, a = targets, []  # store the ground truth targets (t)
        gwh = t[:, 4:6] * ng  # set the grid width and height

        if nt:  # if there is any ground truth target to be detected
            iou = wh_iou(anchor_vec, gwh)  # obtain iou metric between anchors vec and ground truth boxes

            na = anchor_vec.shape[0]  # number of anchors
            a = torch.arange(na).view((-1, 1)).repeat([1, nt]).view(-1)
            t = targets.repeat([na, 1])
            gwh = gwh.repeat([na, 1])

            # reject anchors below iou_thres
            j = iou.view(-1) > iou_thres  # mask with accepted bouding boxes
            t, a, gwh = t[j], a[j], gwh[j]

        # Indices
        b, c = t[:, :2].long().t()  # target image, class
        gxy = t[:, 2:4] * ng  # grid x, y
        gi, gj = gxy.long().t()  # grid x, y indices
        indices.append((b, a, gj, gi))

        # Box
        gxy -= gxy.floor()  # xy
        tbox.append(torch.cat((gxy, gwh), 1))  # xywh (grids)
        av.append(anchor_vec[a])  # anchor vec

        # Class
        tcls.append(c)

    return tcls, tbox, indices, av


def non_max_suppression(prediction, conf_thres=0.1, iou_thres=0.6):
    """
    Performs  Non-Maximum Suppression on inference results
    Returns detections with shape:
        nx6 (x1, y1, x2, y2, conf, cls)
    """

    # Box constraints
    min_wh, max_wh = 2, 100  # (pixels) minimum and maximum box width and height

    nc = prediction[0].shape[1] - 5  # number of classes
    output = [None] * len(prediction)
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply conf constraint
        x = x[x[:, 4] > conf_thres]

        # Apply width-height constraint
        x = x[((x[:, 2:4] > min_wh) & (x[:, 2:4] < max_wh)).all(1)]

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[..., 5:] *= x[..., 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        i, j = (x[:, 5:] > conf_thres).nonzero().t()
        x = torch.cat((box[i], x[i, j + 5].unsqueeze(1), j.float().unsqueeze(1)), 1)

        # Apply finite constraint
        if not torch.isfinite(x).all():
            x = x[torch.isfinite(x).all(1)]

        # If none remain process next image
        n = x.shape[0]  # number of boxes
        if not n:
            continue

        # Batched NMS
        c = x[:, 5]  # classes
        boxes, scores = x[:, :4].clone() + c.view(-1, 1) * max_wh, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.boxes.nms(boxes, scores, iou_thres)
        if n < 1000:  # update boxes
            iou = box_iou(boxes, boxes).tril_()  # lower triangular iou matrix
            weights = (iou > iou_thres) * scores.view(-1, 1)
            weights /= weights.sum(0)
            x[:, :4] = torch.mm(weights.T, x[:, :4])  # merged_boxes(n,4) = weights(n,n) * boxes(n,4)
        output[xi] = x[i]

    return output


def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0].clamp_(0, img_shape[1])  # x1
    boxes[:, 1].clamp_(0, img_shape[0])  # y1
    boxes[:, 2].clamp_(0, img_shape[1])  # x2
    boxes[:, 3].clamp_(0, img_shape[0])  # y2


def scale_coords(img1_shape, coords, img0_shape):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    gain = max(img1_shape) / max(img0_shape)  # gain  = old / new
    pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


def fitness(x):
    # Returns fitness (for use with results.txt)
    w = [0.0, 0.01, 0.99, 0.00]  # weights for [P, R, mAP, F1]@0.5 or [P, R, mAP@0.5, mAP@0.5:0.95]
    return (x[:, :4] * w).sum(1)


def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    offset = 1.2
    c1, c2 = (int(x[0]), int(x[1] * offset)), (int(x[2]), int(x[3] * offset))
    cv2.rectangle(img, c1, c2, color, thickness=tl)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
