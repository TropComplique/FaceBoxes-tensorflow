import numpy as np
import tensorflow as tf


"""
For evaluation during the training i use average precision @ iou=0.5
like in PASCAL VOC Challenge (detection task):
http://host.robots.ox.ac.uk/pascal/VOC/voc2012/devkit_doc.pdf
"""


class Box:
    def __init__(self, image, box, score=None):
        """
        Arguments:
            image: a string, identifier of a image.
            box: a numpy float array with shape [4].
            score: a float number or None.
        """
        self.image = image
        self.confidence = score
        self.is_matched = False

        # top left corner
        self.ymin = box[0]
        self.xmin = box[1]

        # bottom right corner
        self.ymax = box[2]
        self.xmax = box[3]


class Evaluator:
    def __init__(self):
        self._initialize()

    def evaluate(self, iou_threshold=0.5):
        self.metrics = evaluate_detector(
            self.groundtruth_by_image,
            self.detections, iou_threshold
        )

    def clear(self):
        self._initialize()

    def get_metric_ops(self, image_name, groundtruth, predictions):
        """
        Arguments:
            image_name: a string tensor with shape [1].
            groundtruth: a dict with the following keys
                'boxes': a float tensor with shape [1, max_num_boxes, 4].
                'num_boxes': an int tensor with shape [1].
            predictions: a dict with the following keys
                'boxes': a float tensor with shape [1, max_num_boxes, 4].
                'scores': a float tensor with shape [1, max_num_boxes].
                'num_boxes': an int tensor with shape [1].
        """

        def update_op_func(image_name, gt_boxes, gt_num_boxes, boxes, scores, num_boxes):
            self.add_groundtruth(image_name, gt_boxes, gt_num_boxes)
            self.add_detections(image_name, boxes, scores, num_boxes)

        tensors = [
            image_name[0], groundtruth['boxes'][0], groundtruth['num_boxes'][0],
            predictions['boxes'][0], predictions['scores'][0], predictions['num_boxes'][0]
        ]
        update_op = tf.py_func(update_op_func, tensors, [], stateful=True)

        def evaluate_func():
            self.evaluate()
            self.clear()
        evaluate_op = tf.py_func(evaluate_func, [], [])

        def get_value_func(measure):
            def value_func():
                return np.float32(self.metrics[measure])
            return value_func

        with tf.control_dependencies([evaluate_op]):

            metric_names = ['AP', 'precision', 'recall', 'mean_iou', 'threshold', 'FP', 'FN']
            eval_metric_ops = {
                'metrics/' + measure:
                (tf.py_func(get_value_func(measure), [], tf.float32), update_op)
                for measure in metric_names
            }

        return eval_metric_ops

    def _initialize(self):
        self.detections = []
        self.groundtruth_by_image = {}

    def add_detections(self, image_name, boxes, scores, num_boxes):
        """
        Arguments:
            images: a numpy string array with shape [].
            boxes: a numpy float array with shape [N, 4].
            scores: a numpy float array with shape [N].
            num_boxes: a numpy int array with shape [].
        """
        boxes, scores = boxes[:num_boxes], scores[:num_boxes]
        for box, score in zip(boxes, scores):
            self.detections.append(Box(image_name, box, score))

    def add_groundtruth(self, image_name, boxes, num_boxes):
        for box in boxes[:num_boxes]:
            if image_name in self.groundtruth_by_image:
                self.groundtruth_by_image[image_name] += [Box(image_name, box)]
            else:
                self.groundtruth_by_image[image_name] = [Box(image_name, box)]


def evaluate_detector(groundtruth_by_img, all_detections, iou_threshold=0.5):
    """
    Arguments:
        groundtruth_by_img: a dict of lists with boxes,
            image -> list of groundtruth boxes on the image.
        all_detections: a list of boxes.
        iou_threshold: a float number.
    Returns:
        a dict with seven values.
    """

    # each ground truth box is either TP or FN
    n_groundtruth_boxes = 0

    for boxes in groundtruth_by_img.values():
        n_groundtruth_boxes += len(boxes)
    n_groundtruth_boxes = max(n_groundtruth_boxes, 1)

    # sort by confidence in decreasing order
    all_detections.sort(key=lambda box: box.confidence, reverse=True)

    n_correct_detections = 0
    n_detections = 0
    mean_iou = 0.0
    precision = [0.0]*len(all_detections)
    recall = [0.0]*len(all_detections)
    confidences = [box.confidence for box in all_detections]

    for k, detection in enumerate(all_detections):

        # each detection is either TP or FP
        n_detections += 1

        if detection.image in groundtruth_by_img:
            groundtruth_boxes = groundtruth_by_img[detection.image]
        else:
            groundtruth_boxes = []

        best_groundtruth_i, max_iou = match(detection, groundtruth_boxes)
        mean_iou += max_iou

        if best_groundtruth_i >= 0 and max_iou >= iou_threshold:
            box = groundtruth_boxes[best_groundtruth_i]
            if not box.is_matched:
                box.is_matched = True
                n_correct_detections += 1  # increase number of TP

        precision[k] = float(n_correct_detections)/float(n_detections)  # TP/(TP + FP)
        recall[k] = float(n_correct_detections)/float(n_groundtruth_boxes)  # TP/(TP + FN)

    ap = compute_ap(precision, recall)
    best_threshold, best_precision, best_recall = compute_best_threshold(
        precision, recall, confidences
    )
    mean_iou /= max(n_detections, 1)
    return {
        'AP': ap, 'precision': best_precision,
        'recall': best_recall, 'threshold': best_threshold,
        'mean_iou': mean_iou, 'FP': n_detections - n_correct_detections,
        'FN': n_groundtruth_boxes - n_correct_detections
    }


def compute_best_threshold(precision, recall, confidences):
    """
    Arguments:
        precision, recall, confidences: lists of floats of the same length.

    Returns:
        1. a float number, best confidence threshold.
        2. a float number, precision at the threshold.
        3. a float number, recall at the threshold.
    """
    if len(confidences) == 0:
        return 0.0, 0.0, 0.0

    precision = np.asarray(precision)
    recall = np.asarray(recall)
    confidences = np.asarray(confidences)

    diff = np.abs(precision - recall)
    prod = precision*recall
    best_i = np.argmax(prod*(1.0 - diff))
    best_threshold = confidences[best_i]

    return best_threshold, precision[best_i], recall[best_i]


def compute_iou(box1, box2):
    w = min(box1.xmax, box2.xmax) - max(box1.xmin, box2.xmin)
    if w > 0:
        h = min(box1.ymax, box2.ymax) - max(box1.ymin, box2.ymin)
        if h > 0:
            intersection = w*h
            w1 = box1.xmax - box1.xmin
            h1 = box1.ymax - box1.ymin
            w2 = box2.xmax - box2.xmin
            h2 = box2.ymax - box2.ymin
            union = (w1*h1 + w2*h2) - intersection
            return float(intersection)/float(union)
    return 0.0


def match(detection, groundtruth_boxes):
    """
    Arguments:
        detection: a box.
        groundtruth_boxes: a list of boxes.
    Returns:
        best_i: an integer, index of the best groundtruth box.
        max_iou: a float number.
    """
    best_i = -1
    max_iou = 0.0
    for i, box in enumerate(groundtruth_boxes):
        iou = compute_iou(detection, box)
        if iou > max_iou:
            best_i = i
            max_iou = iou
    return best_i, max_iou


def compute_ap(precision, recall):
    previous_recall_value = 0.0
    ap = 0.0
    # recall is in increasing order
    for p, r in zip(precision, recall):
        delta = r - previous_recall_value
        ap += p*delta
        previous_recall_value = r
    return ap
