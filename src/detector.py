import tensorflow as tf
import tensorflow.contrib.slim as slim
import math

from src.constants import MATCHING_THRESHOLD, PARALLEL_ITERATIONS, BATCH_NORM_MOMENTUM, RESIZE_METHOD
from src.utils import batch_non_max_suppression, batch_decode
from src.training_target_creation import get_training_targets
from src.losses_and_ohem import localization_loss, classification_loss, apply_hard_mining


class Detector:
    def __init__(self, images, feature_extractor, anchor_generator):
        """
        Arguments:
            images: a float tensor with shape [batch_size, height, width, 3],
                a batch of RGB images with pixel values in the range [0, 1].
            feature_extractor: an instance of FeatureExtractor.
            anchor_generator: an instance of AnchorGenerator.
        """

        # sometimes images will be of different sizes,
        # so i need to use the dynamic shape
        h, w = images.shape.as_list()[1:3]

        # image padding here is very tricky and important part of the detector,
        # if we don't do it then some bounding box
        # predictions will be badly shifted!

        x = 128  # mysterious parameter
        # (actually, it is the stride of the last layer)

        self.box_scaler = tf.ones([4], dtype=tf.float32)
        if h is None or w is None or h % x != 0 or w % x != 0:
            h, w = tf.shape(images)[1], tf.shape(images)[2]
            with tf.name_scope('image_padding'):

                # image size must be divisible by 128
                new_h = x * tf.to_int32(tf.ceil(h/x))
                new_w = x * tf.to_int32(tf.ceil(w/x))
                # also we will need to rescale bounding box coordinates
                self.box_scaler = tf.to_float(tf.stack([
                    h/new_h, w/new_w, h/new_h, w/new_w
                ]))
                # pad the images with zeros on the right and on the bottom
                images = tf.image.pad_to_bounding_box(
                    images, offset_height=0, offset_width=0,
                    target_height=new_h, target_width=new_w
                )
                h, w = new_h, new_w

        feature_maps = feature_extractor(images)
        self.is_training = feature_extractor.is_training

        self.anchors = anchor_generator(feature_maps, image_size=(w, h))
        self.num_anchors_per_location = anchor_generator.num_anchors_per_location
        self.num_anchors_per_feature_map = anchor_generator.num_anchors_per_feature_map
        self._add_box_predictions(feature_maps)

    def get_predictions(self, score_threshold=0.1, iou_threshold=0.6, max_boxes=20):
        """Postprocess outputs of the network.

        Returns:
            boxes: a float tensor with shape [batch_size, N, 4].
            scores: a float tensor with shape [batch_size, N].
            num_boxes: an int tensor with shape [batch_size], it
                represents the number of detections on an image.

            where N = max_boxes.
        """
        with tf.name_scope('postprocessing'):
            boxes = batch_decode(self.box_encodings, self.anchors)
            # if the images were padded we need to rescale predicted boxes:
            boxes = boxes / self.box_scaler
            boxes = tf.clip_by_value(boxes, 0.0, 1.0)
            # it has shape [batch_size, num_anchors, 4]

            scores = tf.nn.softmax(self.class_predictions_with_background, axis=2)[:, :, 1]
            # it has shape [batch_size, num_anchors]

        with tf.name_scope('nms'):
            boxes, scores, num_detections = batch_non_max_suppression(
                boxes, scores, score_threshold, iou_threshold, max_boxes
            )
        return {'boxes': boxes, 'scores': scores, 'num_boxes': num_detections}

    def loss(self, groundtruth, params):
        """Compute scalar loss tensors with respect to provided groundtruth.

        Arguments:
            groundtruth: a dict with the following keys
                'boxes': a float tensor with shape [batch_size, max_num_boxes, 4].
                'num_boxes': an int tensor with shape [batch_size].
                    where max_num_boxes = max(num_boxes).
            params: a dict with parameters for OHEM.
        Returns:
            two float tensors with shape [].
        """
        reg_targets, matches = self._create_targets(groundtruth)

        with tf.name_scope('losses'):

            # whether anchor is matched
            is_matched = tf.greater_equal(matches, 0)
            weights = tf.to_float(is_matched)
            # shape [batch_size, num_anchors]

            # we have binary classification for each anchor
            cls_targets = tf.to_int32(is_matched)

            with tf.name_scope('classification_loss'):
                cls_losses = classification_loss(
                    self.class_predictions_with_background,
                    cls_targets
                )
            with tf.name_scope('localization_loss'):
                location_losses = localization_loss(
                    self.box_encodings,
                    reg_targets, weights
                )
            # they have shape [batch_size, num_anchors]

            with tf.name_scope('normalization'):
                matches_per_image = tf.reduce_sum(weights, axis=1)  # shape [batch_size]
                num_matches = tf.reduce_sum(matches_per_image)  # shape []
                normalizer = tf.maximum(num_matches, 1.0)

            scores = tf.nn.softmax(self.class_predictions_with_background, axis=2)
            # it has shape [batch_size, num_anchors, 2]

            decoded_boxes = batch_decode(self.box_encodings, self.anchors)
            decoded_boxes = decoded_boxes / self.box_scaler
            # it has shape [batch_size, num_anchors, 4]

            # add summaries for predictions
            is_background = tf.equal(matches, -1)
            self._add_scalewise_histograms(tf.to_float(is_background) * scores[:, :, 0], 'background_probability')
            self._add_scalewise_histograms(weights * scores[:, :, 1], 'face_probability')
            ymin, xmin, ymax, xmax = tf.unstack(decoded_boxes, axis=2)
            h, w = ymax - ymin, xmax - xmin
            self._add_scalewise_histograms(weights * h, 'box_heights')
            self._add_scalewise_histograms(weights * w, 'box_widths')

            # add summaries for losses and matches
            self._add_scalewise_matches_summaries(weights)
            self._add_scalewise_summaries(cls_losses, name='classification_losses')
            self._add_scalewise_summaries(location_losses, name='localization_losses')
            tf.summary.scalar('total_mean_matches_per_image', tf.reduce_mean(matches_per_image))

            with tf.name_scope('ohem'):
                location_loss, cls_loss = apply_hard_mining(
                    location_losses, cls_losses,
                    self.class_predictions_with_background,
                    matches, decoded_boxes,
                    loss_to_use=params['loss_to_use'],
                    loc_loss_weight=params['loc_loss_weight'],
                    cls_loss_weight=params['cls_loss_weight'],
                    num_hard_examples=params['num_hard_examples'],
                    nms_threshold=params['nms_threshold'],
                    max_negatives_per_positive=params['max_negatives_per_positive'],
                    min_negatives_per_image=params['min_negatives_per_image']
                )
                return {'localization_loss': location_loss/normalizer, 'classification_loss': cls_loss/normalizer}

    def _add_scalewise_summaries(self, tensor, name, percent=0.2):
        """Adds histograms of the biggest 20 percent of
        tensor's values for each scale (feature map).

        Arguments:
            tensor: a float tensor with shape [batch_size, num_anchors].
            name: a string.
            percent: a float number, default value is 20%.
        """
        index = 0
        for i, n in enumerate(self.num_anchors_per_feature_map):
            k = tf.ceil(tf.to_float(n) * percent)
            k = tf.to_int32(k)
            biggest_values, _ = tf.nn.top_k(tensor[:, index:(index + n)], k, sorted=False)
            # it has shape [batch_size, k]
            tf.summary.histogram(
                name + '_on_scale_' + str(i),
                tf.reduce_mean(biggest_values, axis=0)
            )
            index += n

    def _add_scalewise_histograms(self, tensor, name):
        """Adds histograms of the tensor's nonzero values for each scale (feature map).

        Arguments:
            tensor: a float tensor with shape [batch_size, num_anchors].
            name: a string.
        """
        index = 0
        for i, n in enumerate(self.num_anchors_per_feature_map):
            values = tf.reshape(tensor[:, index:(index + n)], [-1])
            nonzero = tf.greater(values, 0.0)
            values = tf.boolean_mask(values, nonzero)
            tf.summary.histogram(name + '_on_scale_' + str(i), values)
            index += n

    def _add_scalewise_matches_summaries(self, weights):
        """Adds summaries for the number of matches on each scale."""
        index = 0
        for i, n in enumerate(self.num_anchors_per_feature_map):
            matches_per_image = tf.reduce_sum(weights[:, index:(index + n)], axis=1)
            tf.summary.scalar(
                'mean_matches_per_image_on_scale_' + str(i),
                tf.reduce_mean(matches_per_image, axis=0)
            )
            index += n

    def _create_targets(self, groundtruth):
        """
        Arguments:
            groundtruth: a dict with the following keys
                'boxes': a float tensor with shape [batch_size, N, 4].
                'num_boxes': an int tensor with shape [batch_size].
        Returns:
            reg_targets: a float tensor with shape [batch_size, num_anchors, 4].
            matches: an int tensor with shape [batch_size, num_anchors].
        """
        def fn(x):
            boxes, num_boxes = x
            boxes = boxes[:num_boxes]
            # if the images are padded we need to rescale groundtruth boxes:
            boxes = boxes * self.box_scaler
            reg_targets, matches = get_training_targets(
                self.anchors, boxes, threshold=MATCHING_THRESHOLD
            )
            return reg_targets, matches

        with tf.name_scope('target_creation'):
            reg_targets, matches = tf.map_fn(
                fn, [groundtruth['boxes'], groundtruth['num_boxes']],
                dtype=(tf.float32, tf.int32),
                parallel_iterations=PARALLEL_ITERATIONS,
                back_prop=False, swap_memory=False, infer_shape=True
            )
            return reg_targets, matches

    def _add_box_predictions(self, feature_maps):
        """Adds box predictors to each feature map, reshapes, and returns concatenated results.

        Arguments:
            feature_maps: a list of float tensors where the ith tensor has shape
                [batch, height_i, width_i, channels_i].

        It creates two tensors:
            box_encodings: a float tensor with shape [batch_size, num_anchors, 4].
            class_predictions_with_background: a float tensor with shape
                [batch_size, num_anchors, 2].
        """
        num_anchors_per_location = self.num_anchors_per_location
        num_feature_maps = len(feature_maps)
        box_encodings, class_predictions_with_background = [], []

        with tf.variable_scope('prediction_layers'):
            for i in range(num_feature_maps):

                x = feature_maps[i]
                num_predictions_per_location = num_anchors_per_location[i]

                y = slim.conv2d(
                    x, num_predictions_per_location * 4,
                    [3, 3], activation_fn=None, scope='box_encoding_predictor_%d' % i,
                    data_format='NHWC', padding='SAME'
                )
                # it has shape [batch_size, height_i, width_i, num_predictions_per_location * 4]
                box_encodings.append(y)

                import numpy as np
                biases = np.zeros([num_predictions_per_location, 2], dtype='float32')
                biases[:, 0] = np.log(0.99)  # background class
                biases[:, 1] = np.log(0.01)  # object class
                biases = biases.reshape(num_predictions_per_location * 2)

                y = slim.conv2d(
                    x, num_predictions_per_location * 2,
                    [3, 3], activation_fn=None, scope='class_predictor_%d' % i,
                    data_format='NHWC', padding='SAME',
                    biases_initializer=tf.constant_initializer(biases)
                )
                # it has  shape [batch_size, height_i, width_i, num_predictions_per_location * 2]
                class_predictions_with_background.append(y)

        # it is important that reshaping here is the same as when anchors were generated
        with tf.name_scope('reshaping'):
            for i in range(num_feature_maps):

                x = feature_maps[i]
                num_predictions_per_location = num_anchors_per_location[i]
                batch_size = tf.shape(x)[0]
                height_i = tf.shape(x)[1]
                width_i = tf.shape(x)[2]
                num_anchors_on_feature_map = height_i * width_i * num_predictions_per_location

                y = box_encodings[i]
                y = tf.reshape(y, tf.stack([batch_size, height_i, width_i, num_predictions_per_location, 4]))
                box_encodings[i] = tf.reshape(y, [batch_size, num_anchors_on_feature_map, 4])

                y = class_predictions_with_background[i]
                y = tf.reshape(y, [batch_size, height_i, width_i, num_predictions_per_location, 2])
                class_predictions_with_background[i] = tf.reshape(y, tf.stack([batch_size, num_anchors_on_feature_map, 2]))

            self.box_encodings = tf.concat(box_encodings, axis=1)
            self.class_predictions_with_background = tf.concat(class_predictions_with_background, axis=1)
