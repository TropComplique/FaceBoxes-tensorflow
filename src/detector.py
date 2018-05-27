import tensorflow as tf
import tensorflow.contrib.slim as slim
import math

from src.constants import MATCHING_THRESHOLD, PARALLEL_ITERATIONS, BATCH_NORM_MOMENTUM
from src.utils import batch_non_max_suppression, batch_decode
from src.training_target_creation import get_training_targets
from src.losses_and_ohem import localization_loss, classification_loss, apply_hard_mining


class Detector:
    def __init__(self, images, feature_extractor, anchor_generator):
        """
        Arguments:
            images: a float tensor with shape [batch_size, 3, height, width],
                a batch of RGB images with pixel values in the range [0, 1].
            feature_extractor: an instance of FeatureExtractor.
            anchor_generator: an instance of AnchorGenerator.
        """
        feature_maps = feature_extractor(images)
        self.is_training = feature_extractor.is_training
        
        # sometimes images will be of different sizes,
        # so i need use dynamic shape
        h, w = images.shape.as_list()[2:]
        if h is None or w is None:
            h, w = tf.shape(images)[2], tf.shape(images)[3]

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
            # it has shape [batch_size, num_anchors, 4]

            class_predictions_with_background = tf.nn.softmax(
                self.class_predictions_with_background, axis=2
            )
            class_predictions_without_background = tf.slice(
                class_predictions_with_background,
                [0, 0, 1], [-1, -1, -1]
            )
            scores = tf.squeeze(class_predictions_without_background, axis=2)
            # it has shape [batch_size, num_anchors]

        with tf.device('/cpu:0'), tf.name_scope('nms'):
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
            cls_targets = tf.one_hot(cls_targets, depth=2, axis=2, dtype=tf.float32)

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

            self._add_scalewise_matches_summaries(weights)
            self._add_scalewise_summaries(cls_losses, name='classification_losses')
            self._add_scalewise_summaries(location_losses, name='localization_losses')
            tf.summary.scalar('total_mean_matches_per_image', tf.reduce_mean(matches_per_image))
                
            with tf.name_scope('ohem'):
                location_loss, cls_loss = apply_hard_mining(
                    location_losses, cls_losses,
                    self.class_predictions_with_background,
                    self.box_encodings, matches, self.anchors,
                    loss_to_use=params['loss_to_use'],
                    loc_loss_weight=params['loc_loss_weight'],
                    cls_loss_weight=params['cls_loss_weight'],
                    num_hard_examples=params['num_hard_examples'],
                    nms_threshold=params['nms_threshold'],
                    max_negatives_per_positive=params['max_negatives_per_positive'],
                    min_negatives_per_image=params['min_negatives_per_image']
                )
                return {'localization_loss': location_loss/normalizer, 'classification_loss': cls_loss/normalizer}

    def _add_scalewise_summaries(self, tensor, name):
        """Adds histograms of the biggest 20 percent of
        tensor's values for each scale (feature map).

        Arguments:
            tensor: a float tensor with shape [batch_size, num_anchors].
            name: a string.
        """
        index = 0
        for i, n in enumerate(self.num_anchors_per_feature_map):

            k = tf.ceil(tf.to_float(n) * 0.20)  # top 20%
            k = tf.to_int32(k)

            biggest_values, _ = tf.nn.top_k(tensor[:, index:(index + n)], k, sorted=False)
            # it has shape [batch_size, k]
            tf.summary.histogram(
                name + '_on_scale_' + str(i),
                tf.reduce_mean(biggest_values, axis=0)
            )
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
                [batch, channels_i, height_i, width_i].

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
                    data_format='NCHW', padding='SAME'
                )
                # it has shape [batch_size, num_predictions_per_location * 4, height_i, width_i]
                box_encodings.append(y)

                y = slim.conv2d(
                    x, num_predictions_per_location * 2,
                    [3, 3], activation_fn=None, scope='class_predictor_%d' % i,
                    data_format='NCHW', padding='SAME'
                )
                # it has  shape [batch_size, num_predictions_per_location * 2, height_i, width_i]
                class_predictions_with_background.append(y)

        # it is important that reshaping here is the same as when anchors were generated
        with tf.name_scope('reshaping'):
            for i in range(num_feature_maps):

                x = feature_maps[i]
                num_predictions_per_location = num_anchors_per_location[i]
                batch_size = tf.shape(x)[0]
                height_i = tf.shape(x)[2]
                width_i = tf.shape(x)[3]
                num_anchors_on_feature_map = height_i * width_i * num_predictions_per_location

                y = box_encodings[i]
                y = tf.transpose(y, perm=[0, 2, 3, 1])
                y = tf.reshape(y, tf.stack([batch_size, height_i, width_i, num_predictions_per_location, 4]))
                box_encodings[i] = tf.reshape(y, [batch_size, num_anchors_on_feature_map, 4])

                y = class_predictions_with_background[i]
                y = tf.transpose(y, perm=[0, 2, 3, 1])
                y = tf.reshape(y, [batch_size, height_i, width_i, num_predictions_per_location, 2])
                class_predictions_with_background[i] = tf.reshape(y, tf.stack([batch_size, num_anchors_on_feature_map, 2]))
            
            self.box_encodings = tf.concat(box_encodings, axis=1)
            self.class_predictions_with_background = tf.concat(class_predictions_with_background, axis=1)
