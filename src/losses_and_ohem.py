import tensorflow as tf
from src.utils import batch_decode


"""
Note that we have only one label (it is 'face'),
so num_classes = 1.
"""


def localization_loss(predictions, targets, weights):
    """A usual L1 smooth loss.

    Arguments:
        predictions: a float tensor with shape [batch_size, num_anchors, 4],
            representing the (encoded) predicted locations of objects.
        targets: a float tensor with shape [batch_size, num_anchors, 4],
            representing the regression targets.
        weights: a float tensor with shape [batch_size, num_anchors].
    Returns:
        a float tensor with shape [batch_size, num_anchors].
    """
    abs_diff = tf.abs(predictions - targets)
    abs_diff_lt_1 = tf.less(abs_diff, 1.0)
    return weights * tf.reduce_sum(
        tf.where(abs_diff_lt_1, 0.5 * tf.square(abs_diff), abs_diff - 0.5), axis=2
    )


def classification_loss(predictions, targets):
    """
    Arguments:
        predictions: a float tensor with shape [batch_size, num_anchors, num_classes + 1],
            representing the predicted logits for each class.
        targets: an int tensor with shape [batch_size, num_anchors].
    Returns:
        a float tensor with shape [batch_size, num_anchors].
    """
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=targets, logits=predictions
    )
    return cross_entropy


def apply_hard_mining(
        location_losses, cls_losses,
        class_predictions_with_background,
        matches, decoded_boxes,
        loss_to_use='classification',
        loc_loss_weight=1.0, cls_loss_weight=1.0,
        num_hard_examples=3000, nms_threshold=0.99,
        max_negatives_per_positive=3, min_negatives_per_image=0):
    """Applies hard mining to anchorwise losses.

    Arguments:
        location_losses: a float tensor with shape [batch_size, num_anchors].
        cls_losses: a float tensor with shape [batch_size, num_anchors].
        class_predictions_with_background: a float tensor with shape [batch_size, num_anchors, num_classes + 1].
        matches: an int tensor with shape [batch_size, num_anchors].
        decoded_boxes: a float tensor with shape [batch_size, num_anchors, 4].
        loss_to_use: a string, only possible values are ['classification', 'both'].
        loc_loss_weight: a float number.
        cls_loss_weight: a float number.
        num_hard_examples: an integer.
        nms_threshold: a float number.
        max_negatives_per_positive: a float number.
        min_negatives_per_image: an integer.
    Returns:
        two float tensors with shape [].
    """

    # when training it is important that
    # batch size is known
    batch_size, num_anchors = matches.shape.as_list()
    assert batch_size is not None
    decoded_boxes.set_shape([batch_size, num_anchors, 4])
    location_losses.set_shape([batch_size, num_anchors])
    cls_losses.set_shape([batch_size, num_anchors])
    # all `set_shape` above are dirty tricks,
    # without them shape information is lost for some reason

    # all these tensors must have static first dimension (batch size)
    decoded_boxes_list = tf.unstack(decoded_boxes, axis=0)
    location_losses_list = tf.unstack(location_losses, axis=0)
    cls_losses_list = tf.unstack(cls_losses, axis=0)
    matches_list = tf.unstack(matches, axis=0)
    # they all lists with length = batch_size

    batch_size = len(decoded_boxes_list)
    num_positives_list, num_negatives_list = [], []
    mined_location_losses, mined_cls_losses = [], []

    # do OHEM for each image in the batch
    for i, box_locations in enumerate(decoded_boxes_list):
        image_losses = cls_losses_list[i] * cls_loss_weight
        if loss_to_use == 'both':
            image_losses += (location_losses_list[i] * loc_loss_weight)
        # it has shape [num_anchors]

        selected_indices = tf.image.non_max_suppression(
            box_locations, image_losses, num_hard_examples, nms_threshold
        )
        selected_indices, num_positives, num_negatives = _subsample_selection_to_desired_neg_pos_ratio(
             selected_indices, matches_list[i],
             max_negatives_per_positive, min_negatives_per_image
        )
        num_positives_list.append(num_positives)
        num_negatives_list.append(num_negatives)
        mined_location_losses.append(
            tf.reduce_sum(tf.gather(location_losses_list[i], selected_indices), axis=0)
        )
        mined_cls_losses.append(
            tf.reduce_sum(tf.gather(cls_losses_list[i], selected_indices), axis=0)
        )

    mean_num_positives = tf.reduce_mean(tf.stack(num_positives_list, axis=0), axis=0)
    mean_num_negatives = tf.reduce_mean(tf.stack(num_negatives_list, axis=0), axis=0)
    tf.summary.scalar('mean_num_positives', mean_num_positives)
    tf.summary.scalar('mean_num_negatives', mean_num_negatives)

    location_loss = tf.reduce_sum(tf.stack(mined_location_losses, axis=0), axis=0)
    cls_loss = tf.reduce_sum(tf.stack(mined_cls_losses, axis=0), axis=0)
    return location_loss, cls_loss


def _subsample_selection_to_desired_neg_pos_ratio(
        indices, match, max_negatives_per_positive, min_negatives_per_image):
    """Subsample a collection of selected indices to a desired neg:pos ratio.

    Arguments:
        indices: an int or long tensor with shape [M],
            it represents a collection of selected anchor indices.
        match: an int tensor with shape [num_anchors].
        max_negatives_per_positive: a float number, maximum number
            of negatives for each positive anchor.
        min_negatives_per_image: an integer, minimum number of negative anchors for a given
            image. Allows sampling negatives in image without any positive anchors.
    Returns:
        selected_indices: an int or long tensor with shape [M'] and with M' <= M.
            It represents a collection of selected anchor indices.
        num_positives: an int tensor with shape []. It represents the
            number of positive examples in selected set of indices.
        num_negatives: an int tensor with shape []. It represents the
            number of negative examples in selected set of indices.
    """
    positives_indicator = tf.gather(tf.greater_equal(match, 0), indices)
    negatives_indicator = tf.logical_not(positives_indicator)
    # they have shape [num_hard_examples]

    # all positives in `indices` will be kept
    num_positives = tf.reduce_sum(tf.to_int32(positives_indicator), axis=0)
    max_negatives = tf.maximum(
        min_negatives_per_image,
        tf.to_int32(max_negatives_per_positive * tf.to_float(num_positives))
    )

    top_k_negatives_indicator = tf.less_equal(
        tf.cumsum(tf.to_int32(negatives_indicator), axis=0),
        max_negatives
    )
    subsampled_selection_indices = tf.where(
        tf.logical_or(positives_indicator, top_k_negatives_indicator)
    )  # shape [num_hard_examples, 1]
    subsampled_selection_indices = tf.squeeze(subsampled_selection_indices, axis=1)
    selected_indices = tf.gather(indices, subsampled_selection_indices)

    num_negatives = tf.size(subsampled_selection_indices) - num_positives
    return selected_indices, num_positives, num_negatives
