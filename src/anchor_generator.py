import tensorflow as tf
import math
from src.utils.box_utils import to_minmax_coordinates


ANCHOR_SPECIFICATIONS = [
    [(32, 1.0, 4), (64, 1.0, 2), (128, 1.0, 1)],  # scale 0
    [(256, 1.0, 1)],  # scale 1
    [(512, 1.0, 1)],  # scale 2
]
# every tuple represents (box scale, aspect ratio, densification parameter)

"""
Notes:
1. Assume that size(image) = (image_width, image_height),
   then by definition
   image_aspect_ratio := image_width/image_height

2. All anchor boxes are in normalized coordinates (in [0, 1] range).
   So, for each box:
   (width * image_width)/(height * image_height) = aspect_ratio

3. Scale of an anchor box is defined like this:
   scale := sqrt(height * image_height * width * image_width)

4. Total number of anchor boxes depends on image size.

5. `scale` and `aspect_ratio` are independent of image size.
   `width` and `height` depend on image size.

6. If we change image size then normalized coordinates of
   the anchor boxes will change.
"""


class AnchorGenerator:
    def __init__(self):
        self.box_specs_list = ANCHOR_SPECIFICATIONS

    def __call__(self, image_features, image_size):
        """
        Arguments:
            image_features: a list of float tensors where the ith tensor
                has shape [batch, height_i, width_i, channels_i].
            image_size: a tuple of integers (int tensors with shape []) (width, height).
        Returns:
            a float tensor with shape [num_anchor, 4],
            boxes with normalized coordinates (and clipped to the unit square).
        """
        feature_map_shape_list = []
        for feature_map in image_features:

            height_i, width_i = feature_map.shape.as_list()[1:3]
            if height_i is None or width_i is None:
                height_i, width_i = tf.shape(feature_map)[1], tf.shape(feature_map)[2]

            feature_map_shape_list.append((height_i, width_i))
        image_width, image_height = image_size

        # number of anchors per cell in a grid
        self.num_anchors_per_location = [
            sum(n*n for _, _, n in layer_box_specs)
            for layer_box_specs in self.box_specs_list
        ]

        with tf.name_scope('anchor_generator'):
            anchor_grid_list, num_anchors_per_feature_map = [], []
            for grid_size, box_spec in zip(feature_map_shape_list, self.box_specs_list):

                h, w = grid_size
                stride = (1.0/tf.to_float(h), 1.0/tf.to_float(w))
                offset = (0.5/tf.to_float(h), 0.5/tf.to_float(w))

                local_anchors = []
                for scale, aspect_ratio, n in box_spec:
                    local_anchors.append(tile_anchors(
                        image_size=(image_width, image_height),
                        grid_height=h, grid_width=w, scale=scale,
                        aspect_ratio=aspect_ratio, anchor_stride=stride,
                        anchor_offset=offset, n=n
                    ))

                # reshaping in the right order is important
                local_anchors = tf.concat(local_anchors, axis=2)
                local_anchors = tf.reshape(local_anchors, [-1, 4])
                anchor_grid_list.append(local_anchors)

                num_anchors_per_feature_map.append(h * w * sum(n*n for _, _, n in box_spec))

        # constant tensors, anchors for each feature map
        self.anchor_grid_list = anchor_grid_list
        self.num_anchors_per_feature_map = num_anchors_per_feature_map

        with tf.name_scope('concatenate'):
            anchors = tf.concat(anchor_grid_list, axis=0)
            ymin, xmin, ymax, xmax = to_minmax_coordinates(tf.unstack(anchors, axis=1))
            anchors = tf.stack([ymin, xmin, ymax, xmax], axis=1)
            anchors = tf.clip_by_value(anchors, 0.0, 1.0)
            return anchors


def tile_anchors(
        image_size, grid_height, grid_width,
        scale, aspect_ratio, anchor_stride, anchor_offset, n):
    """
    Arguments:
        image_size: a tuple of integers (width, height).
        grid_height: an integer, size of the grid in the y direction.
        grid_width: an integer, size of the grid in the x direction.
        scale: a float number.
        aspect_ratio: a float number.
        anchor_stride: a tuple of float numbers, difference in centers between
            anchors for adjacent grid positions.
        anchor_offset: a tuple of float numbers,
            center of the anchor on upper left element of the grid ((0, 0)-th anchor).
        n: an integer, densification parameter.
    Returns:
        a float tensor with shape [grid_height, grid_width, n*n, 4].
    """
    ratio_sqrt = tf.sqrt(aspect_ratio)
    unnormalized_height = scale / ratio_sqrt
    unnormalized_width = scale * ratio_sqrt

    # to [0, 1] range
    image_width, image_height = image_size
    height = unnormalized_height/tf.to_float(image_height)
    width = unnormalized_width/tf.to_float(image_width)
    # (sometimes it could be outside the range, but we clip it)

    boxes = generate_anchors_at_upper_left_corner(height, width, anchor_offset, n)
    # shape [n*n, 4]

    y_translation = tf.to_float(tf.range(grid_height)) * anchor_stride[0]
    x_translation = tf.to_float(tf.range(grid_width)) * anchor_stride[1]
    x_translation, y_translation = tf.meshgrid(x_translation, y_translation)
    # they have shape [grid_height, grid_width]

    center_translations = tf.stack([y_translation, x_translation], axis=2)
    translations = tf.pad(center_translations, [[0, 0], [0, 0], [0, 2]])
    translations = tf.expand_dims(translations, 2)
    translations = tf.tile(translations, [1, 1, n*n, 1])
    # shape [grid_height, grid_width, n*n, 4]

    boxes = tf.reshape(boxes, [1, 1, n*n, 4])
    boxes = boxes + translations  # shape [grid_height, grid_width, n*n, 4]
    return boxes


def generate_anchors_at_upper_left_corner(height, width, anchor_offset, n):
    """Generate densified anchor boxes at (0, 0) grid position."""

    # a usual center, if n = 1 it will be returned
    cy, cx = anchor_offset[0], anchor_offset[1]

    # a usual left upper corner
    ymin, xmin = cy - 0.5*height, cx - 0.5*width

    # now i shift the usual center a little (densification)
    sy, sx = height/n, width/n

    center_ids = tf.to_float(tf.range(n))
    # shape [n]

    # shifted centers
    new_centers_y = ymin + 0.5*sy + sy*center_ids
    new_centers_x = xmin + 0.5*sx + sx*center_ids
    # they have shape [n]

    # now i must get all pairs of y, x coordinates
    new_centers_y = tf.expand_dims(new_centers_y, 0)  # shape [1, n]
    new_centers_x = tf.expand_dims(new_centers_x, 1)  # shape [n, 1]

    new_centers_y = tf.tile(new_centers_y, [n, 1])
    new_centers_x = tf.tile(new_centers_x, [1, n])
    # they have shape [n, n]

    centers = tf.stack([new_centers_y, new_centers_x], axis=2)
    # shape [n, n, 2]

    sizes = tf.stack([height, width], axis=0)  # shape [2]
    sizes = tf.expand_dims(sizes, 0)
    sizes = tf.expand_dims(sizes, 0)  # shape [1, 1, 2]
    sizes = tf.tile(sizes, [n, n, 1])

    boxes = tf.stack([centers, sizes], axis=2)
    boxes = tf.reshape(boxes, [-1, 4])
    return boxes
