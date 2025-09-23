"""
Mask R-CNN
Common utility functions and classes.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""

import sys
import os
import logging
import math
import random
import numpy as np
import tensorflow as tf
import scipy
import skimage.color
import skimage.io
import skimage.transform
import cv2
import urllib.request
import shutil
import zipfile
import warnings
from distutils.version import LooseVersion
from tqdm.notebook import tqdm
from PIL import Image

#confusion matrix 

from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.collections import QuadMesh
import seaborn as sn
import seaborn as sns
from sklearn.metrics import confusion_matrix
from pandas import DataFrame
from string import ascii_uppercase
import mrcnn.model as modellib
import pandas as pd
import StereoFractAnalyzer as SF
import tempfile

# URL from which to download the latest COCO trained weights
COCO_MODEL_URL = "https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5"

#URL from which to download zip of SAGE pretrained models
PRETRAIN_URL = "https://github.com/timday23/SAGE_TPD/releases/download/v1.0.0/pretrained_models.zip"


############################################################
#  Bounding Boxes
############################################################

def extract_bboxes(mask):
    """Compute bounding boxes from masks.
    mask: [height, width, num_instances]. Mask pixels are either 1 or 0.

    Returns: bbox array [num_instances, (y1, x1, y2, x2)].
    """
    boxes = np.zeros([mask.shape[-1], 4], dtype=np.int32)
    for i in range(mask.shape[-1]):
        m = mask[:, :, i]
        # Bounding box.
        horizontal_indicies = np.where(np.any(m, axis=0))[0]
        vertical_indicies = np.where(np.any(m, axis=1))[0]
        if horizontal_indicies.shape[0]:
            x1, x2 = horizontal_indicies[[0, -1]]
            y1, y2 = vertical_indicies[[0, -1]]
            # x2 and y2 should not be part of the box. Increment by 1.
            x2 += 1
            y2 += 1
        else:
            # No mask for this instance. Might happen due to
            # resizing or cropping. Set bbox to zeros
            x1, x2, y1, y2 = 0, 0, 0, 0
        boxes[i] = np.array([y1, x1, y2, x2])
    return boxes.astype(np.int32)


def compute_iou(box, boxes, box_area, boxes_area):
    """Calculates IoU of the given box with the array of the given boxes.
    box: 1D vector [y1, x1, y2, x2]
    boxes: [boxes_count, (y1, x1, y2, x2)]
    box_area: float. the area of 'box'
    boxes_area: array of length boxes_count.

    Note: the areas are passed in rather than calculated here for
    efficiency. Calculate once in the caller to avoid duplicate work.
    """
    # Calculate intersection areas
    y1 = np.maximum(box[0], boxes[:, 0])
    y2 = np.minimum(box[2], boxes[:, 2])
    x1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    union = box_area + boxes_area[:] - intersection[:]
    iou = intersection / union
    return iou


def compute_overlaps(boxes1, boxes2):
    """Computes IoU overlaps between two sets of boxes.
    boxes1, boxes2: [N, (y1, x1, y2, x2)].

    For better performance, pass the largest set first and the smaller second.
    """
    # Areas of anchors and GT boxes
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # Compute overlaps to generate matrix [boxes1 count, boxes2 count]
    # Each cell contains the IoU value.
    overlaps = np.zeros((boxes1.shape[0], boxes2.shape[0]))
    for i in range(overlaps.shape[1]):
        box2 = boxes2[i]
        overlaps[:, i] = compute_iou(box2, boxes1, area2[i], area1)
    return overlaps


def compute_overlaps_masks(masks1, masks2):
    """Computes IoU overlaps between two sets of masks.
    masks1, masks2: [Height, Width, instances]
    """
    
    # If either set of masks is empty return empty result
    if masks1.shape[-1] == 0 or masks2.shape[-1] == 0:
        return np.zeros((masks1.shape[-1], masks2.shape[-1]))
    # flatten masks and compute their areas
    masks1 = np.reshape(masks1 > .5, (-1, masks1.shape[-1])).astype(np.float32)
    masks2 = np.reshape(masks2 > .5, (-1, masks2.shape[-1])).astype(np.float32)
    area1 = np.sum(masks1, axis=0)
    area2 = np.sum(masks2, axis=0)

    # intersections and union
    intersections = np.dot(masks1.T, masks2)
    union = area1[:, None] + area2[None, :] - intersections
    overlaps = intersections / union

    return overlaps

def compute_DICE(masks1,masks2):
    if masks1.shape[-1] == 0 or masks2.shape[-1] == 0:
        return np.zeros((masks1.shape[-1], masks2.shape[-1]))
    # flatten masks and compute their areas
    masks1 = np.reshape(masks1 > .5, (-1, masks1.shape[-1])).astype(np.float32)
    masks2 = np.reshape(masks2 > .5, (-1, masks2.shape[-1])).astype(np.float32)
    area1 = np.sum(masks1, axis=0)
    area2 = np.sum(masks2, axis=0)

    # intersections and union
    intersections = np.dot(masks1.T, masks2)
    union = area1[:, None] + area2[None, :]
    
    dice = (2.*intersections)/ union
    
    return dice
    
    


def non_max_suppression(boxes, scores, threshold):
    """Performs non-maximum suppression and returns indices of kept boxes.
    boxes: [N, (y1, x1, y2, x2)]. Notice that (y2, x2) lays outside the box.
    scores: 1-D array of box scores.
    threshold: Float. IoU threshold to use for filtering.
    """
    assert boxes.shape[0] > 0
    if boxes.dtype.kind != "f":
        boxes = boxes.astype(np.float32)

    # Compute box areas
    y1 = boxes[:, 0]
    x1 = boxes[:, 1]
    y2 = boxes[:, 2]
    x2 = boxes[:, 3]
    area = (y2 - y1) * (x2 - x1)

    # Get indicies of boxes sorted by scores (highest first)
    ixs = scores.argsort()[::-1]

    pick = []
    while len(ixs) > 0:
        # Pick top box and add its index to the list
        i = ixs[0]
        pick.append(i)
        # Compute IoU of the picked box with the rest
        iou = compute_iou(boxes[i], boxes[ixs[1:]], area[i], area[ixs[1:]])
        # Identify boxes with IoU over the threshold. This
        # returns indices into ixs[1:], so add 1 to get
        # indices into ixs.
        remove_ixs = np.where(iou > threshold)[0] + 1
        # Remove indices of the picked and overlapped boxes.
        ixs = np.delete(ixs, remove_ixs)
        ixs = np.delete(ixs, 0)
    return np.array(pick, dtype=np.int32)


def apply_box_deltas(boxes, deltas):
    """Applies the given deltas to the given boxes.
    boxes: [N, (y1, x1, y2, x2)]. Note that (y2, x2) is outside the box.
    deltas: [N, (dy, dx, log(dh), log(dw))]
    """
    boxes = boxes.astype(np.float32)
    # Convert to y, x, h, w
    height = boxes[:, 2] - boxes[:, 0]
    width = boxes[:, 3] - boxes[:, 1]
    center_y = boxes[:, 0] + 0.5 * height
    center_x = boxes[:, 1] + 0.5 * width
    # Apply deltas
    center_y += deltas[:, 0] * height
    center_x += deltas[:, 1] * width
    height *= np.exp(deltas[:, 2])
    width *= np.exp(deltas[:, 3])
    # Convert back to y1, x1, y2, x2
    y1 = center_y - 0.5 * height
    x1 = center_x - 0.5 * width
    y2 = y1 + height
    x2 = x1 + width
    return np.stack([y1, x1, y2, x2], axis=1)


def box_refinement_graph(box, gt_box):
    """Compute refinement needed to transform box to gt_box.
    box and gt_box are [N, (y1, x1, y2, x2)]
    """
    box = tf.cast(box, tf.float32)
    gt_box = tf.cast(gt_box, tf.float32)

    height = box[:, 2] - box[:, 0]
    width = box[:, 3] - box[:, 1]
    center_y = box[:, 0] + 0.5 * height
    center_x = box[:, 1] + 0.5 * width

    gt_height = gt_box[:, 2] - gt_box[:, 0]
    gt_width = gt_box[:, 3] - gt_box[:, 1]
    gt_center_y = gt_box[:, 0] + 0.5 * gt_height
    gt_center_x = gt_box[:, 1] + 0.5 * gt_width

    dy = (gt_center_y - center_y) / height
    dx = (gt_center_x - center_x) / width
    dh = tf.math.log(gt_height / height)
    dw = tf.math.log(gt_width / width)

    result = tf.stack([dy, dx, dh, dw], axis=1)
    return result


def box_refinement(box, gt_box):
    """Compute refinement needed to transform box to gt_box.
    box and gt_box are [N, (y1, x1, y2, x2)]. (y2, x2) is
    assumed to be outside the box.
    """
    box = box.astype(np.float32)
    gt_box = gt_box.astype(np.float32)

    height = box[:, 2] - box[:, 0]
    width = box[:, 3] - box[:, 1]
    center_y = box[:, 0] + 0.5 * height
    center_x = box[:, 1] + 0.5 * width

    gt_height = gt_box[:, 2] - gt_box[:, 0]
    gt_width = gt_box[:, 3] - gt_box[:, 1]
    gt_center_y = gt_box[:, 0] + 0.5 * gt_height
    gt_center_x = gt_box[:, 1] + 0.5 * gt_width

    dy = (gt_center_y - center_y) / height
    dx = (gt_center_x - center_x) / width
    dh = np.log(gt_height / height)
    dw = np.log(gt_width / width)

    return np.stack([dy, dx, dh, dw], axis=1)


############################################################
#  Dataset
############################################################

class Dataset(object):
    """The base class for dataset classes.
    To use it, create a new class that adds functions specific to the dataset
    you want to use. For example:

    class CatsAndDogsDataset(Dataset):
        def load_cats_and_dogs(self):
            ...
        def load_mask(self, image_id):
            ...
        def image_reference(self, image_id):
            ...

    See COCODataset and ShapesDataset as examples.
    """

    def __init__(self, class_map=None):
        self._image_ids = []
        self.image_info = []
        # Background is always the first class
        self.class_info = [{"source": "", "id": 0, "name": "BG"}]
        self.source_class_ids = {}

    def add_class(self, source, class_id, class_name):
        assert "." not in source, "Source name cannot contain a dot"
        # Does the class exist already?
        for info in self.class_info:
            if info['source'] == source and info["id"] == class_id:
                # source.class_id combination already available, skip
                return
        # Add the class
        self.class_info.append({
            "source": source,
            "id": class_id,
            "name": class_name,
        })

    def add_image(self, source, image_id, path,basename, **kwargs):
        image_info = {
            "id": image_id,
            "source": source,
            "path": path,
            "basename": basename, #add basename to be able to retrieve original file name 
        }
        image_info.update(kwargs)
        self.image_info.append(image_info)

    def image_reference(self, image_id):
        """Return a link to the image in its source Website or details about
        the image that help looking it up or debugging it.

        Override for your dataset, but pass to this function
        if you encounter images not in your dataset.
        """
        return ""

    def prepare(self, class_map=None):
        """Prepares the Dataset class for use.

        TODO: class map is not supported yet. When done, it should handle mapping
              classes from different datasets to the same class ID.
        """

        def clean_name(name):
            """Returns a shorter version of object names for cleaner display."""
            return ",".join(name.split(",")[:1])

        # Build (or rebuild) everything else from the info dicts.
        self.num_classes = len(self.class_info)
        self.class_ids = np.arange(self.num_classes)
        self.class_names = [clean_name(c["name"]) for c in self.class_info]
        self.num_images = len(self.image_info)
        self._image_ids = np.arange(self.num_images)

        # Mapping from source class and image IDs to internal IDs
        self.class_from_source_map = {"{}.{}".format(info['source'], info['id']): id
                                      for info, id in zip(self.class_info, self.class_ids)}
        self.image_from_source_map = {"{}.{}".format(info['source'], info['id']): id
                                      for info, id in zip(self.image_info, self.image_ids)}

        # Map sources to class_ids they support
        self.sources = list(set([i['source'] for i in self.class_info]))
        self.source_class_ids = {}
        # Loop over datasets
        for source in self.sources:
            self.source_class_ids[source] = []
            # Find classes that belong to this dataset
            for i, info in enumerate(self.class_info):
                # Include BG class in all datasets
                if i == 0 or source == info['source']:
                    self.source_class_ids[source].append(i)

    def map_source_class_id(self, source_class_id):
        """Takes a source class ID and returns the int class ID assigned to it.

        For example:
        dataset.map_source_class_id("coco.12") -> 23
        """
        return self.class_from_source_map[source_class_id]

    def get_source_class_id(self, class_id, source):
        """Map an internal class ID to the corresponding class ID in the source dataset."""
        info = self.class_info[class_id]
        assert info['source'] == source
        return info['id']

    @property
    def image_ids(self):
        return self._image_ids

    def source_image_link(self, image_id):
        """Returns the path or URL to the image.
        Override this to return a URL to the image if it's available online for easy
        debugging.
        """
        return self.image_info[image_id]["path"]

    def load_image(self, image_id):
        """Load the specified image and return a [H,W,3] Numpy array.
        """
        # Load image
        image = skimage.io.imread(self.image_info[image_id]['path'])
        # If grayscale. Convert to RGB for consistency.
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        # If has an alpha channel, remove it for consistency
        if image.shape[-1] == 4:
            image = image[..., :3]
        return image

    def load_mask(self, image_id):
        """Load instance masks for the given image.

        Different datasets use different ways to store masks. Override this
        method to load instance masks and return them in the form of am
        array of binary masks of shape [height, width, instances].

        Returns:
            masks: A bool array of shape [height, width, instance count] with
                a binary mask per instance.
            class_ids: a 1D array of class IDs of the instance masks.
        """
        # Override this function to load a mask from your dataset.
        # Otherwise, it returns an empty mask.
        logging.warning("You are using the default load_mask(), maybe you need to define your own one.")
        mask = np.empty([0, 0, 0])
        class_ids = np.empty([0], np.int32)
        return mask, class_ids


def resize_image(image, min_dim=None, max_dim=None, min_scale=None, mode="square"):
    """Resizes an image keeping the aspect ratio unchanged.

    min_dim: if provided, resizes the image such that it's smaller
        dimension == min_dim
    max_dim: if provided, ensures that the image longest side doesn't
        exceed this value.
    min_scale: if provided, ensure that the image is scaled up by at least
        this percent even if min_dim doesn't require it.
    mode: Resizing mode.
        none: No resizing. Return the image unchanged.
        square: Resize and pad with zeros to get a square image
            of size [max_dim, max_dim].
        pad64: Pads width and height with zeros to make them multiples of 64.
               If min_dim or min_scale are provided, it scales the image up
               before padding. max_dim is ignored in this mode.
               The multiple of 64 is needed to ensure smooth scaling of feature
               maps up and down the 6 levels of the FPN pyramid (2**6=64).
        crop: Picks random crops from the image. First, scales the image based
              on min_dim and min_scale, then picks a random crop of
              size min_dim x min_dim. Can be used in training only.
              max_dim is not used in this mode.

    Returns:
    image: the resized image
    window: (y1, x1, y2, x2). If max_dim is provided, padding might
        be inserted in the returned image. If so, this window is the
        coordinates of the image part of the full image (excluding
        the padding). The x2, y2 pixels are not included.
    scale: The scale factor used to resize the image
    padding: Padding added to the image [(top, bottom), (left, right), (0, 0)]
    """
    # Keep track of image dtype and return results in the same dtype
    image_dtype = image.dtype
    # print(image_dtype)
    # Default window (y1, x1, y2, x2) and default scale == 1.
    h, w = image.shape[:2]
    window = (0, 0, h, w)
    scale = 1
    padding = [(0, 0), (0, 0), (0, 0)]
    crop = None

    if mode == "none":
        return image, window, scale, padding, crop

    # Scale?
    if min_dim:
        # Scale up but not down
        scale = max(1, min_dim / min(h, w))
    if min_scale and scale < min_scale:
        scale = min_scale

    # Does it exceed max dim?
    if max_dim and mode == "square":
        image_max = max(h, w)
        if round(image_max * scale) > max_dim:
            scale = max_dim / image_max

    # Resize image using bilinear interpolation
    if scale != 1:
        image = resize(image, (round(h * scale), round(w * scale)),
                       preserve_range=True)

    # Need padding or cropping?
    if mode == "square":
        # Get new height and width
        h, w = image.shape[:2]
        top_pad = (max_dim - h) // 2
        bottom_pad = max_dim - h - top_pad
        left_pad = (max_dim - w) // 2
        right_pad = max_dim - w - left_pad
        padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
        image = np.pad(image, padding, mode='constant', constant_values=0)
        window = (top_pad, left_pad, h + top_pad, w + left_pad)
    elif mode == "pad64":
        h, w = image.shape[:2]
        # Both sides must be divisible by 64
        assert min_dim % 64 == 0, "Minimum dimension must be a multiple of 64"
        # Height
        if h % 64 > 0:
            max_h = h - (h % 64) + 64
            top_pad = (max_h - h) // 2
            bottom_pad = max_h - h - top_pad
        else:
            top_pad = bottom_pad = 0
        # Width
        if w % 64 > 0:
            max_w = w - (w % 64) + 64
            left_pad = (max_w - w) // 2
            right_pad = max_w - w - left_pad
        else:
            left_pad = right_pad = 0
        padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
        image = np.pad(image, padding, mode='constant', constant_values=0)
        window = (top_pad, left_pad, h + top_pad, w + left_pad)
    elif mode == "crop":
        # Pick a random crop
        h, w = image.shape[:2]
        y = random.randint(0, (h - min_dim))
        x = random.randint(0, (w - min_dim))
        crop = (y, x, min_dim, min_dim)
        image = image[y:y + min_dim, x:x + min_dim]
        window = (0, 0, min_dim, min_dim)
    else:
        raise Exception("Mode {} not supported".format(mode))
    return image.astype(image_dtype), window, scale, padding, crop


def resize_mask(mask, scale, padding, crop=None):
    """Resizes a mask using the given scale and padding.
    Typically, you get the scale and padding from resize_image() to
    ensure both, the image and the mask, are resized consistently.

    scale: mask scaling factor
    padding: Padding to add to the mask in the form
            [(top, bottom), (left, right), (0, 0)]
    """
    # Suppress warning from scipy 0.13.0, the output shape of zoom() is
    # calculated with round() instead of int()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mask = scipy.ndimage.zoom(mask, zoom=[scale, scale, 1], order=0)
    if crop is not None:
        y, x, h, w = crop
        mask = mask[y:y + h, x:x + w]
    else:
        mask = np.pad(mask, padding, mode='constant', constant_values=0)
    return mask


def minimize_mask(bbox, mask, mini_shape):
    """Resize masks to a smaller version to reduce memory load.
    Mini-masks can be resized back to image scale using expand_masks()

    See inspect_data.ipynb notebook for more details.
    """
    mini_mask = np.zeros(mini_shape + (mask.shape[-1],), dtype=bool)
    for i in range(mask.shape[-1]):
        # Pick slice and cast to bool in case load_mask() returned wrong dtype
        m = mask[:, :, i].astype(bool)
        y1, x1, y2, x2 = bbox[i][:4]
        m = m[y1:y2, x1:x2]
        if m.size == 0:
            raise Exception("Invalid bounding box with area of zero")
        # Resize with bilinear interpolation
        m = resize(m, mini_shape)
        mini_mask[:, :, i] = np.around(m).astype(np.bool)
    return mini_mask


def expand_mask(bbox, mini_mask, image_shape):
    """Resizes mini masks back to image size. Reverses the change
    of minimize_mask().

    See inspect_data.ipynb notebook for more details.
    """
    mask = np.zeros(image_shape[:2] + (mini_mask.shape[-1],), dtype=bool)
    for i in range(mask.shape[-1]):
        m = mini_mask[:, :, i]
        y1, x1, y2, x2 = bbox[i][:4]
        h = y2 - y1
        w = x2 - x1
        # Resize with bilinear interpolation
        m = resize(m, (h, w))
        mask[y1:y2, x1:x2, i] = np.around(m).astype(np.bool)
    return mask


# TODO: Build and use this function to reduce code duplication
def mold_mask(mask, config):
    pass


def unmold_mask(mask, bbox, image_shape):
    """Converts a mask generated by the neural network to a format similar
    to its original shape.
    mask: [height, width] of type float. A small, typically 28x28 mask.
    bbox: [y1, x1, y2, x2]. The box to fit the mask in.

    Returns a binary mask with the same size as the original image.
    """
    threshold = 0.5
    y1, x1, y2, x2 = bbox
    mask = resize(mask, (y2 - y1, x2 - x1))
    mask = np.where(mask >= threshold, 1, 0).astype(np.bool)

    # Put the mask in the right location.
    full_mask = np.zeros(image_shape[:2], dtype=np.bool)
    full_mask[y1:y2, x1:x2] = mask
    return full_mask


############################################################
#  Anchors
############################################################

def generate_anchors(scales, ratios, shape, feature_stride, anchor_stride):
    """
    scales: 1D array of anchor sizes in pixels. Example: [32, 64, 128]
    ratios: 1D array of anchor ratios of width/height. Example: [0.5, 1, 2]
    shape: [height, width] spatial shape of the feature map over which
            to generate anchors.
    feature_stride: Stride of the feature map relative to the image in pixels.
    anchor_stride: Stride of anchors on the feature map. For example, if the
        value is 2 then generate anchors for every other feature map pixel.
    """
    # Get all combinations of scales and ratios
    scales, ratios = np.meshgrid(np.array(scales), np.array(ratios))
    scales = scales.flatten()
    ratios = ratios.flatten()

    # Enumerate heights and widths from scales and ratios
    heights = scales / np.sqrt(ratios)
    widths = scales * np.sqrt(ratios)

    # Enumerate shifts in feature space
    shifts_y = np.arange(0, shape[0], anchor_stride) * feature_stride
    shifts_x = np.arange(0, shape[1], anchor_stride) * feature_stride
    shifts_x, shifts_y = np.meshgrid(shifts_x, shifts_y)

    # Enumerate combinations of shifts, widths, and heights
    box_widths, box_centers_x = np.meshgrid(widths, shifts_x)
    box_heights, box_centers_y = np.meshgrid(heights, shifts_y)

    # Reshape to get a list of (y, x) and a list of (h, w)
    box_centers = np.stack(
        [box_centers_y, box_centers_x], axis=2).reshape([-1, 2])
    box_sizes = np.stack([box_heights, box_widths], axis=2).reshape([-1, 2])

    # Convert to corner coordinates (y1, x1, y2, x2)
    boxes = np.concatenate([box_centers - 0.5 * box_sizes,
                            box_centers + 0.5 * box_sizes], axis=1)
    return boxes


def generate_pyramid_anchors(scales, ratios, feature_shapes, feature_strides,
                             anchor_stride):
    """Generate anchors at different levels of a feature pyramid. Each scale
    is associated with a level of the pyramid, but each ratio is used in
    all levels of the pyramid.

    Returns:
    anchors: [N, (y1, x1, y2, x2)]. All generated anchors in one array. Sorted
        with the same order of the given scales. So, anchors of scale[0] come
        first, then anchors of scale[1], and so on.
    """
    # Anchors
    # [anchor_count, (y1, x1, y2, x2)]
    anchors = []
    for i in range(len(scales)):
        anchors.append(generate_anchors(scales[i], ratios, feature_shapes[i],
                                        feature_strides[i], anchor_stride))
    return np.concatenate(anchors, axis=0)


############################################################
#  Miscellaneous
############################################################

def trim_zeros(x):
    """It's common to have tensors larger than the available data and
    pad with zeros. This function removes rows that are all zeros.

    x: [rows, columns].
    """
    assert len(x.shape) == 2
    return x[~np.all(x == 0, axis=1)]


def compute_matches(gt_boxes, gt_class_ids, gt_masks,
                    pred_boxes, pred_class_ids, pred_scores, pred_masks,
                    iou_threshold=0.5, score_threshold=0.0):
    """Finds matches between prediction and ground truth instances.

    Returns:
        gt_match: 1-D array. For each GT box it has the index of the matched
                  predicted box.
        pred_match: 1-D array. For each predicted box, it has the index of
                    the matched ground truth box.
        overlaps: [pred_boxes, gt_boxes] IoU overlaps.
    """
    # Trim zero padding
    # TODO: cleaner to do zero unpadding upstream
    gt_boxes = trim_zeros(gt_boxes)
    gt_masks = gt_masks[..., :gt_boxes.shape[0]]
    pred_boxes = trim_zeros(pred_boxes)
    pred_scores = pred_scores[:pred_boxes.shape[0]]
    # Sort predictions by score from high to low
    indices = np.argsort(pred_scores)[::-1]
    pred_boxes = pred_boxes[indices]
    pred_class_ids = pred_class_ids[indices]
    pred_scores = pred_scores[indices]
    pred_masks = pred_masks[..., indices]

    # Compute IoU overlaps [pred_masks, gt_masks]
    overlaps = compute_overlaps_masks(pred_masks, gt_masks)

    # Loop through predictions and find matching ground truth boxes
    match_count = 0
    pred_match = -1 * np.ones([pred_boxes.shape[0]])
    gt_match = -1 * np.ones([gt_boxes.shape[0]])
    for i in range(len(pred_boxes)):
        # Find best matching ground truth box
        # 1. Sort matches by score
        sorted_ixs = np.argsort(overlaps[i])[::-1]
        # 2. Remove low scores
        low_score_idx = np.where(overlaps[i, sorted_ixs] < score_threshold)[0]
        if low_score_idx.size > 0:
            sorted_ixs = sorted_ixs[:low_score_idx[0]]
        # 3. Find the match
        for j in sorted_ixs:
            # If ground truth box is already matched, go to next one
            if gt_match[j] > -1:
                continue
            # If we reach IoU smaller than the threshold, end the loop
            iou = overlaps[i, j]
            if iou < iou_threshold:
                break
            # Do we have a match?
            if pred_class_ids[i] == gt_class_ids[j]:
                match_count += 1
                gt_match[j] = i
                pred_match[i] = j
                break

    return gt_match, pred_match, overlaps


def compute_ap(gt_boxes, gt_class_ids, gt_masks,
               pred_boxes, pred_class_ids, pred_scores, pred_masks,
               iou_threshold=0.5):
    """Compute Average Precision at a set IoU threshold (default 0.5).

    Returns:
    mAP: Mean Average Precision
    precisions: List of precisions at different class score thresholds.
    recalls: List of recall values at different class score thresholds.
    overlaps: [pred_boxes, gt_boxes] IoU overlaps.
    """
    # Get matches and overlaps
    gt_match, pred_match, overlaps = compute_matches(
        gt_boxes, gt_class_ids, gt_masks,
        pred_boxes, pred_class_ids, pred_scores, pred_masks,
        iou_threshold)

    # Compute precision and recall at each prediction box step
    precisions = np.cumsum(pred_match > -1) / (np.arange(len(pred_match)) + 1)
    recalls = np.cumsum(pred_match > -1).astype(np.float32) / len(gt_match)

    # Pad with start and end values to simplify the math
    precisions = np.concatenate([[0], precisions, [0]])
    recalls = np.concatenate([[0], recalls, [1]])

    # Ensure precision values decrease but don't increase. This way, the
    # precision value at each recall threshold is the maximum it can be
    # for all following recall thresholds, as specified by the VOC paper.
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = np.maximum(precisions[i], precisions[i + 1])

    # Compute mean AP over recall range
    indices = np.where(recalls[:-1] != recalls[1:])[0] + 1
    mAP = np.sum((recalls[indices] - recalls[indices - 1]) *
                 precisions[indices])

    return mAP, precisions, recalls, overlaps


def compute_ap_range(gt_box, gt_class_id, gt_mask,
                     pred_box, pred_class_id, pred_score, pred_mask,
                     iou_thresholds=None, verbose=1):
    """Compute AP over a range or IoU thresholds. Default range is 0.5-0.95."""
    # Default is 0.5 to 0.95 with increments of 0.05
    iou_thresholds = iou_thresholds or np.arange(0.5, 1.0, 0.05)
    
    # Compute AP over range of IoU thresholds
    AP = []
    for iou_threshold in iou_thresholds:
        ap, precisions, recalls, overlaps =\
            compute_ap(gt_box, gt_class_id, gt_mask,
                        pred_box, pred_class_id, pred_score, pred_mask,
                        iou_threshold=iou_threshold)
        if verbose:
            print("AP @{:.2f}:\t {:.3f}".format(iou_threshold, ap))
        AP.append(ap)
    AP = np.array(AP).mean()
    if verbose:
        print("AP @{:.2f}-{:.2f}:\t {:.3f}".format(
            iou_thresholds[0], iou_thresholds[-1], AP))
    return AP


def compute_recall(pred_boxes, gt_boxes, iou):
    """Compute the recall at the given IoU threshold. It's an indication
    of how many GT boxes were found by the given prediction boxes.

    pred_boxes: [N, (y1, x1, y2, x2)] in image coordinates
    gt_boxes: [N, (y1, x1, y2, x2)] in image coordinates
    """
    # Measure overlaps
    overlaps = compute_overlaps(pred_boxes, gt_boxes)
    iou_max = np.max(overlaps, axis=1)
    iou_argmax = np.argmax(overlaps, axis=1)
    positive_ids = np.where(iou_max >= iou)[0]
    matched_gt_boxes = iou_argmax[positive_ids]

    recall = len(set(matched_gt_boxes)) / gt_boxes.shape[0]
    return recall, positive_ids


# ## Batch Slicing
# Some custom layers support a batch size of 1 only, and require a lot of work
# to support batches greater than 1. This function slices an input tensor
# across the batch dimension and feeds batches of size 1. Effectively,
# an easy way to support batches > 1 quickly with little code modification.
# In the long run, it's more efficient to modify the code to support large
# batches and getting rid of this function. Consider this a temporary solution
def batch_slice(inputs, graph_fn, batch_size, names=None):
    """Splits inputs into slices and feeds each slice to a copy of the given
    computation graph and then combines the results. It allows you to run a
    graph on a batch of inputs even if the graph is written to support one
    instance only.

    inputs: list of tensors. All must have the same first dimension length
    graph_fn: A function that returns a TF tensor that's part of a graph.
    batch_size: number of slices to divide the data into.
    names: If provided, assigns names to the resulting tensors.
    """
    if not isinstance(inputs, list):
        inputs = [inputs]

    outputs = []
    for i in range(batch_size):
        inputs_slice = [x[i] for x in inputs]
        output_slice = graph_fn(*inputs_slice)
        if not isinstance(output_slice, (tuple, list)):
            output_slice = [output_slice]
        outputs.append(output_slice)
    # Change outputs from a list of slices where each is
    # a list of outputs to a list of outputs and each has
    # a list of slices
    outputs = list(zip(*outputs))

    if names is None:
        names = [None] * len(outputs)

    result = [tf.stack(o, axis=0, name=n)
              for o, n in zip(outputs, names)]
    if len(result) == 1:
        result = result[0]

    return result


def download_trained_weights(coco_model_path, verbose=1):
    """Download COCO trained weights from Releases.

    coco_model_path: local path of COCO trained weights
    """
    if verbose > 0:
        print("Downloading pretrained model to " + coco_model_path + " ...")
    with urllib.request.urlopen(COCO_MODEL_URL) as resp, open(coco_model_path, 'wb') as out:
        shutil.copyfileobj(resp, out)
    if verbose > 0:
        print("... done downloading pretrained model!")


def download_pretrained_models(pretrained_dir_path, verbose=1):
    """Download SAGE trained weights from Releases.

    pretrained_dir_path: local path of SAGE pretrained weights
    """
    os.makedirs(pretrained_dir_path, exist_ok=True)

    zip_path = os.path.join(pretrained_dir_path, "pretrained_models.zip")
    
    if verbose > 0:
        print("Downloading pretrained models to " + pretrained_dir_path + " ...")

    with urllib.request.urlopen(PRETRAIN_URL) as resp, open(zip_path, 'wb') as out:
        shutil.copyfileobj(resp, out)

    if verbose > 0:
        print("Download complete! Extracting...")

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(pretrained_dir_path)

    #remove zip file after extraction
    os.remove(zip_path)
    
    if verbose > 0:
        print("... done downloading pretrained models!")

def norm_boxes(boxes, shape):
    """Converts boxes from pixel coordinates to normalized coordinates.
    boxes: [N, (y1, x1, y2, x2)] in pixel coordinates
    shape: [..., (height, width)] in pixels

    Note: In pixel coordinates (y2, x2) is outside the box. But in normalized
    coordinates it's inside the box.

    Returns:
        [N, (y1, x1, y2, x2)] in normalized coordinates
    """
    h, w = shape
    scale = np.array([h - 1, w - 1, h - 1, w - 1])
    shift = np.array([0, 0, 1, 1])
    return np.divide((boxes - shift), scale).astype(np.float32)


def denorm_boxes(boxes, shape):
    """Converts boxes from normalized coordinates to pixel coordinates.
    boxes: [N, (y1, x1, y2, x2)] in normalized coordinates
    shape: [..., (height, width)] in pixels

    Note: In pixel coordinates (y2, x2) is outside the box. But in normalized
    coordinates it's inside the box.

    Returns:
        [N, (y1, x1, y2, x2)] in pixel coordinates
    """
    h, w = shape
    scale = np.array([h - 1, w - 1, h - 1, w - 1])
    shift = np.array([0, 0, 1, 1])
    return np.around(np.multiply(boxes, scale) + shift).astype(np.int32)


def resize(image, output_shape, order=1, mode='constant', cval=0, clip=True,
           preserve_range=False, anti_aliasing=False, anti_aliasing_sigma=None):
    """A wrapper for Scikit-Image resize().

    Scikit-Image generates warnings on every call to resize() if it doesn't
    receive the right parameters. The right parameters depend on the version
    of skimage. This solves the problem by using different parameters per
    version. And it provides a central place to control resizing defaults.
    """
    if LooseVersion(skimage.__version__) >= LooseVersion("0.14"):
        # New in 0.14: anti_aliasing. Default it to False for backward
        # compatibility with skimage 0.13.
        return skimage.transform.resize(
            image, output_shape,
            order=order, mode=mode, cval=cval, clip=clip,
            preserve_range=preserve_range, anti_aliasing=anti_aliasing,
            anti_aliasing_sigma=anti_aliasing_sigma)
    else:
        return skimage.transform.resize(
            image, output_shape,
            order=order, mode=mode, cval=cval, clip=clip,
            preserve_range=preserve_range)

    


###### Added utils for SAGE ########

class SAGEDataset(Dataset):
    """Load Dataset
    """
    def __init__(self,images_dir, particle_masks_dir, cluster_masks_dir, load_particle=True, load_cluster=True):
        super().__init__()
        self.images_dir = images_dir
        self.particle_masks_dir = particle_masks_dir
        self.cluster_masks_dir = cluster_masks_dir
        self.load_particle = load_particle  # Correctly initialize the attribute
        self.load_cluster = load_cluster  # Correctly initialize the attribute
   
        if load_particle and load_cluster:
            self.class_names = ["particle", "cluster"]
            self.add_class("SAGE",1,"particle") #add particle class
            self.add_class("SAGE",2,"cluster") #add cluster class 
        elif load_particle:
            self.class_names=["particle"]
            self.add_class("SAGE",1,"particle") #add particle class
        elif load_cluster:
            self.class_names=["cluster"]
            self.add_class("SAGE",1,"cluster") #add cluster class 
            
        #print(self.class_names)
    
    def load_dataset(self, dataset_name=None, mask_position=2):
        """Load images and masks from specified directories."""
        #load images
        image_filenames = [f for f in os.listdir(self.images_dir) if f.endswith('.png')]
        #print(f"unsorted: {image_filenames}")
        #sort them by number
        image_filenames.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
        #print(f"Sorted: {image_filenames}")
        
        for image_id, filename in enumerate(image_filenames):
            #tqdm(image_filenames, desc="Adding images", dynamic_ncols=True,position=1)):   removing because it already loads fast
            image_path = os.path.join(self.images_dir, filename)
            image_no = filename.split('_')[1] #get image number
            basename = os.path.splitext(filename)[0] 
            #print(basename)
            self.add_image("SAGE", image_id=image_id, path=image_path,basename=basename,width=None, height=None)
            
        #load masks for each image
        desc = f"Loading masks for {dataset_name}" if dataset_name else "Loading masks"
        for image_id in tqdm(range(len(self.image_info)), desc=desc, dynamic_ncols=True, position=mask_position, leave=False):
            #print(f"Loading Masks for Image {image_id}", end="\r")
            #sys.stdout.flush()
            self.load_mask(image_id)

        

    def load_image(self, image_id):
        """ Load an image from the dataset."""
        info = self.image_info[image_id]
        image = cv2.imread(info['path'])
        return image

    def image_reference(self, image_id):
        """Return the particle data of the image."""
        info = self.image_info[image_id]
        if info["source"] == "SAGE":
            return info["path"]
        else:
            return super(self.__class__).image_reference(self, image_id)

    def load_mask(self, image_id):
        """load instance masks for the particle of the given image ID."""
        
        info = self.image_info[image_id]
        masks = []
        class_ids = []
        
        if self.load_particle and self.load_cluster:
            masks_particle, class_ids_particle = self._load_class_masks(info, self.particle_masks_dir, 
                                                                        class_id=1,pattern='particle')
            #if masks_particle:
                #print(f"Loaded {len(masks_particle)} particle masks for Image ID {image_id}.")
            masks.extend(masks_particle)
            class_ids.extend(class_ids_particle)
            
            masks_cluster, class_ids_cluster = self._load_class_masks(info, self.cluster_masks_dir, 
                                                                      class_id=2,pattern='cluster')
            #if masks_cluster:
               # print(f"Loaded {len(masks_cluster)} cluster masks for Image ID {image_id}.")
            masks.extend(masks_cluster)
            class_ids.extend(class_ids_cluster)
            #print("Both particle and cluster masks")
            
        #particle masks
        elif self.load_particle:
            masks_particle, class_ids_particle = self._load_class_masks(info, self.particle_masks_dir, 
                                                                        class_id=1,pattern='particle')
            #if masks_particle:
                #print(f"Loaded {len(masks_particle)} particle masks for Image ID {image_id}.")
            masks.extend(masks_particle)
            class_ids.extend(class_ids_particle)
            #rint("only particle masks")
            
        #cluster masks
        elif self.load_cluster:
            masks_cluster, class_ids_cluster = self._load_class_masks(info, self.cluster_masks_dir, 
                                                                      class_id='1',pattern='cluster')
            #if masks_cluster:
               # print(f"Loaded {len(masks_cluster)} cluster masks for Image ID {image_id}.")
            masks.extend(masks_cluster)
            class_ids.extend(class_ids_cluster)
           #print(("only cluster masks"))
            
        #combine masks into 3d array
        if masks:
            combined_mask = np.stack(masks, axis =-1)
            return combined_mask, np.array(class_ids, dtype=np.int32)
                      
        #print(f" No masks found for image ID {image_id}.")
        return np.zeros((0,0), dtype=np.bool_),np.zeros((0,),dtype=np.int32)

    
    def _load_class_masks(self,info, masks_dir, class_id, pattern):
        """Load msks for a specific class based on a pattern"""
        
        masks = []
        class_ids = []
        
        #construct mask filename based on image filename 
        _, image_filename = os.path.split(info['path']) 
        image_no = image_filename.split('_')[1].replace('.png','') #extract the base name without the extension to form mask filename
        #print(image_no)
        #print(f"Loading masks for image number:{image_no}")
        
        #load all masks for the current image
       
        
        if pattern =='particle':
            i = 0
            first_mask_found = False
            while True: 
                mask_filename = f"mask_{image_no}_{i:06d}.png"
                mask_path = os.path.join(masks_dir, mask_filename)
                
                if os.path.exists(mask_path):
                    #print(f"Found mask file: {mask_path}")
                    first_mask_found=True
                    mask = cv2.imread(mask_path,cv2.IMREAD_GRAYSCALE) #load mask
                    if mask is not None:
                        masks.append(mask.astype(np.bool_))
                        class_ids.append(class_id)
                    i += 1
                elif not first_mask_found:
                    i=1
                    continue
                else:
                    #print(f"Mask file not found: {mask_path}")
                    break
        elif pattern == 'cluster':
            # For clusters, load only one mask
            mask_filename = f"mask_{image_no}.png"
            mask_path = os.path.join(masks_dir, mask_filename)
            #print(f"Checking path for mask: {mask_path}")

            if os.path.exists(mask_path):
                #print(f"Found mask file: {mask_path}")
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) # Load mask
                if mask is not None:
                    masks.append(mask.astype(np.bool_))
                    class_ids.append(class_id)
            else:
                print(f"Mask file not found: {mask_path}")
        
        return masks, class_ids
                                 
        
                     

def create_dataset_results_dirs(dataset_name, results_dir):
    """creates the full results directory subfolders for given dataset
    Params:
    - dataset_name(str): name of dataset (e.g. D1e1_test)
    - results_dir (str): path to main results folder
    
    returns:
    -str: path to dataset's results directories
    """
    
    dataset_dir = os.path.join(results_dir,dataset_name)
    
    subdirs = [
        os.path.join(dataset_dir, "PP_Info"),
        os.path.join(dataset_dir, "Visualizations"),
        os.path.join(dataset_dir, "IoUs")
    ]
    
    for path in subdirs:
        os.makedirs(path, exist_ok=True)
    #print("Directories Created")
    return dataset_dir

def load_and_register_dataset(dataset_name, ROOT_DIR, results_dir, load_particle=True, load_cluster=False, create_dirs=True):
  
    images_anlyz_dir =  os.path.join(ROOT_DIR, 'data', dataset_name)    
    particle_masks_anlyz_dir = os.path.join(images_anlyz_dir, 'particle')
    cluster_masks_anlyz_dir = os.path.join(images_anlyz_dir, 'cluster')
    
    if create_dirs:                      
        dataset_results_dir = create_dataset_results_dirs(dataset_name, results_dir)
    #initalize and load dataset
    
    dataset_analyze = SAGEDataset(images_anlyz_dir,particle_masks_anlyz_dir, cluster_masks_anlyz_dir,
                                load_particle=load_particle, load_cluster=load_cluster)
    
    dataset_analyze.load_dataset(dataset_name=dataset_name)
    dataset_analyze.prepare()
    
    #add loaded dataset to dictionary
    #datasets[dataset_name] = dataset_analyze
    
    return dataset_analyze



def print_loaded_datasets(loaded_datasets):
    """
    Prints the names of all the loaded datasets.
    
    Args:
        loaded_datasets (dict): Dictionary of loaded datasets where keys are dataset names.
        
    Usage:
        print_loaded_datasets(loaded_datasets)
    """
    if not loaded_datasets:
        print("No datasets loaded.")
        return
    
    print("Loaded Datasets:")
    for dataset_name in loaded_datasets.keys():
        print(f"- {dataset_name}")

        
def print_active_models(model_dict):
    """
    Prints the names of active models
    Args: model_list (list): list of model names (strings)
    usage:
        print_active_models(model_list)
    """
    
    if not model_dict:
        print("No active models found")
        return
    for idx, (model_name, model_info) in enumerate(model_dict.items(), start=1):
        model = model_info.get('model')  # Get the actual model object
        threshold = model_info.get('confidence')  # Get the confidence threshold (if available)
        
        print(f"Model {idx}: {model_name}")
        #print(f"  Model Object: {model}")  # Prints the model instance, could be customized further to show more details
        print(f"  Confidence Threshold: {threshold}")
        print("-" * 40)  # Separating line for readability
    

    
def print_iou_summary(iou_df, model_name):
    print(f"Model: {model_name}")
    print(iou_df.head())
    print(f"Count: {len(iou_df)}")
    print(f"Range: {iou_df['iou'].min():.4f} - {iou_df['iou'].max():.4f}")
    print(f"Mean: {iou_df['iou'].mean():.4f}")
    
def get_overlaps(model, dataset, config, iou_threshold=0, sort_method='iou', verbose=False,
                 gt_set2 =None, filter=False):
    
    
    iou_values = [] #list to store IoU values
    
    image_ids = dataset.image_ids
    if not verbose:
        image_ids = tqdm(image_ids, desc="Computing Overlaps", unit="img")
        
    for image_id in image_ids:
        
        #load GT data
        image, image_meta, gt_class_ids, gt_bbox, gt_mask =\
            modellib.load_image_gt(dataset, config, image_id)
        
        #if no model is passed, load second set of GT data
        if model is None:
            if gt_set2 is None:
                raise ValueError("Either a model or second ground truth set must be provided")
            #print("2nd dataset passed)")
            _,_,gt_class_ids2, gt_bbox2, gt_mask2 =\
                modellib.load_image_gt(gt_set2, config, image_id)
            if filter:
                gt_mask2 = filter_mask_size(gt_mask2, min_dp_pix = 18)
            #calculate IoUs between GT masks    
            overlaps = compute_overlaps_masks(gt_mask, gt_mask2)
            
            if verbose:
                print(f"Image ID: {image_id}")
                print(f"GT Set 1 Bboxes: {gt_bbox.shape[0]} (GT Boxes), GT Set 2 Bboxes: {gt_bbox2.shape[0]} (Pred Boxes)")
                print(f"Overlaps matrix shape: {overlaps.shape}")
                #print(f"Overlaps: {overlaps}")
                #print(f"Predicted Scores: {len(r['scores'])} scores")
                #print(f"gt_match: {gt_match}, pred_match: {pred_match}")
                
                
        # if model is passed
        else:
            #print('Model Passed')
            #run detection
            results = model.detect([image], verbose=0)
            r= results[0]

        
            if sort_method == 'confidence':
            #print(f"ROIs: {len(r['rois'])}, Scores: {len(r['scores'])}")
                gt_match, pred_match, overlaps = compute_matches(gt_bbox, gt_class_ids, gt_mask, 
                                                             r["rois"], #pred bboxes
                                                             r["class_ids"], #pred class ids
                                                             r["scores"], #pred scores
                                                             r["masks"], #pred masks
                                                             iou_threshold=iou_threshold
                                                            )
            if sort_method == 'iou':
                overlaps1 = compute_overlaps_masks(gt_mask, r['masks'])
                
                pseudo_scores = np.max(overlaps1,axis=0)
                sorted_ix = np.argsort(pseudo_scores)[::-1]
                
                pred_boxes= r['rois'][sorted_ix]
                pred_class_ids= r['class_ids'][sorted_ix]
                pseudo_scores = pseudo_scores[sorted_ix]
                pred_masks = r['masks'][..., sorted_ix]
                
                gt_match, pred_match, overlaps = compute_matches(gt_bbox, gt_class_ids, gt_mask, 
                                                             pred_boxes, #pred bboxes
                                                             pred_class_ids, #pred class ids
                                                             pseudo_scores, #pred scores
                                                             pred_masks, #pred masks
                                                             iou_threshold=iou_threshold
                                                            )
        
            if verbose:
                print(f"Image ID: {image_id}")
                print(f"Ground Truth Bboxes: {gt_bbox.shape[0]} (GT Boxes), Predicted Bboxes: {r['rois'].shape[0]} (Pred Boxes)")
                print(f"Overlaps matrix shape: {overlaps.shape}")
                #print(f"Overlaps: {overlaps}")
                #print(f"Predicted Scores: {len(r['scores'])} scores")
                #print(f"gt_match: {gt_match}, pred_match: {pred_match}")
            max_iou_per_pred = {}
        
        for i in range(overlaps.shape[0]): #loop through pred boxes
            for j in range(overlaps.shape[1]): #loop through gt boxes
                if overlaps[i,j] >= iou_threshold:
                    entry = {
                        'image_id':dataset.image_info[image_id]['id'],
                        'gt_index':j,
                        'pred_index': i,
                        'iou': overlaps[i,j]
                        
                    }
                    
                        
                    iou_values.append(entry)
                        #'confidence score': r['scores'][i] 
                
                        
        #print(iou_values)
    return iou_values


def process_matches(model, dataset_analyze, inference_config, sort_method = 'iou',
                    iou_threshold=0, verbose=False, dataset_analyze2=None, filter=False): 
    """
    Processes prediction/ground truth matches for each image based on IoU scores, and returns a filtered DataFrame of the matches
    
    Parameters:
     
    """
    if model:
        print("Model Passed")
        if sort_method == 'confidence':
            print("--> Confidence sort")
            ious = get_overlaps(model=model,dataset=dataset_analyze, 
                                config=inference_config,sort_method='confidence',
                                 iou_threshold=iou_threshold,verbose=verbose)
        else:
            print("--> IoU sort")
            ious = get_overlaps(model=model,dataset=dataset_analyze, 
                                config=inference_config,sort_method='iou',
                                 iou_threshold=iou_threshold,verbose=verbose)
            
    elif dataset_analyze2:
        print("2nd Dataset Passed")
        model=None
        ious = get_overlaps(model=None, dataset=dataset_analyze, 
                            config=inference_config, iou_threshold=iou_threshold, gt_set2=dataset_analyze2, filter = filter)
    else:
        raise ValueError("Either a model or second dataset must be provided.")
    iou_df = pd.DataFrame(ious)
    #sort by image_id, gt_index, and iou(descending) to prioritize best IoU matches
    sorted_df = iou_df.sort_values(by=['image_id','iou'], ascending=[True, False])
    #print(sorted_df)
        


    #list to store filtered results
    unique_matches = []
    
    #iterate over each image id
    image_ids = sorted_df['image_id'].unique()
    if not verbose:
        image_ids = tqdm(image_ids, desc="Filtering matches", unit="img")
            
    for image_id in image_ids:
        if verbose:
            print(f"Processing image_id:{image_id}")
            
        #get rowsfor current image id
        image_df = sorted_df[sorted_df['image_id']==image_id]
    
        if verbose:
            print(f"Dataframe for image_id {image_id}")
            print(image_df.head(10))
    
        #track which GT and preds are already matched
        matched_gt=set()
        matched_preds=set() 
    
        match_counter = 0
        skip_counter = 0
    
        for _, row in image_df.iterrows():
            #print(f"checking row:{row.to_dict()}")
            #print(f" Pred: {row['pred_index']} and GT: {row['gt_index']} IoU: {row['iou']}")
            #if current GT box has not been matched and prediction has not been used
            if row['pred_index'] not in matched_preds and row['gt_index'] not in matched_gt:
                if verbose:
                    print(f"--> Match found: Prediction {row['pred_index']} with GT {row['gt_index']}  (IoU: {row['iou']:.4f})")
                match_counter +=1
                
                #add to list of unique matches
                unique_matches.append(row)
                #mark GT as matched
                matched_gt.add(row['gt_index'])
                #mark pred as matched
                matched_preds.add(row['pred_index'])
                
                
            elif row['pred_index'] in matched_preds:
                #print(f"--> skipped: Prediction {row['pred_index']} already matched")
                skip_counter +=1
            elif row['gt_index'] in matched_gt:
                #print(f"--> skipped: GT {row['gt_index']} already matched")
                skip_counter +=1
        if verbose:
            print(f"\n Image {image_id} processing complete")
            print(f"Matches:{match_counter}")
            print(f"skips: {skip_counter}")
        
            
    #convert list of unique matches to df
    filtered_df = pd.DataFrame(unique_matches)
    #sort by pred index
    filtered_df = filtered_df.sort_values(by=['image_id', 'pred_index'])

    if verbose:
        print("\nFinal Filtered DataFrame:")
        print(filtered_df)

    
    return filtered_df

 
def print_verbose(message, verbose_level=1):
    """Prints the message only if verbose_level is greater than 0."""
    if verbose_level > 0:
        print(message)
    
    
def load_model(model_path, model_dict, model_list, model_dir, config):
    """ Helper function to load models based on paths given in input"""
    
    model = modellib.MaskRCNN(mode="inference", 
                          config=config,
                          model_dir=model_dir)
    
    print(f"Loading model weights from: {model_path}")
    
    model.load_weights(model_path, by_name=True)
    
    model_name = os.path.basename(os.path.dirname(model_path))
    
    model_dict[model_name] = {
        "model": model,
        "confidence": config.DETECTION_MIN_CONFIDENCE, 
        "path": model_path
        }
    model_list.append(model_name)
    print(f"Loaded model: {model_name}")
    
    return model
    
    
####Postprocessing additions
def compute_equi_diam_pix(masks):
    """Computes the equivalent diameter of particle mask
    masks: [Height, Width, instances] """
    
    if masks.shape[-1] ==0:
        return np.zeros((masks.shape[-1]))
    #flatten masks and compute area
    masks = np.reshape(masks > .5, (-1,masks.shape[-1])).astype(np.float32)
    area = np.sum(masks, axis=0)
   
    dp_pix = np.sqrt((4*area)/np.pi)
      
    return dp_pix, area

def filter_mask_size(masks, min_dp_pix = 18):
    N = masks.shape[-1]
    if N == 0:
        return masks
    # Ensure binary
    masks = masks > 0.5
    #compute diameters and areas
    dp_pix, area = compute_equi_diam_pix(masks)
    print("Min dp: (pre-filter)", min(dp_pix))
    keep = np.ones(masks.shape[-1], dtype=bool)
    keep &= dp_pix >= min_dp_pix
    
    #filter
    filtered_masks = masks [:,:,keep ]
    dp_pix_filtered = dp_pix[keep]
    print("Min dp: (filtered):", min(dp_pix_filtered))
    return filtered_masks
    

def compute_mask_centroids(masks, scale=None):
    N = masks.shape[-1]
    #print("N", N)
    centroids = []
    pixel_counts = []
    
    for i in range(N):
        mask = masks[:, :, i]  # Select the i-th mask

        # Get coordinates of all pixels where the mask is True (nonzero)
        y_coords, x_coords = np.where(mask)

        # Calculate average x and y coordinates (centroid)
        x_avg = np.mean(x_coords)
        y_avg = np.mean(y_coords)

        # Count the number of pixels in the mask
        pixel_count = len(x_coords)

    # Append results
        centroids.append((x_avg, y_avg))
        pixel_counts.append(pixel_count)
    centroids = np.array(centroids)
    pixel_counts = np.array(pixel_counts)

    return centroids, pixel_counts

def calculate_radius_of_gyration(df_main):
    df = df_main.copy()
    
    #Use particle area as weight
    weights = df['PP area (nm^2)']
    
    #compute weights centroids of aggregate
    x_centroid = (df['x_avg'] * weights).sum() / weights.sum()
    y_centroid = (df['y_avg'] * weights).sum() / weights.sum()
    
    #Compute square distances of each particle from aggregate centroid
    df['distance_sq'] = (df['x_avg'] - x_centroid)**2 + (df['y_avg'] - y_centroid)**2
    
    #Caluclate weighted mean of squared distances
    weighted_mean_sq_distance = (weights * df['distance_sq']).sum() / weights.sum()

    #calculate aggregate radius of gyration
    Rg = np.sqrt(weighted_mean_sq_distance)*df['scale_length (nm)'].mean()
    
    return Rg


def compute_fractal_dimension(masks, save_binary=True, save_path=None, show=False, plot=0,):

    analyzer = SF.StereoFractAnalyzer()

    binary_mask = 255 - np.any(masks, axis=-1).astype(np.uint8)*255
    if save_binary:
        if save_path is None:
            raise ValueError("save path must be provided")
        plt.figure(figsize = (binary_mask.shape[1]/100, binary_mask.shape[0]/100),dpi=100)
        plt.imshow(binary_mask, cmap='gray')
        plt.axis('off')
        plt.tight_layout(pad=0.0)
        plt.savefig(save_path, format="png",bbox_inches='tight', pad_inches=0,orientation= 'landscape')
        plt.close()
        plt.tight_layout(pad=0.0)

        if show:
            plt.imshow(binary_mask, cmap='gray')
            plt.axis('off')
            plt.show()
            
        image_path = save_path
    
    else: 
        #save as temp image if not saving
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
            temp_filename = temp_file.name   
        Image.fromarray(binary_mask).save(temp_filename)
        image_path = temp_filename
 
    fractal_dimension = analyzer.get_image_fractal_dimension(image_path, plot=plot)
    
    if not save_binary:
        os. remove(temp_filename)
    
    return fractal_dimension


def process_particles(dataset_name, datasets, model_dict, Results_DIR, model_name=None, 
                      image_scales=None, verbose = 0, save_binary=False, show_binary_union=False, plot_df=0):
    dp_pix_all = []
    dp_nm_all = []
    records = []
    fractal_dims = []
    
    #load dataset (need to use images even if using model)
    dataset = datasets.get(dataset_name, None)
    if dataset is None:
        raise ValueError(f"Dataset '{dataset_name}' not found.")
    
    #run for each image in dataset
    image_ids = dataset.image_ids
    #progress_bar = tqdm(image_ids, desc=f"Processing '{dataset_name}' images", dynamic_ncols=True)
    #progress bar is WIP
    
    for image_id in image_ids:
        image = dataset.load_image(image_id)
        original_filename = dataset.image_info[image_id]['basename']
        base_filename = os.path.splitext(os.path.basename(original_filename))[0] 
        
        
        print(f"↳Processing Image {image_id} ({base_filename})")
        

        #get image scales from either dict or float
        if isinstance(image_scales,dict):
            scale = image_scales.get(base_filename)
            if scale is None:
                print(f"No Scale found for {base_filename}, skipping")
        else:
            scale = image_scales
        
       
        
        if model_name:
            model = model_dict[model_name]['model'] #load model
            confidence = model_dict[model_name]['confidence'] #get conf threshold
            print(f"Using Model {model_name} (conf_thresh = {confidence})")
            binary_dir = os.path.join(Results_DIR, dataset_name,'Visualizations', f"{model_name}_{confidence}", 'Binary_Unions' )
            results = model.detect([image], verbose=0)
            r = results[0]
            masks = r['masks']
                
        else:
            print(f"Analyzing Dataset Masks: {dataset_name}")
            model=None
            binary_dir = os.path.join(Results_DIR, dataset_name, 'Visualizations', dataset_name,'Binary_Unions' )
            masks,_ = dataset.load_mask(image_id)
        
        #optional: set path for saving binary unions
        binary_filename = f"{base_filename}_binary.png"  # Name the file based on the original image
        binary_path = os.path.join(binary_dir, binary_filename)
        if save_binary:
            print_verbose(f"{binary_path}", verbose)
        
        if dataset_name == 'PROCI_EDMWS':
            masks = filter_mask_size(masks,min_dp_pix = 18)
        
        #compute fractal dimension (once per image)
        fractal_dim = compute_fractal_dimension(masks, save_binary=save_binary, save_path=binary_dir, show=show_binary_union, plot=plot_df)
        fractal_dims.append({'image': base_filename, 'fractal dimension': fractal_dim})
        
        #compute particle diameters
        dp_pix, area_pix = compute_equi_diam_pix(masks)
        
        #compute centroids and pixel counts
        centroids, pixel_counts = compute_mask_centroids(masks)
        
        
        
        for i in range(len(dp_pix)):
            x_avg, y_avg = centroids[i]
            pixel_count = pixel_counts[i]
            dp_nm = dp_pix[i]*scale
            
            records.append({
                "image": base_filename,
                "scale_length (nm)": scale,
                "scale_area (nm^2)": scale**2,
                "PP #": i+1,
                "dp (pix)": dp_pix[i],
                "dp (nm)" : dp_nm,
                "PP area (pix)": area_pix[i],
                "PP area (nm^2)": 0.25*np.pi*dp_nm**2,
                "x_avg": x_avg,
                "y_avg": y_avg,
                "num_pixels": pixel_count
                
            })
            dp_pix_all.append(dp_pix[i])
            dp_nm_all.append(dp_nm)

            
    df =pd.DataFrame(records)
    df_fd = pd.DataFrame(fractal_dims)
    
    return df, df_fd, dp_pix_all, dp_nm_all, 

def gather_aggregate_morphology(ref_set, scales, settings, 
                               save_binary=False, show_binary=False, plot=False):
    """wrapper function to gather and save aggregate morphology information"""
    
    dataset_name = settings['dataset_name']
    datasets = settings['datasets']
    model_dict = settings['model_dict']
    Results_DIR = settings['Results_DIR']
    model_name = settings.get('model_name')
    save_results = settings.get('save_results', 0)
    verbose = settings.get('verbose', 0)
    
    #calculate dp and get mean for each image
    dataset = datasets.get(dataset_name, None)
    pp_output_dir = os.path.join(Results_DIR, ref_set, 'PP_Info')
    os.makedirs(pp_output_dir, exist_ok=True)
    if model_name:
        model = model_dict[model_name]['model']
        confidence = model_dict[model_name]['confidence']
        method = model_name
        pp_save_name = os.path.join(pp_output_dir,f"{model_name}_{confidence}_pp_info.csv")
        #add save path
    else:
        model=None
        method = dataset_name
        pp_save_name = os.path.join(pp_output_dir, f"{dataset_name}_pp_info.csv")
    #g -- Process individual particles and fractal dimension
    
    #(dataset_name, model_name=None, image_scales=None, save_binary=False, show_binary_union=False, plot_df=0):
    
    #Process individual particles (give df with dp info)
    df_particles, fractal_dims, dp_pix_all, dp_nm_all = process_particles(dataset_name, datasets, model_dict, Results_DIR, model_name, scales, verbose, 
                                                                          save_binary, show_binary, plot)
    
    #print_verbose(f" Fractal Dimension df {fractal_dims}", verbose)
    #print_verbose(f"", verbose)
    #print_verbose(f"{df_particles.head(5)}", verbose)
    
    particles = df_particles.image.unique()
    
    N_pp = []
    mean_dp_pix = []
    SEM_mdp_pix = []
    STD_mdp_pix = []
    mean_dp_nm = []
    SEM_mdp_nm = []
    STD_mdp_nm = []
    Rg_pix = []
    Rg_nm = []
    methods = []
    scale_list = []

    for i, particle in enumerate(particles):
        particle_data = df_particles[df_particles['image']==particle].copy()
        #print(particle_data)
        scale_length = particle_data['scale_length (nm)'].mean()
        scale_list.append(scale_length)
        print_verbose(f"Processing Image: {particle}", verbose)
        print_verbose(f" ", verbose)
        
        # --Primary particle count
        N = len(particle_data)
        print_verbose(f"---> # of PP: {N}", verbose) 
        N_pp.append(N)
        print_verbose(f" ", verbose)
        
        # -- dp statistics (pixels)
        mdp_pix = particle_data['dp (pix)'].mean()
        sem_dp_pix = particle_data['dp (pix)'].sem()
        std_dp_pix =  particle_data['dp (pix)'].std()
        print_verbose(f"---> Mean dp(pix): {mdp_pix} ||| SEM: {sem_dp_pix} & STD: {std_dp_pix}", verbose)
        mean_dp_pix.append(mdp_pix)
        SEM_mdp_pix.append(sem_dp_pix)
        STD_mdp_pix.append(std_dp_pix)
        
        # -- dp statistics (nm)
        mdp_nm = particle_data['dp (nm)'].mean()
        sem_dp_nm = particle_data['dp (nm)'].sem()
        std_dp_nm =  particle_data['dp (nm)'].std()
        print_verbose(f"---> Mean dp(nm): {mdp_nm} ||| SEM: {sem_dp_nm} & STD: {std_dp_nm}", verbose)
        mean_dp_nm.append(mdp_nm)
        SEM_mdp_nm.append(sem_dp_nm)
        STD_mdp_nm.append(std_dp_nm)
        
        
        # -- Radius of Gyration
        radius = calculate_radius_of_gyration(particle_data)
        print_verbose(f"---> Radius of Gyration (nm) for {particle}: {radius}", verbose)
        Rg_nm.append(radius)
        print_verbose(f"---> Radius of Gyration (pix) for {particle}: {radius/scale_length}", verbose)
        Rg_pix.append(radius/scale_length)
        
        # -- Fractal Dimension
        fractal_dim = fractal_dims.iloc[i]['fractal dimension']
        print_verbose(f"---> Fractal Dimension for {particle}: {fractal_dim}", verbose)
        print_verbose(f"", verbose)
        methods.append(method)
        print(f"--> N: {N} | Mean dp (nm): {mdp_nm:.3f} (SEM: {sem_dp_nm:.3f}, STD: {std_dp_nm}) | Rg (nm): {radius:.3f} | Fractal Dim: {fractal_dim:.3f}")
        
        print_verbose(f"", verbose)
    # --Final Aggregate Summary                  
    aggregates = pd.DataFrame({'image': particles, 
                               "method": methods,
                               "length scale [nm/pix]": scale_list,
                               "# of PP": N_pp,
                               "Mean dp [pix]": mean_dp_pix,
                               "mdp SEM [pix]": SEM_mdp_pix,
                               "mdp STD [pix]": STD_mdp_pix,
                               "Mean dp [nm]": mean_dp_nm,
                               "mdp SEM [nm]": SEM_mdp_nm,
                               "mdp STD [nm]": STD_mdp_nm,
                               "Rg [pix]": Rg_pix,
                               "Rg [nm]": Rg_nm,
                               "fractal_dim": fractal_dims['fractal dimension'],
                               
                              })
    
    # -- Create save path based on reference dataset
    save_dir = os.path.join(Results_DIR, ref_set)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir,"Aggregate_info.csv")
    
    # -- check if file exists to append if it does
    if save_results:
        df_particles.to_csv(pp_save_name, index=False)
        print(f"Saved primary particle info to {pp_save_name}")
        if os.path.exists(save_path):
            aggregates.to_csv(save_path, mode='a', header=False, index=False)
            print(f"Appended aggregate summary to {save_path}")
        else:
            aggregates.to_csv(save_path, index=False)
            print(f"Created new aggregate summary for {ref_set}: {save_path}")
                          
    return aggregates, df_particles


def calc_performance_metrics(ref_name, inference_config, settings,sort_method='iou'):
    
    dataset_name = settings['dataset_name']
    datasets = settings['datasets']
    model_dict = settings['model_dict']
    Results_DIR = settings['Results_DIR']
    model_name = settings.get('model_name')
    save_results = settings.get('save_results', 0)
    verbose = settings.get('verbose', 0)
    
    
    ref_data = datasets.get(ref_name, None)
    if model_name:
        model = model_dict[model_name]['model']
        confidence = model_dict[model_name]['confidence']
        method = model_name
        
        filtered_df = process_matches(model, ref_data, inference_config, sort_method=sort_method,iou_threshold=0, verbose=False)
        #pp_save_name = os.path.join(pp_output_dir,f"{model_name}_{confidence}_pp_info.csv")
        #add save path
    else:
        model=None
        method = dataset_name
        dataset = datasets.get(dataset_name, None)
        print(f"dataset_name: {dataset_name}, loaded dataset: {dataset}")
        if dataset_name =='PROCI_EDMWS' or dataset_name == 'PROCI_EDMWS2':
            filter = True
        else: 
            filter = False
        filtered_df = process_matches(model=None, 
                                       dataset_analyze=ref_data,
                                      inference_config=inference_config,
                                      iou_threshold=0, sort_method=sort_method,
                                      verbose=False,
                                      dataset_analyze2=dataset, filter=filter)
    
    #print(filtered_df)
    
    gt_tot = np.array([])
    pred_tot = np.array([])
    mAP_conf = []
    mAP_iou = []
    metrics_list = []
    mAP_ = []
    AP75_ = []
    mAP_range_ = []
    precision_range_ = []
    recall_range_ = []
    
    
    print_verbose(f"Analyzing {method} performance on {ref_name}", verbose)
    
    print_verbose(f"Sorting {method} predictions by {sort_method}", verbose)
    
    tp_list = []
    fp_list = []
    fn_list = []
    prec_list = []
    
    for image_id in ref_data.image_ids:
        #get data from reference (ground truth) dataset
        image, image_meta, gt_class_id, gt_bbox, gt_mask =\
            modellib.load_image_gt(ref_data, inference_config, image_id)#, use_mini_mask=False)
        info = ref_data.image_info[image_id]
        original_filename = ref_data.image_info[image_id]['basename']
        base_filename = os.path.splitext(os.path.basename(original_filename))[0]
      
        
        tp =0
        fp=0
        fn=0
        
        if model:
            results = model.detect([image], verbose=0)
            r = results[0]
            
            print_verbose(f"\n🔍 Processing image {base_filename} (ID: {image_id})", verbose)
            print_verbose(f"Detected {len(r['class_ids'])} predictions for {base_filename}", verbose)
            
            gt, pred = gt_pred_lists(gt_class_id, gt_bbox, r['class_ids'], r['rois'])
            gt = np.array(gt).astype(int)
            pred = np.array(pred).astype(int)
    
            gt_tot = np.append(gt_tot, gt)
            pred_tot = np.append(pred_tot, pred)
            #print("the actual len of the gt vect is : ", len(gt_tot))
            #print("the actual len of the pred vect is : ", len(pred_tot))
            print_verbose(f"Ground Truth: {len(gt_class_id)} objects | Predictions: {len(r['class_ids'])} objects", verbose)
            
            if sort_method =='confidence':
                '''calcualates metrics using confidence scores to order and decide matches'''
                
                AP_, precision_, recall_, overlap_= compute_ap(gt_bbox, gt_class_id, gt_mask,
                                          r['rois'], r['class_ids'], r['scores'], r['masks'])
                mAP_.append(AP_)
                
                AP75, _, _, _ = compute_ap(gt_bbox,gt_class_id,gt_mask,r['rois'],
                                                 r['class_ids'], r['scores'], r['masks'], iou_threshold=0.75)
                
                AP75_.append(AP75)
                mean_ap_ = compute_ap_range(gt_bbox, gt_class_id, gt_mask, 
                                                                          r['rois'], r['class_ids'], r['scores'], r['masks'])
                
                mAP_range_.append(mean_ap_)
                
                print_verbose(f"AP (conf sort) for {base_filename}: {AP_}",verbose )
                print_verbose(f"AP 0.75: {AP75}", verbose)
                print_verbose(f"", verbose)
                print_verbose(f"AP(ranged) (conf sort) for {base_filename}: {mean_ap_}",verbose)
                #print_verbose(f"Precision (conf sort) for {base_filename}: {precision_conf}", verbose)
                #print_verbose(f"Recall (conf sort) for {base_filename}: {recall_conf}", verbose)
                
            if sort_method =='iou':
                '''use IoU as a 'pseudo score' and sort the same way confidence scores 
                sorted for use in matching and AP calculations'''
                overlaps = compute_overlaps_masks(gt_mask, r['masks'])
                
                pseudo_scores = np.max(overlaps,axis=0)
                sorted_ix = np.argsort(pseudo_scores)[::-1]
                
                pred_boxes= r['rois'][sorted_ix]
                pred_class_ids= r['class_ids'][sorted_ix]
                pseudo_scores = pseudo_scores[sorted_ix]
                pred_masks = r['masks'][..., sorted_ix]
                
                AP_, precision_, recall_, overlap_ = compute_ap(gt_bbox, gt_class_id, gt_mask,
                                          pred_boxes, pred_class_ids, pseudo_scores, pred_masks)
                mAP_.append(AP_)
                AP75, _, _, _= compute_ap(gt_bbox, gt_class_id, gt_mask, pred_boxes, 
                                            pred_class_ids, pseudo_scores, pred_masks, iou_threshold=0.75)
                
                AP75_.append(AP75)
                mean_ap_  = compute_ap_range(gt_bbox, gt_class_id, gt_mask, 
                                                                         pred_boxes, pred_class_ids, pseudo_scores, pred_masks)
                
                mAP_range_.append(mean_ap_)
                
                
                print_verbose(f"AP (iou sort) for {base_filename}: {AP_}", verbose)
                print_verbose(f"AP 0.75: {AP75}", verbose)
                print_verbose(f"AP(ranged) (conf sort) for {base_filename}: {mean_ap_}",verbose)
                
                #print_verbose(f"Precision (iou sort) for {base_filename}: {precision_iou}", verbose)
                #print_verbose(f"Recall (iou sort) for {base_filename}: {recall_iou}", verbose)
                
                
                print_verbose(f"", verbose)
                
        else:
            _,_, data_class_id,data_bbox, data_mask =\
                modellib.load_image_gt(dataset, inference_config, image_id)#, use_mini_mask=False)
            print_verbose(f"[Before filtering] {base_filename}: {data_mask.shape[-1]} masks", verbose)
            if dataset_name == 'PROCI_EDMWS' or dataset_name == 'PROCI_EDMWS2':
                data_mask = filter_mask_size(data_mask, min_dp_pix = 18)
                
            print_verbose(f"[After filtering] {base_filename}: {data_mask.shape[-1]} masks", verbose)
            print_verbose(f"\n🔍 Processing image {base_filename} (ID: {image_id})", verbose)
            print_verbose(f"Detected {len(data_class_id)} predictions for {base_filename}", verbose)
            
            gt, pred = gt_pred_lists(gt_class_id, gt_bbox, data_class_id, data_bbox)
            gt = np.array(gt).astype(int)
            pred = np.array(pred).astype(int)

            gt_tot = np.append(gt_tot, gt)
            pred_tot = np.append(pred_tot, pred)
            #print("the actual len of the gt vect is : ", len(gt_tot))
            #print("the actual len of the pred vect is : ", len(pred_tot))
            print_verbose(f"Ground Truth: {len(gt_class_id)} objects | Predictions: {len(data_class_id)} objects", verbose)
                
            overlaps = compute_overlaps_masks(gt_mask, data_mask)
                
            pseudo_scores = np.max(overlaps,axis=0)
            sorted_ix = np.argsort(pseudo_scores)[::-1]
                
            pred_boxes=data_bbox[sorted_ix]
            pred_class_ids= data_class_id[sorted_ix]
            pseudo_scores = pseudo_scores[sorted_ix]
            pred_masks = data_mask[..., sorted_ix]
                
            AP_, precision_, recall_, overlap_ = compute_ap(gt_bbox, gt_class_id, gt_mask,
                                      pred_boxes, pred_class_ids, pseudo_scores, pred_masks)
            print_verbose(f"{AP_}", verbose)
            mAP_.append(AP_)
            
            AP75, _, _, _= compute_ap(gt_bbox, gt_class_id, gt_mask, pred_boxes, 
                                            pred_class_ids, pseudo_scores, pred_masks, iou_threshold=0.75)
                
            AP75_.append(AP75)
            
             
            mean_ap_ = compute_ap_range(gt_bbox, gt_class_id, gt_mask, 
                                                                         pred_boxes, pred_class_ids, pseudo_scores, pred_masks)
                
            mAP_range_.append(mean_ap_)
            
            
            
            print_verbose(f"AP (iou sort) for {base_filename}: {AP_}", verbose)
            print_verbose(f"AP(ranged) (conf sort) for {base_filename}: {mean_ap_}",verbose)
            
            print_verbose(f"{mAP_}")
        #try ap through the df
        image_df = filtered_df[filtered_df['image_id']==image_id]
            
        tp = len(image_df[image_df['iou']>=0.5])
        fp = len(image_df[image_df['iou']<0.5])
        gtlen = len(gt_class_id)
        #print("tot_gt", gtlen)
            
        fn = gtlen - tp - fp
        
        prec_2 = tp/(tp+fp+fn)
        print_verbose(f"AP from TP,etc on {base_filename}: {prec_2}", verbose)
        tp_list.append(tp)
        fp_list.append(fp) 
        fn_list.append(fn)
        prec_list.append(prec_2)
            
    print_verbose(f"TP list {tp_list}", verbose)
    print_verbose(f"FP list {fp_list}", verbose)
    print_verbose(f"FN list {fn_list}",verbose)
    
    tp = np.sum(tp_list)  
    fp = np.sum(fp_list)  
    fn = np.sum(fn_list)
    
    print(f"CM Values TP: {tp}, FP {fp}, FN: {fn}")
    #prec_tot = tp_tot/(tp_tot+fp_tot+fn_tot)
    #print(prec_tot)
    prec_ave = np.mean(prec_list)
    #print(prec_ave)        
    
                
    mean_iou = filtered_df['iou'].mean()
    print("Mean IoU: ",mean_iou)           
    final_mAP = sum(mAP_)/len(mAP_)    
    print(f"Final AP_0.5: {final_mAP}")
    final_AP75 = sum(AP75_)/len(mAP_)
    print(f"Final AP_0.75: {final_AP75}") 
    final_mAP_range = sum(mAP_range_)/len(mAP_range_)
    print(f"Final mAP (ranged): {final_mAP_range}")
    
    accuracy = (tp) / (tp+fp+fn)
    print('Accuracy:', accuracy)
    precision = tp/(tp+fp)
    print('Precision:', precision)
    recall = tp/(tp+fn)
    print('Recall:', recall)
    F1 = tp/(tp+0.5*(fp+fn))
    print('F1 Score:',F1)
            
        
    metrics_dict = {
    "Method" : method,
    "Test Dataset": ref_name,
    "Confidence Threshold": model_dict[model_name]['confidence'] if model else 'N/A',
    "TP": int(tp),
    "FP": int(fp),
    "FN": int(fn),
    "Accuracy": round(accuracy,4),
    "Precision": round(precision,4),
    "Recall": round(recall,4),
    "F1": round(F1,4),
    "AP_50":round(final_mAP,4),
    "AP_75":round(final_AP75,4),
    "AP_range": round(final_mAP_range,4),
    "Mean IoU": round(mean_iou,4)
    }
    metrics_list.append(metrics_dict)
    metrics_df = pd.DataFrame(metrics_list)

    print(metrics_df.head(5))

    if save_results:
        output_path = os.path.join(Results_DIR, "Metrics") if sort_method == 'confidence' else os.path.join(Results_DIR, "Metrics", "IoUSort" )
        csv_path = os.path.join(output_path, f"{method}_metrics.csv")
        if os.path.exists(csv_path):
            print(f"appended to: {csv_path}")
            metrics_df.to_csv(csv_path, mode='a',header=False, index=False)
        else:
            print(f"created at: {csv_path}")
            metrics_df.to_csv(csv_path, mode='w', header=True, index=False)

            #pp_save_name = os.path.join(pp_output_dir, f"{dataset_name}_pp_info.csv")
        
    return metrics_df

def full_summary(aggs_df, metrics_df, pp_df, settings):
  
 
    Results_DIR = settings['Results_DIR']
    save_results = settings.get('save_results', 0)
    verbose = settings.get('verbose', 0)
    
    summary_dict = {
        "Method": metrics_df['Method'].iloc[0],
        "Conf Thresh": metrics_df['Confidence Threshold'].iloc[0],
        "TP": metrics_df['TP'].iloc[0],
        "FP": metrics_df['FP'].iloc[0],
        "FN": metrics_df['FN'].iloc[0],
        "Acc.": metrics_df['Accuracy'].iloc[0],
        "Prec.": metrics_df['Precision'].iloc[0],
        "Rec.": metrics_df['Recall'].iloc[0],
        "F1": metrics_df['F1'].iloc[0],
        "AP50": metrics_df['AP_50'].iloc[0],
        "AP75": metrics_df['AP_75'].iloc[0],
        "mAP": metrics_df['AP_range'].iloc[0],
        "Mean IoU": metrics_df['Mean IoU'].iloc[0],
        "Mean dF": round(aggs_df['fractal_dim'].mean(),4),
        "dF SEM": round(aggs_df['fractal_dim'].sem(),4),
        "df STD": round(aggs_df['fractal_dim'].std(),4),
        "Mean Rg [nm]": round(aggs_df['Rg [nm]'].mean(),4),
        "Rg SEM": round(aggs_df['Rg [nm]'].sem(),4),
        "Rg STD": round(aggs_df['Rg [nm]'].std(),4),
        "Mean dp [nm]": round( pp_df['dp (nm)'].mean(),4),
        "dp SEM": round( pp_df['dp (nm)'].sem(),4),
        "dp STD": round( pp_df['dp (nm)'].std(),4),
    }
    if verbose:
        for k,v in summary_dict.items():
            print(f"{k}: {v}")
            
    summary_df = pd.DataFrame([summary_dict])
    if save_results:
        output_dir = os.path.join(Results_DIR,metrics_df['Test Dataset'].iloc[0])
        #print(output_dir)
        output_path = os.path.join(output_dir, "Full Summary.csv")
        if os.path.exists(output_path):
            print(f"appended to: {output_path}")
            summary_df.to_csv(output_path, mode='a',header=False, index=False)
        else:
            print(f"created at: {output_path}")
            summary_df.to_csv(output_path, mode='w', header=True, index=False)

            #pp_save_name = os.path.join(pp_output_dir, f"{dataset_name}_pp_info.csv")
        
        
        
    return summary_df
                  

####### Confusion matrix ########

#function 1 to be added to your utils.py
def get_iou(a, b, epsilon=1e-5):
    """ 
    Given two boxes `a` and `b` defined as a list of four numbers:
            [x1,y1,x2,y2]
        where:
            x1,y1 represent the upper left corner
            x2,y2 represent the lower right corner
        It returns the Intersect of Union score for these two boxes.

    Args: 
        a:          (list of 4 numbers) [x1,y1,x2,y2]
        b:          (list of 4 numbers) [x1,y1,x2,y2]
        epsilon:    (float) Small value to prevent division by zero

    Returns:
        (float) The Intersect of Union score.
    """
    # COORDINATES OF THE INTERSECTION BOX
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    # AREA OF OVERLAP - Area where the boxes intersect
    width = (x2 - x1)
    height = (y2 - y1)
    # handle case where there is NO overlap
    if (width<0) or (height <0):
        return 0.0
    area_overlap = width * height

    # COMBINED AREA
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    area_combined = area_a + area_b - area_overlap

    # RATIO OF AREA OF OVERLAP OVER COMBINED AREA
    iou = area_overlap / (area_combined+epsilon)
    return iou


#function 2 to be added to your utils.py
def gt_pred_lists(gt_class_ids, gt_bboxes, pred_class_ids, pred_bboxes, iou_tresh = 0.5):

    """ 
        Given a list of ground truth and predicted classes and their boxes, 
        this function associates the predicted classes to their gt classes using a given Iou (Iou>= 0.5 for example) and returns 
        two normalized lists of len = N containing the gt and predicted classes, 
        filling the non-predicted and miss-predicted classes by the background class (index 0).

        Args    :
            gt_class_ids   :    list of gt classes of size N1
            pred_class_ids :    list of predicted classes of size N2
            gt_bboxes      :    list of gt boxes [N1, (x1, y1, x2, y2)]
            pred_bboxes    :    list of pred boxes [N2, (x1, y1, x2, y2)]
            
        Returns : 
            gt             :    list of size N
            pred           :    list of size N 

    """

    #dict containing the state of each gt and predicted class (0 : not associated to any other class, 1 : associated to a class)
    gt_class_ids_ = {'state' : [0*i for i in range(len(gt_class_ids))], "gt_class_ids":list(gt_class_ids)}
    pred_class_ids_ = {'state' : [0*i for i in range(len(pred_class_ids))], "pred_class_ids":list(pred_class_ids)}

    #the two lists to be returned
    pred=[]
    gt=[]

    for i, gt_class in enumerate(gt_class_ids_["gt_class_ids"]):
        for j, pred_class in enumerate(pred_class_ids_['pred_class_ids']): 
            #check if the gt object is overlapping with a predicted object
            if get_iou(gt_bboxes[i], pred_bboxes[j])>=iou_tresh:
                #change the state of the gt and predicted class when an overlapping is found
                gt_class_ids_['state'][i] = 1
                pred_class_ids_['state'][j] = 1
                #gt.append(gt_class)
                #pred.append(pred_class)
                
                #chack if the overlapping objects are from the same class
                if (gt_class == pred_class):
                    gt.append(gt_class)
                    pred.append(pred_class)
                #if the overlapping objects are not from the same class 
                else : 
                    gt.append(gt_class)
                    pred.append(pred_class)
                
    #look for objects that are not predicted (gt objects that dont exists in pred objects)
    for i, gt_class in enumerate(gt_class_ids_["gt_class_ids"]):
        if gt_class_ids_['state'][i] == 0:
            gt.append(gt_class)
            pred.append(0)
            #match_id += 1
    #look for objects that are mispredicted (pred objects that dont exists in gt objects)
    for j, pred_class in enumerate(pred_class_ids_["pred_class_ids"]):
        if pred_class_ids_['state'][j] == 0:
            gt.append(0)
            pred.append(pred_class)
    return gt, pred



#########  Print confusion matrix for the whole dataset and return tp,fp and fn ##########
#########  The style of this confusion matrix is inspired from https://github.com/wcipriano/pretty-print-confusion-matrix ##########

def get_new_fig(fn, figsize=[9,9]):
    """ Init graphics """
    fig1 = plt.figure(fn, figsize)
    ax1 = fig1.gca()   #Get Current Axis
    ax1.cla() # clear existing plot
    return fig1, ax1
#

def configcell_text_and_colors(array_df, lin, col, oText, facecolors, posi, fz, fmt, show_null_values=0, show_percentages=False, show_totals = False, show_total_percentages=False):
    """
      config cell text and colors
      and return text elements to add and to dell
      @TODO: use fmt
    """
    text_add = []; text_del = [];
    cell_val = array_df[lin][col]
    tot_all = array_df[-1][-1]
    per = (float(cell_val) / tot_all) * 100
    curr_column = array_df[:,col]
    ccl = len(curr_column)
    
    
    #last line  and/or last column
    if(col == (ccl - 1)) or (lin == (ccl - 1)):
        """
        #tots and percents
        if(cell_val != 0):
            if(col == ccl - 1) and (lin == ccl - 1):
                tot_rig = 0
                for i in range(array_df.shape[0] - 1):
                    tot_rig += array_df[i][i]
                per_ok = (float(tot_rig) / cell_val) * 100
            elif(col == ccl - 1):
                tot_rig = array_df[lin][lin]
                per_ok = (float(tot_rig) / cell_val) * 100
            elif(lin == ccl - 1):
                tot_rig = array_df[col][col]
                per_ok = (float(tot_rig) / cell_val) * 100
            per_err = 100 - per_ok
        else:
            per_ok = per_err = 0

        per_ok_s = ['%.2f%%'%(per_ok), '100%'] [per_ok == 100]

        #text to DEL
        text_del.append(oText)

        #set background color for sum cells (last line and last column)
        carr = [0.27, 0.30, 0.27, 1.0]
        if(col == ccl - 1) and (lin == ccl - 1):
            carr = [0.17, 0.20, 0.17, 1.0]
        facecolors[posi] = carr

        #calc luminence
        r, g, b, _ = carr
        luminance = r * 0.3 + g * 0.59 + b * 0.11

        if luminance <0.5:
            text_color = 'w'
        else:
            text_color = 'k'

        #text to ADD
        font_prop = fm.FontProperties(weight='bold', size=fz)
        text_kwargs = dict(color = text_color, ha="center", va="center", gid='sum', fontproperties=font_prop)
        #text_kwargs = dict(color='w', ha="center", va="center", gid='sum', fontproperties=font_prop)


        if show_total_percentages:
            lis_txt = ['%d'%(cell_val), per_ok_s, '%.2f%%'%(per_err)]
        else: 
            lis_txt = ['%d'%(cell_val)]
        lis_kwa = [text_kwargs]


        #dic = text_kwargs.copy(); dic['color'] = 'g'; lis_kwa.append(dic);
        #dic = text_kwargs.copy(); dic['color'] = 'r'; lis_kwa.append(dic);
        lis_pos = [(oText._x, oText._y-0.3), (oText._x, oText._y), (oText._x, oText._y+0.3)]
        #for i in range(len(lis_txt)):
         #   newText = dict(x=lis_pos[i][0], y=lis_pos[i][1], text=lis_txt[i], kw=lis_kwa[i])
          #  #print 'lin: %s, col: %s, newText: %s' %(lin, col, newText)
           # text_add.append(newText)
        #print '\n'

        dic = text_kwargs.copy(); dic['color'] = 'g'; lis_kwa.append(dic);
        dic = text_kwargs.copy(); dic['color'] = 'r'; lis_kwa.append(dic);


        for i in range(len(lis_txt)):
            newText = dict(x=lis_pos[i][0], y=lis_pos[i][1], text=lis_txt[i], kw=text_kwargs)
            #print 'lin: %s, col: %s, newText: %s' %(lin, col, newText)
            text_add.append(newText)"""
    else:
        if(per > 0):
            if show_percentages:
                txt = '%s\n%.2f%%' %(cell_val, per)
            else: txt = '%s' %(cell_val)
        else:
            if(show_null_values == 0):
                txt = ''
            elif(show_null_values == 1):
                txt = '0'
            else:
                txt = '0\n0.0%'
        oText.set_text(txt)

        #calculate luminence based on the cell background color
        r, g, b, _ = facecolors[posi]
        luminence = r*0.3 + g*0.59 + b*0.11 
        if luminence < 0.5:
            oText.set_color('white') #use white text for dark backgrounds
        else:
            oText.set_color('black') #use black text for light backgrounds


        #main diagonal
        if(col == lin):

            #calculate luminence based on the cell background color
            r, g, b, _ = facecolors[posi]
            luminence = r*0.3 + g*0.59 + b*0.11 
            if luminence < 0.5:
                oText.set_color('white') #use white text for dark backgrounds
            else:
                oText.set_color('black') #use black text for light backgrounds

            # set background color in the diagonal to blue
            facecolors[posi] = [0.35, 0.8, 0.55, 1.0]
        else:
            oText.set_color('r')


    return text_add, text_del
#

def insert_totals(df_cm):
    """ insert total column and line (the last ones) """
    sum_col = []
    for c in df_cm.columns:
        sum_col.append( df_cm[c].sum() )
    sum_lin = []
    for item_line in df_cm.iterrows():
        sum_lin.append( item_line[1].sum() )
    df_cm['sum_lin'] = sum_lin
    sum_col.append(np.sum(sum_lin))
    df_cm.loc['sum_col'] = sum_col
    #print ('\ndf_cm:\n', df_cm, '\n\b\n')
#

def pretty_plot_confusion_matrix(df_cm, annot=True, cmap="viridis", fmt='.2f', fz=11,
      lw=0.5, cbar=False, figsize=[8,8], show_null_values=0, pred_val_axis='y,',show_totals=False,save_path=None):
    """
      print conf matrix with default layout (like matlab)
      params:
        df_cm          dataframe (pandas) without totals
        annot          print text in each cell
        cmap           Oranges,Oranges_r,YlGnBu,Blues,RdBu, ... see:
        fz             fontsize
        lw             linewidth
        pred_val_axis  where to show the prediction values (x or y axis)
                        'col' or 'x': show predicted values in columns (x axis) instead lines
                        'lin' or 'y': show predicted values in lines   (y axis)
    """
    if(pred_val_axis in ('col', 'x')):
        xlbl = 'Predicted'
        ylbl = 'Actual'
    else:
        xlbl = 'Actual'
        ylbl = 'Predicted'
        df_cm = df_cm.T

    if show_totals:
        # create "Total" column
        insert_totals(df_cm)

    #this is for print allways in the same window
    fig, ax1 = get_new_fig('Conf matrix default', figsize)

    #thanks for seaborn
    sn.set(font_scale=1.8)
    ax = sn.heatmap(df_cm, annot=annot, annot_kws={"size": fz}, linewidths=lw, ax=ax1,
                    cbar=cbar, cmap=cmap, linecolor='w', fmt=fmt)
    

    #set ticklabels rotation
    ax.set_xticklabels(ax.get_xticklabels(), rotation = 75, fontsize = 26)
    ax.set_yticklabels(ax.get_yticklabels(), rotation = 25, fontsize = 26)

    # Turn off all the ticks
    for t in ax.xaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False
    for t in ax.yaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False

    #face colors list
    quadmesh = ax.findobj(QuadMesh)[0]
    facecolors = quadmesh.get_facecolors()

    #iter in text elements
    array_df = np.array( df_cm.to_records(index=False).tolist() )
    text_add = []; text_del = [];
    posi = -1 #from left to right, bottom to top.
    for t in ax.collections[0].axes.texts: #ax.texts:
        pos = np.array( t.get_position()) - [0.5,0.5]
        lin = int(pos[1]); col = int(pos[0]);
        posi += 1
        #print ('>>> pos: %s, posi: %s, val: %s, txt: %s' %(pos, posi, array_df[lin][col], t.get_text()))

        #set text
        txt_res = configcell_text_and_colors(array_df, lin, col, t, facecolors, posi, fz, fmt, show_null_values)

        text_add.extend(txt_res[0])
        text_del.extend(txt_res[1])

    #remove the old ones
    for item in text_del:
        item.remove()
    #append the new ones
    for item in text_add:
        ax.text(item['x'], item['y'], item['text'], **item['kw'])

    #titles and legends
    ax.set_title('Confusion matrix')
    ax.set_xlabel(xlbl)
    ax.set_ylabel(ylbl)
    plt.tight_layout()  #set layout slim
    if save_path:
        plt.savefig(save_path)
    plt.show()
#

def plot_confusion_matrix_from_data(y_test, predictions, columns=None, annot=True, cmap="viridis",
      fmt='.2f', fz=11, lw=0.5, cbar=False, figsize=[36,36], show_null_values=0, pred_val_axis='lin',show_totals=False, save_path=None):
    """
        plot confusion matrix function with y_test (actual values) and predictions (predic),
        whitout a confusion matrix yet
        return the tp, fp and fn
    """

    #data
    if(not columns):
        columns = ['class %s' %(i) for i in list(ascii_uppercase)[0:max(len(np.unique(y_test)),len(np.unique(predictions)))]]
    
    y_test = np.array(y_test)
    predictions = np.array(predictions)
    #confusion matrix 
    confm = confusion_matrix(y_test, predictions)
    num_classes = len(columns)
    
    #compute tp fn fp 
    
    fp=[0]*num_classes
    fn=[0]*num_classes
    tp=[0]*num_classes
    tn=[0]*num_classes
    for i in range(confm.shape[0]):
        fn[i]+=np.sum(confm[i])-np.diag(confm)[i]
        fp[i]+=np.sum(np.transpose(confm)[i])-np.diag(confm)[i]
        for j in range(confm.shape[1]):
            if i==j:
                tp[i]+=confm[i][j]
                #print(confm)
    #compute tn
    for i in range(num_classes):
        tn[i] = np.sum(confm) - (tp[i] + fp[i] + fn[i])  # Total - (TP + FP + FN)
    
    #print(confm)
    #plot
    df_cm = DataFrame(confm, index=columns, columns=columns)

    pretty_plot_confusion_matrix(df_cm, fz=fz, cmap=cmap, figsize=figsize, show_null_values=show_null_values, 
        pred_val_axis=pred_val_axis, lw=lw, fmt=fmt, show_totals=show_totals, save_path=save_path)
    
    return tp, fp, fn, tn
  