"""
CROPING IS DONE BASE ON NNUNET CODE
#https://github.com/MIC-DKFZ/acvl_utils/blob/master/acvl_utils/cropping_and_padding/bounding_boxes.py

I have added comments to better understant what is each function doing. I did not change any code.
"""
import numpy as np
from scipy.ndimage import binary_fill_holes
from typing import List, Tuple, Union


def create_nonzero_mask(data):
    """
    Script which creates mask. It sets 1 where foreground (any class) is pressent (1/0).

    Args:
        * data, np.ndarray, it must be 3d np.ndarray, either data or seg mask. 
        It must be in a shape (C, X, Y, Z)

    Returns:
        * np.ndarray, binary array with filled holes.    
    """
    assert data.ndim in (3, 4), "data must have shape (C, X, Y, Z) or shape (C, X, Y)"
    nonzero_mask = data[0] != 0
    for c in range(1, data.shape[0]):
        nonzero_mask |= data[c] != 0
    return binary_fill_holes(nonzero_mask)

def bounding_box_to_slice(bounding_box: List[List[int]]):
    """
    This function reformat [[min_x, max_x], [min_y, max_y], [min_z, max_z]] to the 
    tupple contating slice objects which can be then applied in np.ndarray to crop 
    out the bounding box

    Args:
        * bounding_box, list of lists, this arguments is obtained by get_bbox_from_mask

    Returns:
        * sliced array that contains 3 different slices (one for each axis)
    """
    return tuple([slice(*i) for i in bounding_box])

def crop_to_bbox(array: np.ndarray, bounding_box: List[List[int]]):
    """
    Function which crops the box out of a given np.ndarray

    Args:
        * array, np.ndarray, input array which needs to be sliced
        * bounding_box, list of lists, bounding box coordinates obtained by 
        get_bbox_from_mask function
    """
    assert len(bounding_box) == len(array.shape), \
        "Dimensionality of bbox and array do not match. bbox has length " \
        f"{len(bounding_box)} while array has dimension {len(array.shape)}"
    slicer = bounding_box_to_slice(bounding_box)
    return array[slicer]

def get_bbox_from_mask(mask: np.ndarray) -> List[List[int]]:
    """
    Function which gets bounding box around the pixels which are different the 0. 
    It works only for 3D.

    Original comment:
        * this implementation uses less ram than the np.where one and is faster 
        as well IF we expect the bounding box to be close to the image size. 
        If it's not it's likely slower!

    Args:
        * mask, np.ndarray, 1/0 3D array where 1 is foreground (any class)

    Returns:
        * list containing for each axis coordinates of min/max found foreground. 
        
    """
    Z, X, Y = mask.shape
    minzidx, maxzidx, minxidx, maxxidx, minyidx, maxyidx = 0, Z, 0, X, 0, Y
    zidx = list(range(Z))
    for z in zidx:
        if np.any(mask[z]):
            minzidx = z
            break
    for z in zidx[::-1]:
        if np.any(mask[z]):
            maxzidx = z + 1
            break

    xidx = list(range(X))
    for x in xidx:
        if np.any(mask[:, x]):
            minxidx = x
            break
    for x in xidx[::-1]:
        if np.any(mask[:, x]):
            maxxidx = x + 1
            break

    yidx = list(range(Y))
    for y in yidx:
        if np.any(mask[:, :, y]):
            minyidx = y
            break
    for y in yidx[::-1]:
        if np.any(mask[:, :, y]):
            maxyidx = y + 1
            break
    return [[minzidx, maxzidx], [minxidx, maxxidx], [minyidx, maxyidx]]

def get_bbox_from_mask_npwhere(mask: np.ndarray) -> List[List[int]]:
    """
    Function how I would do it... Np where and done.

    Args:
        * mask, np.ndarray, 1/0 3D array where 1 is foreground (any class)

    Returns:
        * list containing for each axis coordinates of min/max found foreground.

    """
    where = np.array(np.where(mask))
    mins = np.min(where, 1)
    maxs = np.max(where, 1) + 1
    return [[i, j] for i, j in zip(mins, maxs)]


def crop_to_nonzero(data: np.ndarray, seg: np.ndarray = None, nonzero_label: int = -1):
    """
    Function which crops volume to its forground by detecting it across all 
    axis and creating the bounding box around it.

    Args:
        * data, np.ndarray, input data to be cropped
        * seg, np.ndarray, segmentation mask which will also be cropped with 
        same bbox as data (optional)
        * nonzero_label, int, label assigned to all places where background 
        segmentation mask is 0 and it is cropped outside.

    Return:
        * cropped_data, data cropped to the usefull context
        * segmentation mask, segmentation mask cropeed to it usefull context
        * b_box, boundary box corrdinates
    """
    # Apply zero-padding
    nonzero_mask = create_nonzero_mask(data)
    bbox = get_bbox_from_mask(nonzero_mask)
    slicer = bounding_box_to_slice(bbox)

    nonzero_mask = nonzero_mask[slicer][None]
    slicer = (slice(None), ) + slicer
    data = data[slicer]

    if seg is not None:
        seg = seg[slicer]
        seg[(seg == 0) & (~nonzero_mask)] = nonzero_label
    else:
        seg = np.where(nonzero_mask, np.int8(0), np.int8(nonzero_label))
    return data, seg, bbox
