import numpy as np
import matplotlib.pyplot as plt
from warnings import warn


def display_overlay(rgb_image: np.ndarray, binary_mask: np.ndarray, ax: plt.axis, check_mask=False):
    """
    Display version of RGB_image where `False` values in binary_mask are darkened.
    :param rgb_image:
    :param binary_mask:
    :param ax:
    :param check_mask:
    :return:
    """
    if check_mask and binary_mask.dtype is not bool:
        warn("`binary_mask` is non-boolean, attempting conversion.")
        binary_mask = binary_mask.astype(bool)

    overlay = rgb_image.copy()
    overlay[~binary_mask] = overlay[~binary_mask] // 2  # Darken the color of the original image where mask is False
    ax.imshow(overlay)


def display_multiclass_overlay(rgb_image: np.ndarray, mask: np.ndarray):
    """
    Given multiclass mask, display all isolated regions in mask.
    :param rgb_image:
    :param mask:
    :return:
    """
    # If there are two dimensions, treat as 2D image
    if len(mask.shape) == 2:
        assert mask.shape[0] == rgb_image.shape[0] and mask.shape[1] == rgb_image.shape[1]
        dimensions = 2
    # If there are three dimensions, it's okay if the mask has 3 or 4 channels (RGB or RGBA)
    elif len(mask.shape) == 3:
        dimensions = 3
    else:
        print("Error: mask has too many dimensions.")
        raise ValueError

    if dimensions == 2:
        mask_values = np.unique(mask)
        fig, ax = plt.subplots(1, len(mask_values) + 1)
        ax[0].imshow(rgb_image)
        for index, val in enumerate(mask_values):
            display_overlay(rgb_image, mask == val, ax[index + 1])
    elif dimensions == 3:
        mask_values = np.vstack(sorted({tuple(r) for r in mask.reshape(-1, 3)}))
        fig, ax = plt.subplots(1, len(mask_values) + 1)
        ax[0].imshow(rgb_image)
        for index, val in enumerate(mask_values):
            display_overlay(rgb_image, np.all(mask == val, axis=-1), ax[index + 1])

    plt.show()


def get_pixels_under_mask(rgb_image: np.ndarray, mask: np.ndarray, value: np.uint8, retain_shape: bool):
    """
    Given an rgb image, a mask, and a mask value of interest, return a flattened list of pixels underneath where
    the mask equals `value`.
    :param rgb_image:
    :param mask:
    :param value:
    :param retain_shape:
    :return:
    """
    if len(mask.shape) == 2:
        if retain_shape:
            masked_rgb = rgb_image.copy()
            masked_rgb[~(mask == value)] = [255, 255, 255]
            return masked_rgb
        else:
            return rgb_image[mask == value]
    elif len(mask.shape) == 3:
        if retain_shape:
            masked_rgb = rgb_image.copy()
            masked_rgb[~np.all(mask == value, axis=-1)] = [255, 255, 255]
            return masked_rgb
        else:
            return rgb_image[np.all(mask == value, axis=-1)]


def separate_mask_regions(rgb_image: np.ndarray, mask: np.ndarray, retain_shape=True):
    """
    Given multiclass mask, return all isolated regions in dictionary.
    :param rgb_image: RGB image that mask corresponds to.
    :param mask: Multiclass mask.
    :param retain_shape: Set to `True` to fill values outside of mask region with white.
    :return: Dictionary where keys are values in mask, dict values are lines of pixels/shapes
    """
    if len(mask.shape) == 2:
        assert mask.shape[0] == rgb_image.shape[0] and mask.shape[1] == rgb_image.shape[1]
        dimensions = 2
    # If there are three dimensions, it's okay if the mask has 3 or 4 channels (RGB or RGBA)
    elif len(mask.shape) == 3:
        dimensions = 3
    else:
        print("Error: mask has too many dimensions.")
        raise ValueError

    value_mask_dictionary = {}
    mask_values = None

    if dimensions == 2:
        mask_values = np.unique(mask)
    elif dimensions == 3:
        mask_values = np.vstack(sorted({tuple(r) for r in mask.reshape(-1, 3)}))

    for index, val in enumerate(mask_values):
        if isinstance(val, np.ndarray):
            key_val = tuple(v for v in val)
        else:
            key_val = val
        value_mask_dictionary[key_val] = get_pixels_under_mask(rgb_image, mask, val, retain_shape=retain_shape)

    return value_mask_dictionary
