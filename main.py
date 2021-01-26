from src.masking_utils import display_multiclass_overlay, separate_mask_regions
from skimage.io import imread


if __name__ == '__main__':
    rgb_image = imread("./images/slide_thumbnail.png")[:, :, :3]
    gray_mask = imread("./images/grayscale_mask.png")
    rgb_mask = imread("./images/rgb_mask.png")

    # Display summaries of the masked regions
    display_multiclass_overlay(rgb_image, gray_mask)
    display_multiclass_overlay(rgb_image, rgb_mask)

    # Separate each slide into distinct regions based on mask.
    extracted_regions_gray = separate_mask_regions(rgb_image, gray_mask, retain_shape=False)
    extracted_regions_rgb = separate_mask_regions(rgb_image, rgb_mask, retain_shape=False)

    # Dictionary of types {value in mask: isolated pixels}
    print(extracted_regions_gray.items())
    print(extracted_regions_rgb.items())
