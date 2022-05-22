"""
Template matching using OpenCV. Basically we need to read text messages in some font or images from the screen to calculate
the reward.
This is not scale-invariant matching. Feature mapping could potentially been implemented in the future.
"""

from src.Helper.configs import TM
import cv2
import numpy as np
import src.Utils.image as image_utils

tm_method = TM.get_method()


def template_matching(base_image, template):
    """
    Check if the template is in the base image.
    """

    gray_base = image_utils.convert_to_grayscale(base_image)
    gray_template = image_utils.convert_to_grayscale(template)

    try:
        res = cv2.matchTemplate(gray_base, gray_template, tm_method)
    except cv2.error as e:
        print(e)
        return 0

    if tm_method == cv2.TM_SQDIFF_NORMED or tm_method == cv2.TM_SQDIFF:
        res = 1 - res

    max_res = np.amax(res)

    # w, h = gray_template.shape[::-1]
    # loc = cv2.minMaxLoc(res)[-1]
    # cv2.rectangle(base_image, loc, (loc[0]+w, loc[1]+h), (0, 255, 255), 2)
    # cv2.imwrite('det.jpg', base_image)

    return max_res
