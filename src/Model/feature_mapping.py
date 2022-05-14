"""
Template matching using OpenCV. Basically we need to read text messages in some font or images from the screen to calculate
the reward.
This is not scale-invariant matching. Feature mapping could potentially been implemented in the future.
"""

from src.Helper.configs import TM
import cv2
import numpy as np
import src.Utils.image as image_utils


def check_single_template(base_image, template):
    """
    Check if the template is in the base image.
    """
    method = TM.get_method()

    gray_base = image_utils.convert_to_grayscale(base_image)
    gray_template = image_utils.convert_to_grayscale(template)

    res = cv2.matchTemplate(gray_base, gray_template, method)

    if method == cv2.TM_SQDIFF_NORMED or method == cv2.TM_SQDIFF:
        res = 1 - res

    max_res = np.amax(res)

    # w, h = gray_template.shape[::-1]
    # loc = cv2.minMaxLoc(res)[-1]
    # cv2.rectangle(base_image, loc, (loc[0]+w, loc[1]+h), (0, 255, 255), 2)
    # cv2.imwrite('det.jpg', base_image)

    if max_res > TM.get_threshold():
        return True, max_res
    else:
        return False, max_res
