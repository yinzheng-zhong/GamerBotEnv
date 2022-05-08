"""
Providing functionalities for image processing, e.g, image resizing, image loading, image saving, etc.
"""
import cv2
import pytesseract
from PIL import Image


def scale_image(image, resolution=(512, 512)):
    image = cv2.resize(image, resolution, interpolation=cv2.INTER_AREA)
    return image


def save_image(image, path):
    try:
        cv2.imwrite(path, image)
    except Exception as e:
        print(e)


def extract_text(image):
    #image = Image.fromarray(image)

    if not image:
        return ''
    else:
        text = pytesseract.image_to_string(image)
        return text
