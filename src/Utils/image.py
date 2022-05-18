"""
Providing functionalities for image processing, e.g, image resizing, image loading, image saving, etc.
"""
import cv2
import pytesseract
from PIL import Image
import os
import pyautogui
import numpy as np

"""check if using Windows """
if os.name == 'nt':
    pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'


def scale_image(image, resolution=(512, 512)):
    image = cv2.resize(image, resolution, interpolation=cv2.INTER_AREA)
    return image


def save_image(image, path):
    try:
        cv2.imwrite(path, image)
    except Exception as e:
        print(e)


def image_normalise(image):
    i = np.asarray(image, np.float32)
    return i / 255


def convert_color_to_rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def invert_color(image):
    image = (255 - image)
    return image


def convert_to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def extract_text(image):
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        invert = invert_color(gray)
        #cv2.imshow('Gray image', invert)
        text = pytesseract.image_to_string(invert)
        return text
    except Exception as e:
        return ''


def load_image(filename):
    try:
        image = cv2.imread('../../var/features/' + filename)
        return image
    except Exception as e:
        print(e)
