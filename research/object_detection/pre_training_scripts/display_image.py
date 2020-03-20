"""
Modified on Tues March 3 2020
@author: abisen1997

Class: CSE 5915 - Information Systems
Section: 6pm TR, Spring 2020
Prof: Prof. Jayanti

A python test script that will display an image using cv2

Assumptions:
    Hard-coding the test image for open

Usage:
    python3 display_image.py
"""

# Import dependencies
import cv2
import os
from PIL import Image

CWD_PATH = os.getcwd()
print("CWD_PATH:", CWD_PATH)

IMAGE_NAME = '/Users/abisen/Desktop/test.jpg'
PATH_TO_IMAGE = os.path.join(CWD_PATH, IMAGE_NAME)
print("PATH_TO_IMAGE ", PATH_TO_IMAGE)

image = cv2.imread(IMAGE_NAME)
# cv2.resizeWindow("Image", 800, 800)
# cv2.imshow("Image", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

im = Image.open(IMAGE_NAME)
im.show()