"""
Modified on Tues March 3 2020
@author: abisen1997

Class: CSE 5915 - Information Systems
Section: 6pm TR, Spring 2020
Prof: Prof. Jayanti

A python script that will rescale the images

Assumptions:
    Script will be replace the image in the directory
    Be careful of the directory you would like to replace

Usage:
    python3 images_to_resize.py --directory --size

Examples:
    python3 images_to_resize.py --directory ../train --size 400 600
"""

# Import dependencies
from PIL import Image
import os
import argparse


def rescale_images(directory, size):
    for img in os.listdir(directory):
        extension = os.path.splitext(img)[1]
        if extension == '.jpg':
            im = Image.open(directory + img)
            rgb_im = im.convert('RGB')
            rgb_im = rgb_im.resize(size)
            rgb_im.save(directory + img)
            print("Successfully rescaled!", img)
        else:
            print("EXTENSION:", extension)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Rescale images")
    parser.add_argument('-d', '--directory', required=True)
    parser.add_argument('-s', '--size', type=int, nargs=2, required=True)
    args = parser.parse_args()
    rescale_images(args.directory, args.size)