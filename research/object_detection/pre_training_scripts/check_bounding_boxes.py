"""
NOTE: This Python script uses some code from Stack Overflow (I lost the original link)

Updated on Tue Apr 8 2020
@author: pranaymethuku

Class: CSE 5915 - Information Systems
Section: 6pm TR, Spring 2020
Prof: Prof. Jayanti

Updates:
    * Incorporated argparse to be consistent with other scripts we're using

Usage:
    python3 check_bounding_boxes.py [-h] -s SOURCE -c CSV_FILE

Examples:
    python3 check_bounding_boxes.py -s=./tier1/test -c=../tier1/test_labels.csv
"""

import csv
import cv2 
import os
import argparse
import numpy as np

if __name__ == "__main__":

    # set up command line
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source", type=str, required=True,
                        help="Path to the source folder to look from")
    parser.add_argument("-c", "--csv_file", type=str, required=True,
                        help="Path to a CSV file to inspect")
    args = parser.parse_args()

    fid = open(args.csv_file, 'r')
    
    csv_file = csv.reader(fid, delimiter=',')
    first = True
    count = 0
    error_count = 0
    error = False
    for row in csv_file:        
        # extract features
        name, width, height, xmin, ymin, xmax, ymax = row[0], int(row[1]), int(row[2]), int(row[4]), int(row[5]), int(row[6]), int(row[7])
        
        # import actual image to compare
        path = os.path.join(args.source, name)
        img = cv2.imread(path)        
        if type(img) == type(None):
            error = True
            print('Could not read image at', path)
            continue
        
        og_height, og_width = img.shape[:2]
        
        if og_width != width:
            error = True
            print('Width mismatch for image: ', name, width, '!=', og_width)
        
        if og_height != height:
            error = True
            print('Height mismatch for image: ', name, height, '!=', og_height)
        
        if xmin > og_width:
            error = True
            print('XMIN > og_width for file', name)
            
        if xmax > og_width:
            error = True
            print('XMAX > og_width for file', name)
        
        if ymin > og_height:
            error = True
            print('YMIN > og_height for file', name)
        
        if ymax > og_height:
            error = True
            print('YMAX > og_height for file', name)
        
        if error:
            print('Error for file: {}\n'.format(name))
            error_count += 1
            error = False
        count += 1
    fid.close()
    print('Checked %d files and realized %d errors' % (count, error_count))
