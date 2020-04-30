"""
Created on Sun Apr 5 2020
@author: Ruksana Kabealo, Pranay Methuku, Abirami Senthilvelan, Malay Shah

Class: CSE 5915 - Information Systems
Section: 6pm TR, Spring 2020
Prof: Prof. Jayanti

A Python 3 script to perform the following tasks in order:
    1) look at source directory, 
    2) extract xml annotations
    3) save its corresponding compilation into a csv file

Assumptions:
    Annotation files all correspond to .jpg images

Usage:
    python3 xml_to_csv.py --source=path/to/source --csv-file=path/to/csv/file

Examples:
    python3 auto_label.py -s=./tier1/test -c=../tier1/test_labels.csv
"""

import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET
import argparse

def retrieve_df(directory_path):
    """
    helper function to take in a directory
    and compile a DataFrame using them
    """
    xml_list = []
    # iterate through all the xml files in directory
    for xml_file in glob.glob(directory_path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        column_names = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
        # get xml tags corresponding to column_names from file and create a row
        for member in root.findall('object'):
            value = (root.find('filename').text, # filename
                     int(root.find('size')[0].text), # width
                     int(root.find('size')[1].text), # height
                     member[0].text, # class
                     int(member[4][0].text), # xmin
                     int(member[4][1].text), # ymin
                     int(member[4][2].text), # xmax
                     int(member[4][3].text) # ymax
                     )
            xml_list.append(value)
    return pd.DataFrame(xml_list, columns=column_names)

if __name__ == "__main__":

    # set up command line
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source", type=str, default="train",
                        help="Path to the source folder to look from, train folder by default")
    parser.add_argument("-c", "--csv-file", type=str, default="train_labels.csv",
                        help="Path to a CSV file to output the annotations into")
    args = parser.parse_args()

    xml_df = retrieve_df(os.path.join(os.getcwd(), args.source))
    xml_df.to_csv(args.csv_file, index=False)
    print('Successfully converted the annotations in {} to a file {}.'.format(args.source, args.csv_file))
