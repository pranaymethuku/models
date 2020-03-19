import os
from os import path
import glob
import pandas as pd
import xml.etree.ElementTree as ET
import pathlib
import shutil


def xml_to_csv(path):
    xml_list = []
    print(path)
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)

    names = []
    for i in xml_df['filename']:
        extension = os.path.splitext(i)[1]
        if extension == '.png' or extension == '.jpeg':
            new_img = i.replace(extension, ".jpg")
            print("New_Image: ", new_img)
            i = new_img
        names.append(i)
    xml_df['filename'] = names
    return xml_df


def main():
    for folder in ['train', 'test']:
        #os.chdir("../images_subset/")
        image_path = os.path.join(os.getcwd(), (folder))
        xml_df = xml_to_csv(image_path)
        csv_file = image_path + '_labels.csv'
        file = pathlib.Path(csv_file)
        file = open(csv_file, "w+")
        xml_df.to_csv(csv_file, index=None)
        print('Successfully converted xml to csv.')
        file.close()


main()
