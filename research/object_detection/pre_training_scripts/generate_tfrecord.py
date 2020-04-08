"""
This Python script was originally acquired from 
https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html#converting-from-csv-to-record

Updated on Sun Apr 5 2020
@author: pranaymethuku

Class: CSE 5915 - Information Systems
Section: 6pm TR, Spring 2020
Prof: Prof. Jayanti

Updates:
    * Changed the input flag names to be consistent with other scripts we're using.
    * Script now takes in a labelmap .pbtxt file as input.

Usage:
    python generate_tfrecord.py --csv_input=path/to/csv  --record-output=path/to/record --image-dir=path/to/image/dir --labelmap=path/to/labelmap 

Examples
    python generate_tfrecord.py -c=tier_1/train/train_labels.csv -r=tier_1/train/train.tfrecord -i=tier_1/train -l=tier_1/labelmap.pbtxt
"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import io
import pandas as pd
import argparse
import tensorflow as tf

from PIL import Image
from object_detection.utils import dataset_util
from object_detection.utils.label_map_util import get_label_map_dict
from collections import namedtuple, OrderedDict

def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]

def create_tf_example(group, path, labelmap_dict):
    with tf.io.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        print(fid)
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for _, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(labelmap_dict[row['class']])

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


if __name__ == "__main__":

    # set up command line
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--csv-input", type=str, required=True,
                        help="Path to the CSV file to look from")
    parser.add_argument("-r", "--record-output", type=str, required=True,
                        help="Path to a .record file to output the annotations into")
    parser.add_argument("-i", "--image-dir", type=str, required=True,
                        help="The directory with the CSV file's corresponding images")
    parser.add_argument("-l", "--labelmap", type=str, required=True,
                        help="The labelmap corresponding to the classification")
    args = parser.parse_args()

    writer = tf.io.TFRecordWriter(args.record_output)
    path = os.path.join(args.image_dir)
    examples = pd.read_csv(args.csv_input)
    labelmap_dict = get_label_map_dict(args.labelmap)
    grouped = split(examples, 'filename')
    
    for group in grouped:
        print(group)
        tf_example = create_tf_example(group, path, labelmap_dict)
        writer.write(tf_example.SerializeToString())
    writer.close()

    print('Successfully created the TFRecords: {}'.format(args.record_output))