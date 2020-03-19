# Write Python3 code here
import os
import cv2
import numpy as np
import tensorflow as tf
import sys
import imghdr
import argparse
from PIL import Image
#import scipy.misc

    
# Import utilites
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
import matplotlib.pyplot as plt

def main():
    # This is needed since the notebook is stored in the object_detection folder.
    sys.path.append("..")

    # set up command line
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source", type=str,
                        help="Path to the source folder to look from")
    parser.add_argument("-t", "--target", type=str,
                        help="Path to target folder to move to, creates/replaces folder named target by default")
    args = parser.parse_args()

    if args.source == None or args.target == None:
        print("ERROR: Source and Target must be specified")
        sys.exit(1)

    if args.source == args.target:
        print("ERROR: Source and Target cannot be same folders")
        sys.exit(1)

    # creating target directory (recursively) if it does not exist
    os.makedirs(args.target, exist_ok=True)

    # Name of the directory containing the object detection module we're using
    MODEL_NAME = 'inference_graph/'

    # The path to the image in which the object has to be detected.

    # Grab path to current working directory
    CWD_PATH = os.getcwd()

    # Path to frozen detection graph .pb file, which contains the model that is used
    # for object detection.
    PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, 'frozen_inference_graph.pb')

    # Path to label map file
    PATH_TO_LABELS = os.path.join(CWD_PATH, 'training', 'labelmap.pbtxt')

    # Number of classes the object detector can identify
    NUM_CLASSES = 2


    # Load the label map.
    # Label maps map indices to category names, so that when our convolution
    # network predicts `5`, we know that this corresponds to `king`.
    # Here we use internal utility functions, but anything that returns a
    # dictionary mapping integers to appropriate string labels would be fine
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)


    # # Load the Tensorflow model into memory.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef()
        with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        sess = tf.compat.v1.Session(graph=detection_graph)

    # Define input and output tensors (i.e. data) for the object detection classifier

    # Input tensor is the image
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    # Output tensors are the detection boxes, scores, and classes
    # Each box represents a part of the image where a particular object was detected
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    # Each score represents level of confidence for each of the objects.
    # The score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

    # Number of objects detected
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    for image_name in os.listdir(args.source):
        # Path to image
        PATH_TO_IMAGE = os.path.join(args.source, image_name)

        # Load image using OpenCV and
        # expand image dimensions to have shape: [1, None, None, 3]
        # i.e. a single-column array, where each item in the column has the pixel RGB value
        image = cv2.imread(PATH_TO_IMAGE)
        image_expanded = np.expand_dims(image, axis=0)

        # Perform the actual detection by running the model with the image as input
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: image_expanded})

        # Draw the results of the detection (aka 'visualize the results')
        vis_util.visualize_boxes_and_labels_on_image_array(
            image,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=8,
            min_score_thresh=0.5)

        img = Image.fromarray(image, 'RGB')
        img.save(os.path.join(args.target, image_name), "jpeg")

if __name__ == "__main__":
    main()