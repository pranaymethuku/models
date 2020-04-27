"""
Created on Thu Apr 9 2020
@author: pranaymethuku

Class: CSE 5915 - Information Systems
Section: 6pm TR, Spring 2020
Prof: Prof. Jayanti

A Python 3 script to perform object detection using Tensorflow on image, video, or webcam, and store/display output accordingly

Usage:
    python3 detection.py [-h] -ig INFERENCE_GRAPH -l LABELMAP
                    [-ii INPUT_IMAGE] [-oi OUTPUT_IMAGE] [-iv INPUT_VIDEO]
                    [-ov OUTPUT_VIDEO] [-iw]

Examples:
    python3 detection.py -ig=inference_graph/inference_graph.pb -ii=./car.jpg -oi=./car_annotated.jpg
    python3 detection.py -ig=inference_graph/inference_graph.tflite -iv=./truck.mp4 -oi=./truck_annotated.mp4
    python3 detection.py -ig=inference_graph/inference_graph.pb -iw
"""

import os
import math
from cv2 import cv2
import numpy as np
import tensorflow as tf
from tensorflow.lite.python.interpreter import Interpreter
import sys
sys.path.append('..')
import argparse
from PIL import Image
import collections
#from object_detection.db import database

# Import utilites
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# Global defaults - adjustable by user in command-line 
MINIMUM_SCORE_THRESHOLD = 0.6 # the minimum confidence for detection 
MAX_BOXES_TO_DRAW = 1 # the maximum number of objects to detect on 

def load_detection_model(inference_graph_path, tflite=True):
    if tflite:
        interpreter = load_tflite_interpreter(inference_graph_path)
        image_tensor, output_tensors = define_tflite_tensors(interpreter)
        image_shape, floating_model = get_tflite_input_metadeta(interpreter)
        return interpreter, image_tensor, output_tensors, image_shape, floating_model
    else:
        sess, detection_graph = load_tf_model(inference_graph_path)
        image_tensor, output_tensors = define_tf_tensors(detection_graph)
        return sess, image_tensor, output_tensors


def load_tf_model(frozen_inference_graph_path):
    """
    Load the Tensorflow model into memory
    """
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef()
        with tf.io.gfile.GFile(frozen_inference_graph_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
        sess = tf.compat.v1.Session(graph=detection_graph)
    return sess, detection_graph


def define_tf_tensors(detection_graph):
    """
    Define input and output tensors (i.e. data) for the object detection classifier
    """
    # Input tensor is the image
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # Output tensors are the detection boxes, scores, and classes
    # Each box represents a part of the image where a particular object was detected
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    # Each score represents level of confidence for each of the objects.
    # The score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name(
        'detection_classes:0')

    # Number of objects detected
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    return image_tensor, [detection_boxes, detection_scores, detection_classes, num_detections]


def load_tflite_interpreter(tflite_graph_path):
    """
    Load the Tensorflow Lite interpreter into memory
    """
    interpreter = Interpreter(model_path=tflite_graph_path)
    interpreter.allocate_tensors()
    return interpreter


def define_tflite_tensors(interpreter):
    """
    Define input and output tensors (i.e. data) for the object detection classifier
    """
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    # Input tensor is the image
    image_tensor = input_details[0]['index']
    # Output tensors are the detection boxes, scores, and classes
    # Each box represents a part of the image where a particular object was detected
    detection_boxes = output_details[0]['index']
    # Each score represents level of confidence for each of the objects.
    # The score is shown on the result image, together with the class label.
    detection_classes = output_details[1]['index']
    detection_scores = output_details[2]['index']
    return image_tensor, [detection_boxes, detection_classes, detection_scores]


def get_tflite_input_metadeta(interpreter):
    input_details = interpreter.get_input_details()
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]

    # shape of the image to resize later
    image_shape = (width, height)
    # whether image is a floating model
    floating_model = (input_details[0]['dtype'] == np.float32)
    return image_shape, floating_model


def load_labelmap(labelmap_path):
    # Number of classes the object detector can identify
    num_classes = len(label_map_util.get_label_map_dict(labelmap_path))

    # Load the label map.
    # Label maps map indices to category names, so that when our convolution
    # network predicts `5`, we know that this corresponds to `king`.
    # Here we use internal utility functions, but anything that returns a
    # dictionary mapping integers to appropriate string labels would be fine
    label_map = label_map_util.load_labelmap(labelmap_path)
    categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes=num_classes, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    return category_index


def visualize_on_single_frame(image_np, boxes, classes, scores, category_index):
    global MINIMUM_SCORE_THRESHOLD
    global MAX_BOXES_TO_DRAW
    # adjust the bounding box size depending on the image size
    height, width = image_np.shape[:2]
    line_thickness_adjustment = math.ceil(max(height, width) / 400)

    # Draw the results of the detection (aka 'visualize the results')
    return vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=4+line_thickness_adjustment,
        min_score_thresh=MINIMUM_SCORE_THRESHOLD,
        max_boxes_to_draw=MAX_BOXES_TO_DRAW)


def detect_on_single_frame(image_np, category_index, detection_model, tflite=True):
    if tflite:
        classification = detect_on_single_frame_tflite(
            image_np, category_index, *detection_model)
    else:
        classification = detect_on_single_frame_tf(
            image_np, category_index, *detection_model)
    return classification


def detect_on_single_frame_tf(image_np, category_index,
                              sess,
                              image_tensor,
                              output_tensors):

    # expand image dimensions to have shape: [1, None, None, 3]
    image_expanded = np.expand_dims(image_np, axis=0)

    # Perform the actual detection by running the model with the image as input
    (boxes, scores, classes, _) = sess.run(
        output_tensors, feed_dict={image_tensor: image_expanded})

    visualize_on_single_frame(image_np, boxes, classes,
                              scores, category_index)

    # Declare a NamedTuple to hold an image, its predicted classes, and the scores associated with each of those classes
    Classification = collections.namedtuple(
        "classification", ["Image", "Classes", "Scores"])
    classification = Classification(image_np, classes, scores)

    # # Here output the best class
    # best_class_id = int(classes[np.argmax(scores)])
    # best_class_name = category_index.get(best_class_id)['name']
    # print("best class: {}".format(best_class_name))
    return classification


def detect_on_single_frame_tflite(image_np,
                                  category_index,
                                  interpreter,
                                  image_tensor,
                                  output_tensors,
                                  image_shape,
                                  floating_model=False,
                                  input_mean=127.5,
                                  input_std=127.5):

    # expand image dimensions to have shape: [1, None, None, 3]
    image_resized = cv2.resize(image_np, image_shape)
    image_expanded = np.expand_dims(image_resized, axis=0)

    # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
    if floating_model:
        image_expanded = (np.float32(image_expanded) - input_mean) / input_std

    # Perform the actual detection by running the model with the image as input
    interpreter.set_tensor(image_tensor, image_expanded)
    interpreter.invoke()

    # Retrieve detection results
    detection_boxes, detection_classes, detection_scores = output_tensors

    # Bounding box coordinates of detected objects
    boxes = interpreter.get_tensor(detection_boxes)[0]
    classes = interpreter.get_tensor(detection_classes)[
        0]  # Class index of detected objects
    scores = interpreter.get_tensor(detection_scores)[
        0]  # Confidence of detected objects

    # Reindex lables to start at 1 (because TFLite requires it that way)
    classes = classes + 1

    visualize_on_single_frame(image_np, boxes, classes, scores, category_index)
    # Declare a NamedTuple to hold an image, its predicted classes, and the scores associated with each of those classes
    Classification = collections.namedtuple(
        "classification", ["Image", "Classes", "Scores"])
    classification = Classification(image_np, classes, scores)

    # Here output the best class
    # best_class_id = int(classes[np.argmax(scores)])
    # best_class_name = category_index.get(best_class_id)['name']
    # print("best class: {}".format(best_class_name))
    return classification


def batch_detection(inference_graph, labelmap, input_folder, output_folder):
    # walk through all directories
    for root, _, files in os.walk(input_folder, topdown=False):
        # we only want to walk through the jpgs here, ignore anything else
        input_files = [
            name for name in files if ".jpg" in name or '.mp4' in name]
        # keep count of how many files we process - batch processing can be slow and
        # we don't want the user to wait without feedback
        total_num_files = len(input_files)
        file_num = 0
        for name in input_files:
            file_num += 1
            original_file = os.path.join(input_folder, name)
            annotated_file = os.path.join(output_folder, name)
            print("Processing {} ({}/{}) in {}".format(name,
                                                       file_num, total_num_files, root))
            if '.jpg' in name:
                image_detection(inference_graph, labelmap,
                                original_file, annotated_file)
            elif '.mp4' in name:
                video_detection(inference_graph, labelmap,
                                original_file, annotated_file)


def image_detection(inference_graph, labelmap, input_image, output_image):
    tflite = '.tflite' in inference_graph
    detection_model = load_detection_model(inference_graph, tflite=tflite)
    category_index = load_labelmap(labelmap)

    # Load image using OpenCV and changing color space to RGB
    image = cv2.imread(input_image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    classification = detect_on_single_frame(
        image, category_index, detection_model, tflite=tflite)

    img = Image.fromarray(classification.Image, 'RGB')
    img.save(output_image, "jpeg")

    #conn = database.create_connection("../db/detection.db")
    #database.insert_image_detection(
    #    conn, input_image, output_image, inference_graph, category_index, classification)


def video_detection(inference_graph, labelmap, input_video, output_video, print_progress=True):
    tflite = '.tflite' in inference_graph
    detection_model = load_detection_model(inference_graph, tflite=tflite)
    category_index = load_labelmap(labelmap)

    # Load video using OpenCV
    cap = cv2.VideoCapture(input_video)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, 30.0, (int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_count = 0
    # cap.set(cv2.CAP_PROP_POS_FRAMES, 420)
    while cap.isOpened():
        _, frame = cap.read()

        if frame is None:
            out.release()
            cap.release()
        else:
            if print_progress:
                print("Detecting on frame {} of {}".format(
                    frame_count, total_frames))

            classification = detect_on_single_frame(
                frame, category_index, detection_model, tflite=tflite)
            out.write(classification.Image)

        frame_count += 1

def start_any_webcam(): 
    # Start webcam - first try the index setting that's supposed to capture input from any device
    index = -1
    capture = cv2.VideoCapture(index)

    # Attempt indices 0 and 1 if -1 doesn't work 
    while (capture is None or not capture.isOpened()) and index < 2:
        print("WARNING: Unable to open video source: " + str(index))
        index = index + 1
        print("Attempting to use video source: " + str(index))
        capture = cv2.VideoCapture(index)

    # If no index works - terminate the program entirely
    # Otherwise - display to the user that a camera worked
    if (capture is None or not capture.isOpened()) and index == 2: 
        print("WARNING: Unable to open video source: " + str(index))
        print("ERROR: Unable to open any camera. Terminating program...")
        sys.exit()
    else: 
        print("SUCCESS! Using video source: " + str(index))

    return capture

def webcam_detection(inference_graph, labelmap, gui=False, frame=None, quit=False, capture=None):
    tflite = '.tflite' in inference_graph
    detection_model = load_detection_model(inference_graph, tflite=tflite)
    category_index = load_labelmap(labelmap)

    if not gui:
        # Load webcam using OpenCV
        cap = start_any_webcam()
    
        while cap.isOpened():
            _, frame = cap.read()
            classification = detect_on_single_frame(
                frame, category_index, detection_model, tflite=tflite)
            cv2.imshow('Video', classification.Image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                break
        cap.release()


if __name__ == "__main__":
    # set up command line
    parser = argparse.ArgumentParser()
    parser.add_argument("-ig", "--inference_graph", type=str, required=True,
                        help="Path to frozen detection graph .pb file, which contains the model that is used")
    parser.add_argument("-l", "--labelmap", type=str, required=True,
                        help="Path to the labelmap")
    parser.add_argument("-mst", "--minimum_score_threshold", type=float, default=MINIMUM_SCORE_THRESHOLD,
                        help="Threshold for the minimum confidence for detection")
    parser.add_argument("-mbd", "--max_boxes_to_draw", type=int, default=MAX_BOXES_TO_DRAW,
                        help="The maximum number of objects to detect")
    # image detection
    parser.add_argument("-ii", "--input_image", type=str,
                        help="Path to the input image to detect")
    parser.add_argument("-oi", "--output_image", type=str,
                        help="Path to the output the annotated image, only valid with --input-image")
    # batch image detection
    parser.add_argument("-if", "--input_folder", type=str,
                        help="Path to the folder of input images to perform detection on")
    parser.add_argument("-of", "--output_folder", type=str,
                        help="Path to the output folder for annotated images, only valid with --input-folder")
    # stored video detection
    parser.add_argument("-iv", "--input_video", type=str,
                        help="Path to the input video to detect")
    parser.add_argument("-ov", "--output_video", type=str,
                        help="Path to the output the annotated video, only valid with --input-video")
    # webcam detection
    parser.add_argument("-iw", "--input_webcam", action='store_true',
                        help="Signals that webcam input is to be used")
    # other potential input and output streams would be configured here
    args = parser.parse_args()

    MINIMUM_SCORE_THRESHOLD = args.minimum_score_threshold
    MAX_BOXES_TO_DRAW = args.max_boxes_to_draw

    if args.input_image:
        image_detection(args.inference_graph, args.labelmap,
                        args.input_image, args.output_image)
    elif args.input_folder:
        batch_detection(args.inference_graph, args.labelmap,
                        args.input_folder, args.output_folder)
    elif args.input_webcam:
        webcam_detection(args.inference_graph, args.labelmap)
    elif args.input_video:
        video_detection(args.inference_graph, args.labelmap,
                        args.input_video, args.output_video)
