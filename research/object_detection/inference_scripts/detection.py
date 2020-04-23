"""
Created on Thu Apr 9 2020
@author: pranaymethuku

Class: CSE 5915 - Information Systems
Section: 6pm TR, Spring 2020
Prof: Prof. Jayanti

A Python 3 script to perform object detection using Tensorflow on image, video, or webcam, and store/display output accordingly

Usage:
    python3 detection.py [-h] -f FROZEN_INFERENCE_GRAPH -l LABELMAP
                    [-ii INPUT_IMAGE] [-oi OUTPUT_IMAGE] [-iv INPUT_VIDEO]
                    [-ov OUTPUT_VIDEO] [-iw]

Examples:
    python3 detection.py -f=inference_graph/frozen_inference_graph.pb -ii=./car.jpg -oi=./car_annotated.jpg
    python3 detection.py -f=inference_graph/frozen_inference_graph.pb -iv=./truck.mp4 -oi=./truck_annotated.mp4
    python3 detection.py -f=inference_graph/frozen_inference_graph.pb -iw
"""

import os
import math
from cv2 import cv2
import numpy as np
import tensorflow as tf
import sys
import argparse
from PIL import Image

# Import utilites
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


def load_tensorflow_model(frozen_inference_graph_path):
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


def define_tensors(detection_graph):
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

def detect_on_single_frame(image_np, sess,
                           image_tensor,
                           output_tensors,
                           category_index,
                           min_score_thresh=0.9,
                           max_boxes_to_draw=1):

    # adjust the bounding box size depending on the image size
    height, width = image_np.shape[:2]
    line_thickness_adjustment = math.ceil(max(height, width) / 400)

    # expand image dimensions to have shape: [1, None, None, 3]
    image_expanded = np.expand_dims(image_np, axis=0)

    # Perform the actual detection by running the model with the image as input
    (boxes, scores, classes, _) = sess.run(
        output_tensors, feed_dict={image_tensor: image_expanded})

    # Draw the results of the detection (aka 'visualize the results')
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=4+line_thickness_adjustment,
        min_score_thresh=min_score_thresh,
        max_boxes_to_draw=max_boxes_to_draw)

    # # Here output the best class
    # best_class_id = int(classes[np.argmax(scores)])
    # best_class_name = category_index.get(best_class_id)['name']
    # print("best class: {}".format(best_class_name))
    return image_np


def batch_detection(frozen_inference_graph, labelmap, input_folder, output_folder):
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
                image_detection(frozen_inference_graph, labelmap,
                                original_file, annotated_file)
            elif '.mp4' in name:
                video_detection(frozen_inference_graph, labelmap,
                                original_file, annotated_file)


def image_detection(frozen_inference_graph, labelmap, input_image, output_image):
    sess, detection_graph = load_tensorflow_model(frozen_inference_graph)
    category_index = load_labelmap(labelmap)
    image_tensor, output_tensors = define_tensors(detection_graph)

    # Load image using OpenCV and changing color space to RGB
    image = cv2.imread(input_image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    output_image_np = detect_on_single_frame(
        image, sess, image_tensor, output_tensors, category_index)

    img = Image.fromarray(output_image_np, 'RGB')
    img.save(output_image, "jpeg")


def video_detection(frozen_inference_graph, labelmap, input_video, output_video, print_progress=True):
    sess, detection_graph = load_tensorflow_model(frozen_inference_graph)
    category_index = load_labelmap(labelmap)
    image_tensor, output_tensors = define_tensors(detection_graph)

    # Load video using OpenCV
    cap = cv2.VideoCapture(input_video)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, 30.0, (int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Inferencing at a rate of 10 FPS
    #frames_to_skip = int(cap.get(cv2.CAP_PROP_FPS) / 10.0)
    frame_count = 0
    #cap.set(cv2.CAP_PROP_POS_FRAMES,40)
    while cap.isOpened():
        _, frame = cap.read()

        ## Skipping some frames to run inferencing at 10 fps
        #if frame_count % frames_to_skip == 0 and frame_count != 0:
        if frame is None:
            return
        else: 
            if print_progress:
                print("Detecting on frame {} of {}".format(
                    frame_count, total_frames))
            
            output_frame = detect_on_single_frame(
                frame, sess, image_tensor, output_tensors, category_index)
            out.write(output_frame)
        
        frame_count += 1

    cap.release()
    out.release()


def webcam_detection_2(frozen_inference_graph, labelmap, input_image, output_image):
    sess, detection_graph = load_tensorflow_model(frozen_inference_graph)
    category_index = load_labelmap(labelmap)
    image_tensor, output_tensors = define_tensors(detection_graph)

    # Load image using OpenCV and changing color space to RGB
    print(type(input_image))
    image = cv2.imread(input_image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    output_image_np = detect_on_single_frame(
        image, sess, image_tensor, output_tensors, category_index)

    img = Image.fromarray(output_image_np, 'RGB')
    #img.save(output_image, "jpeg")
    return output_image


def webcam_detection(frozen_inference_graph, labelmap):
    sess, detection_graph = load_tensorflow_model(frozen_inference_graph)
    category_index = load_labelmap(labelmap)
    image_tensor, output_tensors = define_tensors(detection_graph)

    # Load webcam using OpenCV
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        _, frame = cap.read()
        output_frame = detect_on_single_frame(
            frame, sess, image_tensor, output_tensors, category_index)

        cv2.imshow('Video', output_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()


if __name__ == "__main__":
    # set up command line
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--frozen_inference_graph", type=str, required=True,
                        help="Path to frozen detection graph .pb file, which contains the model that is used")
    parser.add_argument("-l", "--labelmap", type=str, required=True,
                        help="Path to the labelmap")
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

    if (args.input_image):
        image_detection(args.frozen_inference_graph, args.labelmap,
                        args.input_image, args.output_image)
    elif (args.input_folder):
        batch_detection(args.frozen_inference_graph, args.labelmap,
                        args.input_folder, args.output_folder)
    elif (args.input_webcam):
        webcam_detection(args.frozen_inference_graph, args.labelmap)
    elif (args.input_video):
        video_detection(args.frozen_inference_graph, args.labelmap,
                        args.input_video, args.output_video)
