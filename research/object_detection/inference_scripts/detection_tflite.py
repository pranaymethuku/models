import os
import math
from cv2 import cv2
import numpy as np
import sys
import argparse
from PIL import Image
from tensorflow.lite.python.interpreter import Interpreter

# Import utilities
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


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


def load_tflite_interpreter(tflite_graph_path):
    """
    Load the Tensorflow Lite interpreter into memory
    """
    interpreter = Interpreter(model_path=tflite_graph_path)
    interpreter.allocate_tensors()
    return interpreter


def define_tensors(interpreter):
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


def get_input_metadeta(interpreter):
    input_details = interpreter.get_input_details()
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]

    # shape of the image to resize later
    image_shape = (width, height)
    # whether image is a floating model
    floating_model = (input_details[0]['dtype'] == np.float32)
    return image_shape, floating_model


def detect_on_single_frame(image_np, interpreter,
                           image_tensor,
                           output_tensors,
                           image_shape,
                           category_index,
                           floating_model=False,
                           min_score_thresh=0.9,
                           max_boxes_to_draw=1,
                           input_mean=127.5,
                           input_std=127.5):

    # adjust the bounding box size depending on the image size
    height, width = image_np.shape[:2]
    line_thickness_adjustment = math.ceil(max(height, width) / 400)

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

    # Here output the category as string and score to terminal
    # print([category_index.get(i) for i in classes[0]])
    # print(scores)
    return image_np


def batch_detection(tflite_graph, labelmap, input_folder, output_folder):
    # walk through all directories
    for root, _, files in os.walk(input_folder, topdown=False):
        # we only want to walk through the jpgs here, ignore anything else
        img_files = [name for name in files if ".jpg" in name]
        # keep count of how many files we process - batch processing can be slow and
        # we don't want the user to wait without feedback
        total_num_files = len(img_files)
        file_num = 0
        for name in img_files:
            file_num += 1
            original_image = os.path.join(input_folder, name)
            annotated_image = os.path.join(output_folder, name)
            image_detection(tflite_graph, labelmap,
                            original_image, annotated_image)
            print("* Processed " + str(file_num) + "/" + str(total_num_files)
                  + " images in folder \"" + str(root) + "\"")


def image_detection(tflite_graph, labelmap, input_image, output_image):
    category_index = load_labelmap(args.labelmap)
    interpreter = load_tflite_interpreter(args.tflite_graph)

    image_tensor, output_tensors = define_tensors(interpreter)
    image_shape, floating_model = get_input_metadeta(interpreter)

    # Load image using OpenCV and changing color space to RGB
    image = cv2.imread(input_image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    output_image_np = detect_on_single_frame(
        image, interpreter, image_tensor, output_tensors, image_shape, category_index, floating_model=floating_model)

    img = Image.fromarray(output_image_np, 'RGB')
    img.save(output_image, "jpeg")


def video_detection(tflite_graph, labelmap, input_video, output_video, fps=10.0):
    category_index = load_labelmap(args.labelmap)
    interpreter = load_tflite_interpreter(args.tflite_graph)

    image_tensor, output_tensors = define_tensors(interpreter)
    image_shape, floating_model = get_input_metadeta(interpreter)

    # Load video using OpenCV
    cap = cv2.VideoCapture(input_video)
    print("CONVERT_RGB: {}".format(cap.set(cv2.CAP_PROP_CONVERT_RGB, True)))
    print("FPS changed: {}".format(cap.set(cv2.CAP_PROP_FPS, fps)))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, 30.0, (int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Inferencing at a rate of 10 FPS
    frames_to_skip = int(cap.get(cv2.CAP_PROP_FPS) / 10.0)
    frame_count = 0
    # cap.set(cv2.CAP_PROP_POS_FRAMES,40)
    while cap.isOpened():
        _, frame = cap.read()

        # Skipping some frames to run inferencing at 10 fps
        # if frame_count % frames_to_skip == 0 and frame_count != 0:
        if frame is None:
            break

        print("detecting frame {} of {}".format(frame_count, total_frames))
        output_frame = detect_on_single_frame(
            frame, interpreter, image_tensor, output_tensors, image_shape, category_index, floating_model=floating_model)

        out.write(output_frame)
        frame_count += 1

    cap.release()
    out.release()


def webcam_detection(tflite_graph, labelmap):
    category_index = load_labelmap(args.labelmap)
    interpreter = load_tflite_interpreter(args.tflite_graph)

    image_tensor, output_tensors = define_tensors(interpreter)
    image_shape, floating_model = get_input_metadeta(interpreter)

    # Load webcam using OpenCV
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        _, frame = cap.read()
        output_frame = detect_on_single_frame(
            frame, interpreter, image_tensor, output_tensors, image_shape, category_index, floating_model=floating_model)

        cv2.imshow('Video', output_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()

if __name__ == "__main__":
    # Define and parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--tflite_graph', type=str, required=True,
                        help='Path to the .tflite model graph')
    parser.add_argument('-l', '--labelmap', type=str, required=True,
                        help='Path to the labelmap file')
    parser.add_argument("-ii", "--input_image", type=str,
                        help="Path to the input image to detect")
    parser.add_argument("-oi", "--output_image", type=str,
                        help="Path to the output the annotated image, only valid with --input-image")
    parser.add_argument("-if", "--input_folder", type=str,
                        help="Path to the folder of input images to perform detection on")
    parser.add_argument("-of", "--output_folder", type=str,
                        help="Path to the output folder for annotated images, only valid with --input-folder")
    parser.add_argument("-iv", "--input_video", type=str,
                        help="Path to the input video to detect")
    parser.add_argument("-ov", "--output_video", type=str,
                        help="Path to the output the annotated video, only valid with --input-video")
    parser.add_argument("-iw", "--input_webcam", action='store_true',
                        help="Path to the input video to detect")
    args = parser.parse_args()

    if (args.input_image != None):
        image_detection(args.tflite_graph, args.labelmap,
                        args.input_image, args.output_image)
    elif (args.input_folder):
        batch_detection(args.tflite_graph, args.labelmap,
                        args.input_folder, args.output_folder)
    elif (args.input_webcam):
        webcam_detection(args.tflite_graph, args.labelmap)
    else:
        video_detection(args.tflite_graph, args.labelmap,
                        args.input_video, args.output_video)
