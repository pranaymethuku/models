# Write Python3 code here
import os
import cv2
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
        line_thickness=8,
        min_score_thresh=min_score_thresh,
        max_boxes_to_draw=max_boxes_to_draw)

    # Here output the category as string and score to terminal
    # print([category_index.get(i) for i in classes[0]])
    # print(scores)
    return image_np

def batch_detection(frozen_inference_graph, labelmap, input_folder, output_folder):
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
            image_detection(frozen_inference_graph, labelmap, original_image, annotated_image)
            print("* Processed " + str(file_num) + "/" + str(total_num_files) 
                + " images in folder \"" + str(root) + "\"")

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


def video_detection(frozen_inference_graph, labelmap, input_video, output_video):
    print(frozen_inference_graph)
    sess, detection_graph = load_tensorflow_model(frozen_inference_graph)
    category_index = load_labelmap(labelmap)
    image_tensor, output_tensors = define_tensors(detection_graph)

    # Load video using OpenCV
    cap = cv2.VideoCapture(input_video)
    print("CONVERT_RGB: {}".format(cap.set(cv2.CAP_PROP_CONVERT_RGB, True)))
    print("FPS changed: {}".format(cap.set(cv2.CAP_PROP_FPS, 10.0)))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, 30.0, (int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Inferencing at a rate of 10 FPS
    frames_to_skip = int(cap.get(cv2.CAP_PROP_FPS) / 10.0)
    frame_count = 0

    while cap.isOpened():
        _, frame = cap.read()

        # Skipping some frames to run inferencing at 10 fps
        if frame_count % frames_to_skip == 0 and frame_count != 0:
            if frame is None:
                break

            print("detecting frame {} of {}".format(frame_count, total_frames))
            output_frame = detect_on_single_frame(
                frame, sess, image_tensor, output_tensors, category_index)

            out.write(output_frame)
        frame_count += 1

    cap.release()
    out.release()

def webcam_detection(frozen_inference_graph, labelmap):
    sess, detection_graph = load_tensorflow_model(frozen_inference_graph)
    category_index = load_labelmap(labelmap)
    image_tensor, output_tensors = define_tensors(detection_graph)

    # Load video using OpenCV
    cap = cv2.VideoCapture(0)
    # print("CONVERT_RGB: {}".format(cap.set(cv2.CAP_PROP_CONVERT_RGB, True)))
    # print("FPS changed: {}".format(cap.set(cv2.CAP_PROP_FPS, 10.0)))

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
    parser.add_argument("-f", "--frozen-inference-graph", type=str, required=True,
                        help="Path to frozen detection graph (.pb file) which contains the model that is used")
    parser.add_argument("-l", "--labelmap", type=str, required=True,
                        help="Path to the labelmap")
    parser.add_argument("-ii", "--input-image", type=str,
                        help="Path to the input image to perform detection on")
    parser.add_argument("-oi", "--output-image", type=str,
                        help="Path to the output annotated image, only valid with --input-image")
    parser.add_argument("-if", "--input-folder", type=str,
                        help="Path to the folder of input images to perform detection on")
    parser.add_argument("-of", "--output-folder", type=str,
                        help="Path to the output folder for annotated images, only valid with --input-folder")
    parser.add_argument("-iv", "--input-video", type=str,
                        help="Path to the input video to perform detection on")
    parser.add_argument("-ov", "--output-video", type=str,
                        help="Path to the output annotated video, only valid with --input-video")
    parser.add_argument("-iw", "--input-webcam", action='store_true',
                        help="Path to the input stream to perform detection on")                 
    # other potential input and output streams would be configured here
    args = parser.parse_args()

    # parameter check, it's very ugly but I believe it covers all cases
    # if ((args.input_image == None) != (args.output_image == None)) \
    #         or ((args.input_video == None) != (args.output_video == None)) \
    #         or ((args.output_image == None) == (args.output_video == None)) \
    #         or ((args.input_image == None) == (args.input_video == None)):
    #     print("ERROR: only one of image parameters and video parameters can be specified, " +
    #           "but both input and output must be specified")
    #     sys.exit(1)

    if (args.input_image != None):
        image_detection(args.frozen_inference_graph, args.labelmap,
                     args.input_image, args.output_image)
    elif (args.input_folder):
        batch_detection(args.frozen_inference_graph, args.labelmap, args.input_folder, args.output_folder)
    elif (args.input_webcam):
        webcam_detection(args.frozen_inference_graph, args.labelmap)
    else:
        video_detection(args.frozen_inference_graph, args.labelmap,
                     args.input_video, args.output_video)
