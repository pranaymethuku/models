# Import packages
import os
import numpy as np
import tensorflow as tf
import sys
import imageio
import argparse
import cv2

from datetime import datetime
from PIL import Image

'''
Example command to run:

python video_detection2.py -s right_in_front_trimmed.mp4 -f ../tor_models/tier_2/tier_2_faster_rcnn_inception_v2_coco_2018_01_28/inference_graph/frozen_inference_graph.pb -l ../tor_models/tier_2/labelmap.pbtxt
'''

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util
from object_detection.utils import ops as utils_ops

# set up command line
parser = argparse.ArgumentParser()
parser.add_argument("-s", "--source", type=str,
                    help="Path of the input video")
parser.add_argument("-f", "--frozen_inference_graph", type=str,
                    help="Path to frozen detection graph .pb file, which contains the model that is used")
parser.add_argument("-l", "--labelmap", type=str,
                    help="Path to the labelmap")
args = parser.parse_args()

# Name of the directory containing the object detection module we're using
MODEL_NAME = 'inference_graph'
VIDEO_NAME = 'right_in_front.mp4'

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = args.frozen_inference_graph

# Path to label map file
PATH_TO_LABELS = args.labelmap

# Path to video
PATH_TO_VIDEO = args.source

# Number of classes the object detector can identify
NUM_CLASSES = len(label_map_util.get_label_map_dict(args.labelmap))

# Load the label map.
# Label maps map indices to category names, so that when our convolution
# network predicts `5`, we know that this corresponds to `king`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


cap = cv2.VideoCapture('right_in_front_trimmed.mp4')

out = cv2.VideoWriter('test.mp4', cv2.VideoWriter_fourcc(
        'M', 'J', 'P', 'G'), 10.0, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)

# Define input and output tensors (i.e. data) for the object detection classifier

with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:

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

        frames_to_skip = int(cap.get(cv2.CAP_PROP_FPS) / 10.0)
        print(frames_to_skip)
        num_frames = 0
        # cap.set(cv2.CAP_PROP_POS_FRAMES, 515)
        time = datetime.now()
        while(cap.isOpened()):
            ret, frame = cap.read()

            if num_frames % frames_to_skip == 0 and num_frames != 0:
                print("doing frame", repr(num_frames))
                if frame is None:
                    break
                
                color_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image_np_expanded = np.expand_dims(color_frame, axis=0)

                # actual inference
                (boxes, scores, classes, num) = sess.run(
                        [detection_boxes, detection_scores,
                            detection_classes, num_detections],
                        feed_dict={image_tensor: image_np_expanded})

                # drawing boxes n stuf
                vis_util.visualize_boxes_and_labels_on_image_array(
                        color_frame,
                        np.squeeze(boxes),
                        np.squeeze(classes).astype(np.int32),
                        np.squeeze(scores),
                        category_index,
                        use_normalized_coordinates=True,
                        line_thickness=8,
                        min_score_thresh=.20)

                # show frame and save to new output vid
                frame2 = cv2.resize(color_frame, (800, 600))
                cv2.imshow('frame', frame2)
                output_rgb = cv2.cvtColor(color_frame, cv2.COLOR_RGB2BGR)
                out.write(output_rgb)
                # print(num_frames)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                print("Skipped frame", repr(num_frames))
            num_frames += 1




        out.release()
        cap.release()
        cv2.destroyAllWindows()
        finished = datetime.now()
        print("Took", (finished-time).total_seconds())
        # input_video = 'right_in_front_trimmed'
        # video_reader = imageio.get_reader('%s.mp4'%input_video)
        # video_writer = imageio.get_writer('%s_annotated.mp4'% input_video, fps=10)

        # t0 = datetime.now()
        # n_frames = 0
        
        # for frame in video_reader:
        #     image_np = frame
        #     n_frames += 1

        #     image_np_expanded = np.expand_dims(image_np,axis=0)

        #     (boxes,scores,classes,num)=sess.run(
        #         [detection_boxes,detection_scores,detection_classes,num_detections],
        #         feed_dict={image_tensor: image_np_expanded})

        #     vis_util.visualize_boxes_and_labels_on_image_array(
        #         frame,
        #         np.squeeze(boxes),
        #         np.squeeze(classes).astype(np.int32),
        #         np.squeeze(scores),
        #         category_index,
        #         use_normalized_coordinates=True,
        #         line_thickness=8,
        #         min_score_thresh=0.60)

        #     video_writer.append_data(image_np)
        #     print(n_frames)
        # fps=n_frames/(datetime.now()-t0).total_seconds()
        # print("Frames processed: %s,Speed:%s fps"%(n_frames,fps))

        # video_writer.close()