# Write Python3 code here
import os
import cv2
import numpy as np
import tensorflow as tf
import sys
import imghdr
from PIL import Image
# import scipy.misc

def main(filename, tier, model):
    # This is needed since the notebook is stored in the object_detection folder.
    sys.path.append("..")
    print(tier)
    # Import utilites
    from object_detection.utils import label_map_util
    from object_detection.utils import visualization_utils as vis_util
    import matplotlib.pyplot as plt

    # Name of the directory containing the object detection module we're using
    if tier == 'Tier X':
        if model == "1":
            print("1")
        elif model == "2":
            print("2")
    # this will expand based on tier and model etc
    
    MODEL_NAME = 'inference_graph' # use tier to get the appropriate model
    IMAGE_NAME = filename
    file_type = imghdr.what(IMAGE_NAME)
    print(file_type)
    #'images_subset/test/5NPD84LF9LH508220-2.jpg'
    #'images_subset/test/7DUDXTWLXBRMFIOGAF4SGD55A4-600.jpg'

    # The path to the image in which the object has to be detected.

    # Grab path to current working directory
    CWD_PATH = os.getcwd()
    print("CWD_PATH:", CWD_PATH)

    # Path to frozen detection graph .pb file, which contains the model that is used
    # for object detection.
    PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, 'frozen_inference_graph.pb')

    # Path to label map file
    PATH_TO_LABELS = os.path.join(CWD_PATH, 'training', 'labelmap.pbtxt')

    # Path to image
    PATH_TO_IMAGE = os.path.join(CWD_PATH, IMAGE_NAME)

    # Number of classes the object detector can identify
    NUM_CLASSES = 2

    print("MODELNAME ", MODEL_NAME)
    print("IMAGENAME ", IMAGE_NAME)
    print("CWDPATH ", CWD_PATH)
    print("PATH_TO_CKPT ", PATH_TO_CKPT)
    print("PATH_TO_LABELS ", PATH_TO_LABELS)
    print("PATH_TO_IMAGE ", PATH_TO_IMAGE)

    # Load the label map.
    # Label maps map indices to category names, so that when our convolution
    # network predicts `5`, we know that this corresponds to `king`.
    # Here we use internal utility functions, but anything that returns a
    # dictionary mapping integers to appropriate string labels would be fine
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    print("LABELMAP ", label_map)
    print("CATEGORIES ", categories)
    print("category_index ", category_index)

    # Load the Tensorflow model into memory.
    detection_graph = tf.Graph()
    print("DETECTION_GRAPH ", detection_graph)
    with detection_graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
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

    # Load image using OpenCV and
    # expand image dimensions to have shape: [1, None, None, 3]
    # i.e. a single-column array, where each item in the column has the pixel RGB value
    image = cv2.imread(PATH_TO_IMAGE)
    print("TYPE OF IMAGE", type(image))
    print("IMAGE", image)
    image_expanded = np.expand_dims(image, axis=0)
    print("IMAGEEXP", image_expanded)

    # Perform the actual detection by running the model with the image as input
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: image_expanded})
    print("BOXES: ", boxes)
    print("SCORES: ", scores)
    print("CLASSES: ", classes)
    print("NUM: ", num)

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

    print(type(image))
    # # All the results have been drawn on the image. Now display the image.
    # cv2.imshow('Image', image)
    #
    # # Press any key to close the image
    # cv2.waitKey(0)
    #
    # # Clean up
    # cv2.destroyAllWindows()

    # im = Image.open(image)
    # im.show()

    img = Image.fromarray(image, 'RGB')
    # img.show()
    img.save("predicted.jpg")

if __name__ == "__main__":
    main()