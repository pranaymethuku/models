import sqlite3
import os
import numpy as np
from sqlite3 import Error
from datetime import datetime

def create_connection(db_file):
    """ create a database connection to the SQLite database
        specified by db_file
    """
    return sqlite3.connect(db_file)

def prepare_detection_row(input_image, output_image, inference_graph, category_index, classification):
    """
    Helper function to to pass into SQL insert statement
    """
    model = os.path.basename(inference_graph)
    # retrieve the label and confidence score from classification
    confidence = float(np.max(classification.Scores))
    label_id = int(classification.Classes[np.argmax(classification.Scores)])
    label = category_index.get(label_id)['name']
    # NOTE: not sure how to get the tier in the best way possible yet, just hardcoding some value for now
    tier = 2
    return (input_image, output_image, confidence, label, tier, model, datetime.now())

def __create_detection(conn, detection):
    sql = ''' INSERT into Detection(img_path, labeled_img_path, confidence, label, tier, model, time_stamp) 
              VALUES(?,?,?,?,?,?,?) '''
    cur = conn.cursor()
    cur.execute(sql, detection)
    conn.commit()
    return cur.lastrowid

def insert_image_detection(conn, input_image, output_image, inference_graph, category_index, classification):
    detection_row = prepare_detection_row(input_image, output_image, inference_graph, category_index, classification)
    return __create_detection(conn, detection_row)

if __name__ == "__main__":
    # create a database connection
    conn = create_connection("./detection.db")
    detection = ("capture/test.jpg", "detection/result.jpg", 0.99, "Car", 1, "SSD Inception V2 Coco", datetime.now())
    detection_id = __create_detection(conn, detection)
    print(detection_id)