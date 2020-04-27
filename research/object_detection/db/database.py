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

def prepare_detection_rows(input_image, output_image, inference_graph, tier, classification):
    """
    Helper function to to pass into SQL insert statement
    """
    model = os.path.basename(inference_graph)
    detection_rows = []
    for i in range(len(classification.Classes)):
        row = (input_image, output_image, float(classification.Scores[i]), classification.Classes[i], tier, model, datetime.now())
        detection_rows.append(row)
    return detection_rows

def __create_detection(conn, detection):
    sql = ''' INSERT into Detection(file_path, labeled_file_path, confidence, label, tier, model, time_stamp) 
              VALUES(?,?,?,?,?,?,?) '''
    cur = conn.cursor()
    cur.execute(sql, detection)
    conn.commit()
    return cur.lastrowid

def insert_image_detection(conn, input_image, output_image, inference_graph, tier, classification):
    detection_rows = prepare_detection_rows(input_image, output_image, inference_graph, tier, classification)
    for detection in detection_rows:
        __create_detection(conn, detection)

if __name__ == "__main__":
    # create a database connection
    conn = create_connection("./detection.db")
    cur = conn.cursor()
    cur.execute("SELECT * FROM Detection")

    rows = cur.fetchall()

    for row in rows:
        print(row)