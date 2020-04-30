"""
Created on Thu Apr 9 2020
@author: pranaymethuku

Class: CSE 5915 - Information Systems
Section: 6pm TR, Spring 2020
Prof: Prof. Jayanti

A Python 3 script for holding utility functions to update the database

Usage:
    python3 database.py
"""
import sqlite3
import os
import numpy as np
from sqlite3 import Error
from datetime import datetime
import object_detection

DATABASE_PATH = os.path.join(os.path.dirname(__file__), "detection.db")

def create_connection(db_file):
    """ create a database connection to the SQLite database
        specified by db_file
    """
    return sqlite3.connect(db_file)

def __prepare_image_detection_rows(input_image, output_image, inference_graph, tier, classification):
    """
    Helper function to to pass into SQL insert statement
    """
    model = os.path.basename(inference_graph)
    detection_rows = []
    for i in range(len(classification.Classes)):
        datetime_now = datetime.now().strftime("%H:%M:%S on %m-%d-%Y")
        row = (input_image, output_image, float(classification.Scores[i]), classification.Classes[i], tier, model, datetime_now)
        detection_rows.append(row)
    return detection_rows

def __create_detection(conn, detection):
    sql = ''' INSERT into Detection(file_path, labeled_file_path, confidence, label, tier, model, time_stamp) 
              VALUES(?,?,?,?,?,?,?) '''
    cur = conn.cursor()
    cur.execute(sql, detection)
    conn.commit()
    print("Inserted detection {} into database".format(cur.lastrowid))
    return cur.lastrowid

def insert_webcam_detection(conn, output_image, best_score, overall_detected_class, tier, inference_graph):
    model = os.path.basename(inference_graph)
    datetime_now = datetime.now().strftime("%H:%M:%S on %m-%d-%Y")
    detection = (None, output_image, float(best_score), overall_detected_class, tier, model, datetime_now)
    __create_detection(conn, detection)

def insert_image_detection(conn, input_image, output_image, inference_graph, tier, classification):
    detection_rows = __prepare_image_detection_rows(input_image, output_image, inference_graph, tier, classification)
    for detection in detection_rows:
        __create_detection(conn, detection)

def insert_video_detection(conn, input_video, output_video, best_score, overall_detected_class, tier, inference_graph):
    model = os.path.basename(inference_graph)
    datetime_now = datetime.now().strftime("%H:%M:%S on %m-%d-%Y")
    detection = (input_video, output_video, float(best_score), overall_detected_class, tier, model, datetime_now)
    __create_detection(conn, detection)

if __name__ == "__main__":
    # create a database connection
    conn = create_connection("./detection.db")
    cur = conn.cursor()
    cur.execute("SELECT * FROM Detection")

    rows = cur.fetchall()

    for row in rows:
        print(row)