import sqlite3
from sqlite3 import Error
from datetime import datetime

def create_connection(db_file):
    """ create a database connection to the SQLite database
        specified by db_file
    :param db_file: database file
    :return: Connection object or None
    """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
    except Error as e:
        print(e)

    return conn


def create_vehicle(conn, vehicle):
    sql = ''' INSERT into Vehicle(img_path, labeled_img_path, confidence, label, tier, loc, time_stamp) 
              VALUES(?,?,?,?,?,?,?) '''
    cur = conn.cursor()
    cur.execute(sql, vehicle)
    return cur.lastrowid

if __name__ == '__main__':
    database = "detection.db"

    # create a database connection
    conn = create_connection(database)
    with conn:
        vehicle = ("capture/test.jpg", "detection/result.jpg", 0.99, "Car", 1, "N.HighSt", datetime.now())
        vehicle_id = create_vehicle(conn, vehicle)
        print(vehicle_id)