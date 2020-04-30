"""
Created on Thu Apr 9 2020
@author: malayshah

Class: CSE 5915 - Information Systems
Section: 6pm TR, Spring 2020
Prof: Prof. Jayanti

A Python 3 script for a GUI that allows for image and video inference as well as real-time inferencing utilizing PyQt5

Usage:
    python3 gui.py
"""

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtPrintSupport import *
from PyQt5.QtMultimedia import *
from PyQt5.QtMultimediaWidgets import *
from datetime import datetime
import sys
from PIL import Image
import os
import detection
import numpy as np
from scipy import stats
from cv2 import cv2
import notification
import threading
from object_detection.db import database

VIDEOS = [".mov", ".mp4", ".flv", ".avi", ".ogg", ".wmv"]

class UIMainWindow(QWidget):
    def setupUi(self, MainWindow):
        # Set up MainWindow information
        self.create_main_window(MainWindow)

        # Creates the widget that contains all the other buttons, dropdowns, media, etc.
        self.central_widget = QtWidgets.QWidget(MainWindow)
        self.central_widget.setObjectName("central_widget")

        # Allows for automatic resizing depending on screen size
        self.grid_layout = QtWidgets.QGridLayout(self.central_widget)
        self.grid_layout.setObjectName("grid_layout")

        # Displays the title
        self.display_title()

        # Creates the layout for model and tier customization as well as uploading and capturing
        self.create_detection_layout()

        # Creates the first user instruction - tier selection
        self.create_tier_selection()

        # Populates the models based on what tier is selected
        self.tier_dropdown.currentIndexChanged[str].connect(self.on_tier_current_index_changed)

        # Adjusts the spacing between tier and model selection
        spacer_item = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.detection_info_layout.addItem(spacer_item)

        # Creates the second user instruction - model selection
        self.create_model_selection()

        # Adjusts the spacing between model and upload/capture feature
        spacer_item_one = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.detection_info_layout.addItem(spacer_item_one)

        # Creates the inference media layout
        self.create_inference_media()

        # Displays the upload and capture buttons in the layout
        self.display_upload_button()
        self.display_capture_button()

        # Add the detection layouts to the main central layout
        self.grid_layout.addLayout(self.detection_info_layout, 1, 0, 1, 2)

        # Display the loading gif when a video is inferencing
        self.display_loading_animation()

        # self.model_view = QtWidgets.QGraphicsView(self.central_widget)
        self.model_view = QtWidgets.QWidget(self.central_widget)
        self.model_view.setGeometry(QtCore.QRect(10, 140, 961, 491))
        self.model_view.setStyleSheet("border: 2px solid black; background-color: #e8e9eb")

        # Sets up the media that will display image, video, and webcam
        self.create_media()

        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.model_view.sizePolicy().hasHeightForWidth())
        self.model_view.setSizePolicy(sizePolicy)
        self.model_view.setObjectName("model_view")

        self.model_layout.addWidget(self.model_view, 0, 0, 1, 1)
        self.grid_layout.addLayout(self.model_layout, 2, 0, 1, 2)

        # self.horizontalLayout_media = QtWidgets.QHBoxLayout(
        #     self.central_widget)
        # self.horizontalLayout_media.setSizeConstraint(
        #     QtWidgets.QLayout.SetDefaultConstraint)
        # self.horizontalLayout_media.setObjectName("horizontalLayout_media")
        self.media_label = QtWidgets.QLabel(self)
        # self.horizontalLayout_media.addWidget(self.media_label)

        # Adds logo to the GUI
        self.create_logo()

        # Creates the exit layout
        self.create_exit_layout()

        spacer_item_two = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        self.exit_button_layout.addItem(spacer_item_two)

        spacer_item_three = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.exit_button_layout.addItem(spacer_item_three)
        self.exit_button = QtWidgets.QPushButton(self.central_widget)

        # Displays the stop button
        self.display_stop_button()

        # Displays the exit button
        self.display_exit_button()

        # Add all the features and widgets created to the MainWindow
        MainWindow.setCentralWidget(self.central_widget)

        self.retranslate_ui(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslate_ui(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("TOR", "TOR"))
        MainWindow.showMaximized()
        self.title.setText(_translate("MainWindow", "Tiered Object Recognition"))
        self.step_1_label.setText(_translate("MainWindow", "Step 1: Choose Tier!"))
        self.step_2_label.setText(_translate("MainWindow", "Step 2: Choose Model!"))
        self.step_3_label.setText(_translate("MainWindow", "Step 3: Upload or Capture!"))
        self.upload_button.setText(_translate("MainWindow", "Upload"))
        self.capture_button.setText(_translate("MainWindow", "Capture"))
        self.exit_button.setText(_translate("MainWindow", "Exit"))
        self.stop_button.setText(_translate("MainWindow", "Stop"))

    def on_tier_current_index_changed(self):
        # Change the models to show based on tier selected
        self.model_dropdown.clear()
        if str(self.tier_dropdown.currentText()) == 'Tier 1':
            self.model_dropdown.addItems(['SSD Inception V2 Coco', 'Faster RCNN Inception V2 Coco'])
        elif str(self.tier_dropdown.currentText()) == 'Tier 2':
            self.model_dropdown.addItems(['SSD Inception V2 Coco', 'Faster RCNN Inception V2 Coco'])
        elif str(self.tier_dropdown.currentText()) == 'Tier 3':
            self.model_dropdown.addItems(['SSD Inception V2 Coco', 'Faster RCNN Inception V2 Coco'])
        else:
            self.model_dropdown.addItems(['SSD Inception V2 Coco', 'Faster RCNN Inception V2 Coco'])

    def open_file(self):
        name = QFileDialog.getOpenFileName(self, 'Open File')[0]
        file_extension = os.path.splitext(name)
        tier = self.tier_dropdown.currentText().split(" ")[1]
        # Get path of labelmap and frozen inference graph
        labelmap, inference_graph = self.get_path()

        #self.loading_animation.show()

        if file_extension[1] == ".jpg" or file_extension[1] == ".jpeg":
            # Run inference on image and display
            detection.image_detection(
                inference_graph, labelmap, tier, name, os.path.abspath("predicted.jpg"))
            self.display(os.path.abspath("predicted.jpg"))

        if file_extension[1] in VIDEOS:
            # Run inference on video and display
            #detection.video_detection(inference_graph, labelmap, tier, name, os.path.abspath("predicted.mp4"))
            self.movie.start()

            thread = threading.Thread(target=detection.video_detection, args=(
               inference_graph, labelmap, tier, name, os.path.abspath("predicted.mp4")))
            thread.start()

            while True:
                QtWidgets.qApp.processEvents()
                if thread.isAlive():
                    self.movie.start()
                else:
                    self.movie.stop()
                    #self.movie.disconnect()
                    self.loading_animation.hide()
                    self.loading_animation.clear()
                    self.clear_screen()
                    break

            self.display(os.path.abspath("predicted.mp4"))

    def capture_media(self):
        self.tier = self.tier_dropdown.currentText().split(" ")[1]
        # Get path of labelmap and frozen inference graph
        self.labelmap, self.inference_graph = self.get_path()
        self.tflite = '.tflite' in self.inference_graph

        self.detection_model = detection.load_detection_model(
            self.inference_graph, tflite=self.tflite)
        self.category_index = detection.load_labelmap(self.labelmap)

        # Create lists which handle the notification for detections
        self.seen_classes = []
        self.seen_scores = []
        self.seen_frames = []
        self.current_detection_window = False

        # Start webcam
        self.capture = detection.start_any_webcam()

        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)

        self.media_label.show()

        # Show stop button
        self.stop_button.setVisible(True)
        self.capture_button.setEnabled(False)
        self.upload_button.setEnabled(False)

        self.conn = database.create_connection(database.DATABASE_PATH)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(5)

    def stop_webcam(self):
        self.timer.stop()
        self.capture.release()
        cv2.destroyAllWindows()
        self.stop_button.setVisible(False)
        self.capture_button.setEnabled(True)
        self.upload_button.setEnabled(True)
        self.media_label.hide()

    def exit(self):
        sys.exit()

    def clear_screen(self):
        for i in reversed(range(self.media.count())):
            self.media.itemAt(i).widget().setParent(None)

    def get_path(self):
        # Get path from the tier number
        tier = self.tier_dropdown.currentText().split(" ")[1]
        labelmap = os.path.abspath("../tor_results/tier_{}/labelmap.pbtxt".format(tier))
        inference_graph = ""
        if self.model_dropdown.currentText() == "Faster RCNN Inception V2 Coco":
            inference_graph = os.path.abspath("../tor_results/tier_{}/faster_rcnn_inception_v2_coco_2018_01_28.pb".format(tier))
        elif self.model_dropdown.currentText() == "SSD Inception V2 Coco":
            inference_graph = os.path.abspath("../tor_results/tier_{}/ssd_inception_v2_coco_2018_01_28.tflite".format(tier))
        return labelmap, inference_graph

    def display(self, media=None):
        # Show dialog if File->Open
        if media is False:
            media = QFileDialog.getOpenFileName(self, 'Open File')[0]

        file_extension = os.path.splitext(media)

        if file_extension[1] == ".jpg":
            # Remove all other media
            # self.clear_screen()

            # Display image
            pixmap = QPixmap(media)
            if pixmap.width() > 791 and pixmap.height() > 451:
                pixmap = pixmap.scaledToWidth(960)
                pixmap = pixmap.scaledToWidth(720)
            self.media_label.setPixmap(pixmap)
            self.resize(pixmap.width(), pixmap.height())
            self.media.addWidget(self.media_label)
            self.media.setAlignment(Qt.AlignCenter)
        else:
            # Remove all other media
            # self.clear_screen()

            # Play video
            self.video = QVideoWidget()
            self.video.resize(300, 300)
            self.video.move(0, 0)

            self.player = QMediaPlayer()
            self.player.setVideoOutput(self.video)
            self.player.setMedia(QMediaContent(QUrl.fromLocalFile(media)))
            # to start at the beginning of the video every time
            self.player.setPosition(0)

            self.media.addWidget(self.video)
            self.video.show()
            self.player.play()

    def update_frame(self):
        # Get frame
        _, self.image = self.capture.read()

        # Run inference on frame and display to screen
        classification = detection.detect_on_single_frame(
            self.image, self.category_index, self.detection_model, tflite=self.tflite)
        self.detected_image = classification.Image

        self.current_detection_window, results = detection.update_wake_up_state(
            classification, self.seen_classes, self.seen_scores, self.seen_frames, self.current_detection_window)

        if results:
            best_frame, overall_detected_class, best_score, average_score, detection_time = results

            filename = "{} {} at {}.jpeg".format(
                overall_detected_class, best_score, detection_time).replace(" ", "_")
            cv2.imwrite(filename, best_frame)

            database.insert_webcam_detection(self.conn, os.path.abspath(
                filename), best_score, overall_detected_class, self.tier, self.inference_graph)

            # Send the notification email
            t1 = threading.Thread(target=notification.send_notification_email, args=(
                (filename, overall_detected_class, best_score, average_score, detection_time)))
            t1.start()

        # Display the classified frame to the screen
        self.display_frame(self.detected_image)

    def display_frame(self, frame):
        # Image is stored using 8-bit indexes into a colormap
        qformat = QImage.Format_Indexed8

        # Adjust based on current frame
        if len(frame.shape) == 3:
            if frame.shape[2] == 4:
                # 32-bit RGBA format
                qformat = QImage.Format_RGBA8888
            else:
                # 24-bit RGB format
                qformat = QImage.Format_RGB888

        # convert to QImage
        # Params are: data, width, height, bytesPerLine, format
        outImage = QImage(
            frame, frame.shape[1], frame.shape[0], frame.strides[0], qformat)

        # Conversion
        outImage = outImage.rgbSwapped()

        # Displaying the frame on the screen
        pixmap = QPixmap.fromImage(outImage)
        self.media_label.setPixmap(pixmap)
        self.media_label.setScaledContents(True)
        self.resize(pixmap.width(), pixmap.height())
        self.media.addWidget(self.media_label)
        self.media.setAlignment(Qt.AlignCenter)

    def update_font(self):
        # Set up fonts for the GUI
        font = QtGui.QFont()
        font.setFamily("consolas")
        font.setPointSize(24)
        MainWindow.setFont(font)

    def display_title(self):
        self.title_layout = QtWidgets.QHBoxLayout()
        self.title_layout.setObjectName("title_layout")
        self.title = QtWidgets.QLabel(self.central_widget)
        font = QtGui.QFont()
        font.setFamily("consolas")
        font.setPointSize(40)
        font.setUnderline(False)
        self.title.setFont(font)
        self.title.setAlignment(QtCore.Qt.AlignCenter)
        self.title.setObjectName("title")
        self.title_layout.addWidget(self.title)
        self.grid_layout.addLayout(self.title_layout, 0, 0, 1, 2)

    def create_main_window(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.setMinimumSize(QtCore.QSize(982, 713))
        MainWindow.showMaximized()
        MainWindow.setWindowFlags(Qt.WindowCloseButtonHint | Qt.WindowMinimizeButtonHint)
        MainWindow.setWindowIcon(QtGui.QIcon("images/tor_logo.svg"))

    def create_detection_layout(self):
        # Creates the layout for model and tier customization as well as uploading and capturing
        self.detection_info_layout = QtWidgets.QHBoxLayout()
        self.detection_info_layout.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.detection_info_layout.setObjectName("detection_info_layout")

    def create_tier_selection(self):
        # Creates the instruction for tier selection
        self.create_step_1_label()
        self.detection_info_layout.addWidget(self.step_1_label)

        self.tier_dropdown = QtWidgets.QComboBox(self.central_widget)
        font = QtGui.QFont()
        font.setPointSize(13)
        self.tier_dropdown.setFont(font)
        self.tier_dropdown.setObjectName("tier_dropdown")
        self.tier_dropdown.addItems(["Tier 1", "Tier 2", "Tier 3", "Tier 4"])
        self.detection_info_layout.addWidget(self.tier_dropdown)

    def create_step_1_label(self):
        self.step_1_label = QtWidgets.QLabel(self.central_widget)
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setUnderline(True)
        self.step_1_label.setFont(font)
        self.step_1_label.setObjectName("step_1_label")

    def create_model_selection(self):
        self.create_step_2_label()
        self.detection_info_layout.addWidget(self.step_2_label)

        self.model_dropdown = QtWidgets.QComboBox(self.central_widget)
        font = QtGui.QFont()
        font.setPointSize(13)
        self.model_dropdown.setFont(font)
        self.model_dropdown.setObjectName("model_dropdown")
        self.model_dropdown.addItems(['SSD Inception V2 Coco', 'Faster RCNN Inception V2 Coco'])
        self.detection_info_layout.addWidget(self.model_dropdown)

    def create_step_2_label(self):
        self.step_2_label = QtWidgets.QLabel(self.central_widget)
        font = QtGui.QFont()
        font.setUnderline(True)
        font.setPointSize(14)
        self.step_2_label.setFont(font)
        self.step_2_label.setObjectName("step_2_label")

    def create_inference_media(self):
        self.create_step_3_label()
        self.detection_info_layout.addWidget(self.step_3_label)

    def create_step_3_label(self):
        self.step_3_label = QtWidgets.QLabel(self.central_widget)
        font = QtGui.QFont()
        font.setUnderline(True)
        font.setPointSize(14)
        self.step_3_label.setFont(font)
        self.step_3_label.setObjectName("step_3_label")

        self.model_layout = QtWidgets.QGridLayout()
        self.model_layout.setObjectName("model_layout")

    def display_upload_button(self):
        self.upload_button = QtWidgets.QPushButton(self.central_widget)
        font = QtGui.QFont()
        font.setFamily("consolas")
        font.setPointSize(13)
        self.upload_button.setFont(font)
        self.upload_button.setObjectName("upload_button")
        self.detection_info_layout.addWidget(self.upload_button)
        self.upload_button.clicked.connect(self.open_file)

    def display_capture_button(self):
        self.capture_button = QtWidgets.QPushButton(self.central_widget)
        font = QtGui.QFont()
        font.setFamily("consolas")
        font.setPointSize(13)
        self.capture_button.setFont(font)
        self.capture_button.setObjectName("capture_button")
        self.detection_info_layout.addWidget(self.capture_button)
        self.capture_button.clicked.connect(self.capture_media)

    def display_loading_animation(self):
        self.loading_animation = QtWidgets.QLabel(self)
        self.movie = QMovie(os.path.abspath("images/loading.gif"))
        self.loading_animation.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignHCenter)
        self.loading_animation.setMovie(self.movie)
        self.movie.setCacheMode(QMovie.CacheAll)
        self.model_layout.addWidget(self.loading_animation)

    def create_media(self):
        self.media = QHBoxLayout(self.model_view)
        self.media.setContentsMargins(0, 0, 0, 0)
        self.media.setObjectName("media")

    def create_logo(self):
        self.logo_layout = QtWidgets.QHBoxLayout()
        self.logo_layout.setSizeConstraint(QtWidgets.QLayout.SetFixedSize)
        self.logo_layout.setObjectName("logo_layout")
        self.logo = QtWidgets.QLabel(self.central_widget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.logo.sizePolicy().hasHeightForWidth())

        self.logo.setSizePolicy(sizePolicy)
        self.logo.setMinimumSize(QtCore.QSize(141, 31))
        self.logo.setMaximumSize(QtCore.QSize(141, 31))
        font = QtGui.QFont()
        font.setStrikeOut(True)
        self.logo.setFont(font)
        self.logo.setText("")

        self.logo.setPixmap(QtGui.QPixmap("images/zel_tech_logo_white.png"))
        self.logo.setScaledContents(True)
        self.logo.setAlignment(QtCore.Qt.AlignBottom | QtCore.Qt.AlignLeading | QtCore.Qt.AlignLeft)
        self.logo.setIndent(0)
        self.logo.setObjectName("logo")
        self.logo_layout.addWidget(self.logo)
        self.grid_layout.addLayout(self.logo_layout, 3, 0, 1, 1)

    def create_exit_layout(self):
        self.exit_button_layout = QtWidgets.QHBoxLayout()
        self.exit_button_layout.setSizeConstraint(QtWidgets.QLayout.SetFixedSize)
        self.exit_button_layout.setContentsMargins(0, -1, -1, -1)
        self.exit_button_layout.setObjectName("exit_button_layout")

    def display_stop_button(self):
        self.stop_button = QtWidgets.QPushButton(self.central_widget)
        font = QtGui.QFont()
        font.setFamily("consolas")
        font.setPointSize(13)
        self.stop_button.setFont(font)
        self.stop_button.setObjectName("stop_button")
        self.stop_button.clicked.connect(self.stop_webcam)
        self.stop_button.setVisible(False)

    def display_exit_button(self):
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.exit_button.sizePolicy().hasHeightForWidth())
        self.exit_button.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("consolas")
        font.setPointSize(13)
        self.exit_button.setFont(font)
        self.exit_button.setIconSize(QtCore.QSize(0, 0))
        self.exit_button.setObjectName("exit_button")
        self.exit_button_layout.addWidget(self.stop_button)
        self.exit_button_layout.addWidget(self.exit_button)
        self.grid_layout.addLayout(self.exit_button_layout, 3, 1, 1, 1)
        self.exit_button.clicked.connect(self.exit)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)

    QtGui.QFontDatabase.addApplicationFont('/fonts/Consolas.ttf')
    stylesheet = open("style.qss").read()
    app.setStyleSheet(stylesheet)

    MainWindow = QtWidgets.QMainWindow()
    ui = UIMainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())