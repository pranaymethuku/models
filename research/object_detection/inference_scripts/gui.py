# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'design.ui'
#
# Created by: PyQt5 UI code generator 5.14.2
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtPrintSupport import *
from PyQt5.QtMultimedia import *
from PyQt5.QtMultimediaWidgets import *
import sys
from PIL import Image
import os
import detection
import numpy as np
import cv2
from collections import Counter
import statistics

VIDEOS = [".mov", ".mp4", ".flv", ".avi", ".ogg", ".wmv"]


class Ui_MainWindow(QWidget):
    def setupUi(self, MainWindow):
        # Set up MainWindow information
        MainWindow.setObjectName("MainWindow")
        MainWindow.setMinimumSize(QtCore.QSize(982, 713))
        MainWindow.showMaximized()
        MainWindow.setWindowFlags(
            Qt.WindowCloseButtonHint | Qt.WindowMinimizeButtonHint)
        MainWindow.setWindowIcon(QtGui.QIcon("images/tor_logo.svg"))

        self.central_widget = QtWidgets.QWidget(MainWindow)
        self.central_widget.setObjectName("central_widget")

        self.gridLayout_2 = QtWidgets.QGridLayout(self.central_widget)
        self.gridLayout_2.setObjectName("gridLayout_2")

        self.display_title()

        self.detection_info_layout = QtWidgets.QHBoxLayout()
        self.detection_info_layout.setSizeConstraint(
            QtWidgets.QLayout.SetDefaultConstraint)
        self.detection_info_layout.setObjectName("detection_info_layout")

        self.step_1_Label = QtWidgets.QLabel(self.central_widget)
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setUnderline(True)
        self.step_1_Label.setFont(font)
        self.step_1_Label.setObjectName("step_1_Label")
        self.detection_info_layout.addWidget(self.step_1_Label)
        self.tier_dropdown = QtWidgets.QComboBox(self.central_widget)
        font = QtGui.QFont()
        font.setPointSize(13)
        self.tier_dropdown.setFont(font)
        self.tier_dropdown.setObjectName("tier_dropdown")
        self.tier_dropdown.addItems(["Tier 1", "Tier 2", "Tier 3", "Tier 4"])
        self.detection_info_layout.addWidget(self.tier_dropdown)

        # Based on what tier is selected, models will be placed
        self.tier_dropdown.currentIndexChanged[str].connect(
            self.on_tier_currentIndexChanged)

        spacerItem = QtWidgets.QSpacerItem(
            40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.detection_info_layout.addItem(spacerItem)

        self.step_2_label = QtWidgets.QLabel(self.central_widget)
        font = QtGui.QFont()
        font.setUnderline(True)
        font.setPointSize(14)
        self.step_2_label.setFont(font)
        self.step_2_label.setObjectName("step_2_label")
        self.detection_info_layout.addWidget(self.step_2_label)

        self.model_dropdown = QtWidgets.QComboBox(self.central_widget)
        font = QtGui.QFont()
        font.setPointSize(13)
        self.model_dropdown.setFont(font)
        self.model_dropdown.setObjectName("model_dropdown")
        self.model_dropdown.addItems(
            ['SSD Inception V2 Coco', 'Faster RCNN Inception V2 Coco'])

        self.detection_info_layout.addWidget(self.model_dropdown)
        spacerItem1 = QtWidgets.QSpacerItem(
            40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.detection_info_layout.addItem(spacerItem1)

        self.step_3_label = QtWidgets.QLabel(self.central_widget)
        font = QtGui.QFont()
        font.setUnderline(True)
        font.setPointSize(14)
        self.step_3_label.setFont(font)
        self.step_3_label.setObjectName("step_3_label")
        self.detection_info_layout.addWidget(self.step_3_label)

        self.display_upload_button()

        self.capture_button = QtWidgets.QPushButton(self.central_widget)
        font = QtGui.QFont()
        font.setFamily("consolas")
        font.setPointSize(13)
        self.capture_button.setFont(font)
        self.capture_button.setObjectName("capture_button")
        self.capture_button.clicked.connect(self.capture_media)

        self.detection_info_layout.addWidget(self.capture_button)
        self.gridLayout_2.addLayout(self.detection_info_layout, 1, 0, 1, 2)

        self.model_layout = QtWidgets.QGridLayout()
        self.model_layout.setObjectName("model_layout")
        #self.model_view = QtWidgets.QGraphicsView(self.central_widget)
        self.model_view = QWidget(self.central_widget)
        self.model_view.setGeometry(QtCore.QRect(10, 140, 961, 491))
        self.model_view.setStyleSheet(
            "border: 2px solid black; background-color: #e8e9eb")

        self.media = QHBoxLayout(self.model_view)
        self.media.setContentsMargins(0, 0, 0, 0)
        self.media.setObjectName("media")

        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.model_view.sizePolicy().hasHeightForWidth())
        self.model_view.setSizePolicy(sizePolicy)
        # self.model_view.setAlignment(QtCore.Qt.AlignBottom|QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing)
        self.model_view.setObjectName("model_view")

        #sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        # sizePolicy.setHorizontalStretch(0)
        # sizePolicy.setVerticalStretch(0)
        # sizePolicy.setHeightForWidth(self.model_view.sizePolicy().hasHeightForWidth())
        # self.model_view.setSizePolicy(sizePolicy)
        # self.model_view.setAlignment(QtCore.Qt.AlignBottom|QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing)
        # self.model_view.setObjectName("model_view")
        self.model_layout.addWidget(self.model_view, 0, 0, 1, 1)
        self.gridLayout_2.addLayout(self.model_layout, 2, 0, 1, 2)

        self.horizontalLayout_media = QtWidgets.QHBoxLayout(
            self.central_widget)
        self.horizontalLayout_media.setSizeConstraint(
            QtWidgets.QLayout.SetDefaultConstraint)
        self.horizontalLayout_media.setObjectName("horizontalLayout_media")
        self.media_label = QtWidgets.QLabel(self)
        self.horizontalLayout_media.addWidget(self.media_label)

        self.logo_layout = QtWidgets.QHBoxLayout()
        self.logo_layout.setSizeConstraint(QtWidgets.QLayout.SetFixedSize)
        self.logo_layout.setObjectName("logo_layout")
        self.logo = QtWidgets.QLabel(self.central_widget)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.logo.sizePolicy().hasHeightForWidth())

        self.logo.setSizePolicy(sizePolicy)
        self.logo.setMinimumSize(QtCore.QSize(141, 31))
        self.logo.setMaximumSize(QtCore.QSize(141, 31))
        font = QtGui.QFont()
        font.setStrikeOut(True)
        self.logo.setFont(font)
        self.logo.setText("")

        self.logo.setPixmap(QtGui.QPixmap("images/zel_tech_logo_white.png"))
        self.logo.setScaledContents(True)
        self.logo.setAlignment(QtCore.Qt.AlignBottom |
                               QtCore.Qt.AlignLeading | QtCore.Qt.AlignLeft)
        self.logo.setIndent(0)
        self.logo.setObjectName("logo")
        self.logo_layout.addWidget(self.logo)
        self.gridLayout_2.addLayout(self.logo_layout, 3, 0, 1, 1)

        self.exit_button_layout = QtWidgets.QHBoxLayout()
        self.exit_button_layout.setSizeConstraint(
            QtWidgets.QLayout.SetFixedSize)
        self.exit_button_layout.setContentsMargins(0, -1, -1, -1)
        self.exit_button_layout.setObjectName("exit_button_layout")

        spacerItem2 = QtWidgets.QSpacerItem(
            20, 20, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        self.exit_button_layout.addItem(spacerItem2)

        spacerItem3 = QtWidgets.QSpacerItem(
            40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.exit_button_layout.addItem(spacerItem3)
        self.exit_button = QtWidgets.QPushButton(self.central_widget)

        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.exit_button.sizePolicy().hasHeightForWidth())

        self.stop_button = QtWidgets.QPushButton(self.central_widget)
        font = QtGui.QFont()
        font.setFamily("consolas")
        font.setPointSize(13)
        self.stop_button.setFont(font)
        self.stop_button.setObjectName("stop_button")
        self.stop_button.clicked.connect(self.stop_webcam)
        self.stop_button.setVisible(False)

        self.exit_button.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("consolas")
        font.setPointSize(13)
        self.exit_button.setFont(font)
        self.exit_button.setIconSize(QtCore.QSize(0, 0))
        self.exit_button.setObjectName("exit_button")
        self.exit_button_layout.addWidget(self.stop_button)
        self.exit_button_layout.addWidget(self.exit_button)
        self.gridLayout_2.addLayout(self.exit_button_layout, 3, 1, 1, 1)
        self.exit_button.clicked.connect(self.exit)

        MainWindow.setCentralWidget(self.central_widget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("TOR", "TOR"))
        MainWindow.showMaximized()
        self.title.setText(_translate(
            "MainWindow", "Tiered Object Recognition"))
        self.step_1_Label.setText(_translate(
            "MainWindow", "Step 1: Choose Tier!"))
        self.step_2_label.setText(_translate(
            "MainWindow", "Step 2: Choose Model!"))
        self.step_3_label.setText(_translate(
            "MainWindow", "Step 3: Upload or Capture!"))
        self.upload_button.setText(_translate("MainWindow", "Upload"))
        self.capture_button.setText(_translate("MainWindow", "Capture"))
        self.exit_button.setText(_translate("MainWindow", "Exit"))
        self.stop_button.setText(_translate("MainWindow", "Stop"))

    def on_tier_currentIndexChanged(self, index):
        # Change the models to show based on tier selected
        self.model_dropdown.clear()
        if str(self.tier_dropdown.currentText()) == 'Tier 1':
            self.model_dropdown.addItems(
                ['SSD Inception V2 Coco', 'Faster RCNN Inception V2 Coco'])
        elif str(self.tier_dropdown.currentText()) == 'Tier 2':
            self.model_dropdown.addItems(
                ['SSD Inception V2 Coco', 'Faster RCNN Inception V2 Coco'])
        elif str(self.tier_dropdown.currentText()) == 'Tier 3':
            self.model_dropdown.addItems(
                ['SSD Inception V2 Coco', 'Faster RCNN Inception V2 Coco'])
        else:
            self.model_dropdown.addItems(
                ['SSD Inception V2 Coco', 'Faster RCNN Inception V2 Coco'])

    def open_file(self):
        name = QFileDialog.getOpenFileName(self, 'Open File')[0]
        file_extension = os.path.splitext(name)
        tier = self.tier_dropdown.currentText().split(" ")[1]
        # Get path of labelmap and frozen inference graph
        labelmap, frozen_graph = self.get_path()

        if file_extension[1] == ".jpg" or file_extension[1] == ".jpeg":
            # Run inference on image and display
            detection.image_detection(
                frozen_graph, labelmap, tier, name, os.path.abspath("predicted.jpg"))
            self.display(os.path.abspath("predicted.jpg"))

        if file_extension[1] in VIDEOS:
            # Clear area so everything isn't weird
            for i in reversed(range(self.media.count())):
                self.media.itemAt(i).widget().deleteLater()

            # Run inference on video and display
            detection.video_detection(
                frozen_graph, labelmap, tier, name, os.path.abspath("predicted.mp4"))
            self.display(os.path.abspath("predicted.mp4"))

    def capture_media(self):
        # Get path of labelmap and frozen inference graph
        self.labelmap, self.frozen_graph = self.get_path()
        self.tflite = '.tflite' in self.frozen_graph

        self.detection_model = detection.load_detection_model(
            self.frozen_graph, tflite=self.tflite)
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

        for i in reversed(range(self.media.count())):
            self.media.itemAt(i).widget().show()

        # Show stop button
        self.stop_button.setVisible(True)
        self.capture_button.setVisible(False)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(5)

    def stop_webcam(self):
        self.timer.stop()
        self.capture.release()
        cv2.destroyAllWindows()
        self.stop_button.setVisible(False)
        self.capture_button.setVisible(True)
        self.clear_screen()

    def exit(self):
        sys.exit()

    def clear_screen(self):
        for i in reversed(range(self.media.count())):
            self.media.itemAt(i).widget().hide()

    def get_path(self):
        # Get path from the tier number
        tier = self.tier_dropdown.currentText().split(" ")[1]
        labelmap = os.path.abspath(
            "../tor_results/tier_{}/labelmap.pbtxt".format(tier))
        frozen_graph = ""
        if self.model_dropdown.currentText() == "Faster RCNN Inception V2 Coco":
            frozen_graph = os.path.abspath(
                "../tor_results/tier_{}/faster_rcnn_inception_v2_coco_2018_01_28.pb".format(tier))
        elif self.model_dropdown.currentText() == "SSD Inception V2 Coco":
            frozen_graph = os.path.abspath(
                "../tor_results/tier_{}/ssd_inception_v2_coco_2018_01_28.tflite".format(tier))
        return labelmap, frozen_graph

    def display(self, media=None):
        # Show dialog if File->Open
        if media is False:
            media = QFileDialog.getOpenFileName(self, 'Open File')[0]

        file_extension = os.path.splitext(media)

        #width = self.model_layout.geometry().width()
        #height = self.model_layout.geometry().height()

        if file_extension[1] == ".jpg":
            # Remove all other media
            self.clear_screen()

            # Display image
            pixmap = QPixmap(media)
            if pixmap.width() > 791 and pixmap.height() > 451:
                pixmap = pixmap.scaledToWidth(960)
                pixmap = pixmap.scaledToWidth(720)
            #pixmap = pixmap.scaledToWidth(width)
            #pixmap = pixmap.scaledToHeight(height)
            self.media_label.setPixmap(pixmap)
            self.resize(pixmap.width(), pixmap.height())
            self.media.addWidget(self.media_label)
            self.media.setAlignment(Qt.AlignCenter)
        else:
            # Remove all other media
            self.clear_screen()

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
            self.player.play()

    def update_frame(self):
        # Get frame
        _, self.image = self.capture.read()
        self.image = cv2.flip(self.image, 1)

        # Run inference on frame and display to screen
        classification = detection.detect_on_single_frame(
            self.image, self.category_index, self.detection_model, tflite=self.tflite)
        self.detected_image = classification.Image

        # Keep track of all Classes, Scores, and Images seen 
        self.seen_classes.append(classification.Classes)
        self.seen_scores.append(classification.Scores)
        self.seen_frames.append(classification.Image)

        # The grace period that we wait in order to notify 
        gp = 20

        # If we've seen at least 60 consecutive detections, notify somebody about it! 
        if len(self.seen_classes) > gp and not any(c == [] for c in self.seen_classes[-gp:]) and not self.current_detection_window:
            # Create a detection window consisting of only the last 60 detections
            print("Reached 60 consecutive detections!")
            self.current_detection_window = True 
            self.detection_window_classes = self.seen_classes[-gp:]
            self.detection_window_scores = self.seen_scores[-gp:]
            self.detection_window_frames = self.seen_frames[-gp:]
            
            # Flatten the arrays 
            self.detection_window_classes = [item for sublist in self.detection_window_classes for item in sublist]
            self.detection_window_scores = [item for sublist in self.detection_window_scores for item in sublist]

            print(self.detection_window_classes)
            print(self.detection_window_scores)
            
            # Get the most common class 
            most_common_class = Counter(self.detection_window_classes).most_common(1)[0][0]
            print("Most common class: " + str(most_common_class))
            self.detected_class_indices = [i for i,c in enumerate(self.detection_window_classes) if c == most_common_class]
            scores = [self.detection_window_scores[i] for i in self.detected_class_indices]
            print("Average score: " + str(statistics.mean(scores)))

            # img = Image.fromarray(self.detection_window_frames[scores.index(max(scores))])
            # img.save("test.jpg", "jpeg")
            img = self.detection_window_frames[scores.index(max(scores))]
            cv2.imwrite("output.jpg", img)

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
        self.gridLayout_2.addLayout(self.title_layout, 0, 0, 1, 2)

    def display_upload_button(self):
        self.upload_button = QtWidgets.QPushButton(self.central_widget)
        font = QtGui.QFont()
        font.setFamily("consolas")
        font.setPointSize(13)
        self.upload_button.setFont(font)
        self.upload_button.setObjectName("upload_button")
        self.detection_info_layout.addWidget(self.upload_button)
        self.upload_button.clicked.connect(self.open_file)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)

    QtGui.QFontDatabase.addApplicationFont('/fonts/Consolas.ttf')
    stylesheet = open("style.qss").read()
    app.setStyleSheet(stylesheet)

    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
