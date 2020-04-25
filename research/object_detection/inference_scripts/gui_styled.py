# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file
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
import os
import detection
import cv2
import numpy as np
import tensorflow as tf
from object_detection.utils import label_map_util

from object_detection.utils import visualization_utils as vis_util
from PIL import Image

VIDEOS = [".mov", ".mp4", ".flv", ".avi", ".ogg", ".wmv"]

class Ui_MainWindow(QWidget):
    def setupUi(self, MainWindow):
        # Initial setup of the GUI
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(978, 748)
        MainWindow.setWindowFlags(Qt.WindowCloseButtonHint | Qt.WindowMinimizeButtonHint)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(24)
        MainWindow.setFont(font)
        MainWindow.setWindowIcon(QtGui.QIcon("images/tor_logo.svg"))

        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(90, 10, 841, 41))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")

        # Creates and styles the title "Tiered Object Recognition
        font.setPointSize(36)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.formLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.formLayoutWidget.setGeometry(QtCore.QRect(10, 50, 961, 71))
        self.formLayoutWidget.setObjectName("formLayoutWidget")

        # Set-up for the steps (right below the title)
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.formLayoutWidget)
        self.horizontalLayout.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")

        # Creates and styles step 1 title
        self.step1Label = QtWidgets.QLabel(self.formLayoutWidget)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        font.setUnderline(True)
        self.step1Label.setFont(font)
        self.step1Label.setObjectName("step1Label")
        self.horizontalLayout.addWidget(self.step1Label)

        # Creates the drop-down menu for the tiers
        self.step1ChooseTierComboBox = QtWidgets.QComboBox(self.formLayoutWidget)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.step1ChooseTierComboBox.setFont(font)
        self.step1ChooseTierComboBox.setObjectName("step1ChooseTierComboBox")
        self.step1ChooseTierComboBox.addItem("Tier 1")
        self.step1ChooseTierComboBox.addItem("Tier 2")
        self.step1ChooseTierComboBox.addItem("Tier 3")
        self.step1ChooseTierComboBox.addItem("Tier 4")
        self.horizontalLayout.addWidget(self.step1ChooseTierComboBox)

        # Based on what tier is selected, models will be placed
        self.step1ChooseTierComboBox.currentIndexChanged[str].connect(self.on_tier_currentIndexChanged)

        # Add spacers for GUI to look cleaner
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)

        # Creates and styles the step 2 title
        self.step2Label = QtWidgets.QLabel(self.formLayoutWidget)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        font.setUnderline(True)
        self.step2Label.setFont(font)
        self.step2Label.setObjectName("step2Label")
        self.horizontalLayout.addWidget(self.step2Label)

        # Creates the drop-down menu for the models
        self.step2ChooseModelComboBox = QtWidgets.QComboBox(self.formLayoutWidget)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.step2ChooseModelComboBox.setFont(font)
        self.step2ChooseModelComboBox.setObjectName("step2ChooseModelComboBox")
        self.step2ChooseModelComboBox.addItem("")
        self.step2ChooseModelComboBox.addItem("")
        self.step2ChooseModelComboBox.addItem("")
        self.step2ChooseModelComboBox.addItem("")
        self.horizontalLayout.addWidget(self.step2ChooseModelComboBox)

        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem1)

        # Creates and styles the step 3 title
        self.step3UploadImageOrVideoLabel = QtWidgets.QLabel(self.formLayoutWidget)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        font.setUnderline(True)
        self.step3UploadImageOrVideoLabel.setFont(font)
        self.step3UploadImageOrVideoLabel.setObjectName("step3UploadImageOrVideoLabel")
        self.horizontalLayout.addWidget(self.step3UploadImageOrVideoLabel)

        # Creates the Upload button
        self.pushButton_2 = QtWidgets.QPushButton(self.formLayoutWidget)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.pushButton_2.setFont(font)
        self.pushButton_2.setObjectName("pushButton_2")
        self.horizontalLayout.addWidget(self.pushButton_2)
        #self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        #self.pushButton.setGeometry(QtCore.QRect(870, 110, 101, 51))
        self.pushButton_2.clicked.connect(self.open_file)

        # Creates the Capture button
        self.pushButton_3 = QtWidgets.QPushButton(self.formLayoutWidget)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.pushButton_3.setFont(font)
        self.pushButton_3.setObjectName("pushButton_3")
        self.horizontalLayout.addWidget(self.pushButton_3)
        self.pushButton_3.clicked.connect(self.capture_media)

        # font = QtGui.QFont()
        # font.setPointSize(14)
        # self.pushButton.setFont(font)
        # self.pushButton.setObjectName("pushButton")

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 978, 22))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        # Displays media
        self.media_layout = QWidget(self.centralwidget)
        self.media_layout.setGeometry(QtCore.QRect(10, 140, 961, 491))
        self.media_layout.setObjectName("graphicsView")
        self.media_layout.setStyleSheet("border: 2px solid black")

        self.stop = QtWidgets.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.stop.setFont(font)
        self.stop.setGeometry(QtCore.QRect(740, 640, 113, 32))
        self.stop.setObjectName("stop")
        MainWindow.setCentralWidget(self.centralwidget)
        self.stop.clicked.connect(self.stop_webcam)
        self.stop.setVisible(False)

        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.pushButton.setFont(font)
        self.pushButton.setGeometry(QtCore.QRect(860, 640, 113, 32))
        self.pushButton.setObjectName("pushButton")
        MainWindow.setCentralWidget(self.centralwidget)
        self.pushButton.clicked.connect(self.exit)

        # Creates the layout
        self.media = QHBoxLayout(self.media_layout)
        self.media.setContentsMargins(0, 0, 0, 0)
        self.media.setObjectName("media")

        self.media_label = QLabel(self)
        MainWindow.setCentralWidget(self.centralwidget)

        # Adds logo to GUI
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(20, 640, 151, 31))
        self.label_2.setText("")
        self.label_2.setPixmap(QtGui.QPixmap("images/zel_tech_logo.png"))
        self.label_2.setScaledContents(True)
        self.label_2.setObjectName("label_2")
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)


    def stop_webcam(self):
        self.timer.stop()
        self.stop.setVisible(False)
        self.clear_screen()
        self.capture.release()
        cv2.destroyAllWindows()

    def exit(self):
        sys.exit()

    def clear_screen(self):
        for i in reversed(range(self.media.count())):
            self.media.itemAt(i).widget().setParent(None)

    def open_file(self):
        name = QFileDialog.getOpenFileName(self, 'Open File')[0]
        file_extension = os.path.splitext(name)

        # Get path of labelmap and frozen inference graph
        labelmap, frozen_graph = self.get_path()

        if file_extension[1] == ".jpg" or file_extension[1] == ".jpeg":
            # Run inference on image and display
            detection.image_detection(frozen_graph, labelmap, name, "predicted.jpg")
            self.display("predicted.jpg")

        if file_extension[1] in VIDEOS:
            # Clear area so everything isn't weird
            for i in reversed(range(self.media.count())):
                self.media.itemAt(i).widget().deleteLater()

            # Run inference on video and display
            detection.video_detection(frozen_graph, labelmap, name, os.getcwd() + os.path.sep + "predicted.mp4")
            self.display(os.getcwd() + os.path.sep + "predicted.mp4")

    def get_path(self):
        # Get path from the tier number
        tier = self.step1ChooseTierComboBox.currentText().split(" ")[1]
        labelmap = "../tor_results/tier_" + tier + "/labelmap.pbtxt"
        frozen_graph = ""
        if self.step2ChooseModelComboBox.currentText() == "Faster RCNN Inception V2 Coco":
            frozen_graph = "../tor_results/tier_" + tier + "/faster_rcnn_inception_v2_coco_2018_01_28.pb"
        elif self.step2ChooseModelComboBox.currentText() == "Faster RCNN Resnet101 Kitti":
            frozen_graph = "../tor_results/tier_" + tier + "/_faster_rcnn_resnet101_kitti_2018_01_28.pb"
        elif self.step2ChooseModelComboBox.currentText() == "RFCN Resnet101 Coco":
            frozen_graph = "../tor_results/tier_" + tier + "/rfcn_resnet101_coco_2018_01_28.pb"
        elif self.step2ChooseModelComboBox.currentText() == "SSD Inception V2 Coco":
            frozen_graph = "../tor_results/tier_" + tier + "/ssd_inception_v2_coco_2018_01_28.tflite"
        elif self.step2ChooseModelComboBox.currentText() == "SSD V2 Coco":
            frozen_graph = "../tor_results/tier_" + tier + "/ssd_v2_coco_2018_01_28.tflite"
        return labelmap, frozen_graph

    def display(self, media=None):
        # Show dialog if File->Open
        if media is False:
            media = QFileDialog.getOpenFileName(self, 'Open File')[0]

        file_extension = os.path.splitext(media)

        if file_extension[1] == ".jpg":
            # Remove all other media
            self.clear_screen()

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
            self.clear_screen()

            # Play video
            self.video = QVideoWidget()
            self.video.resize(300, 300)
            self.video.move(0, 0)

            self.player = QMediaPlayer()
            self.player.setVideoOutput(self.video)
            self.player.setMedia(QMediaContent(QUrl.fromLocalFile(media)))
            self.player.setPosition(0)  # to start at the beginning of the video every time

            self.media.addWidget(self.video)
            self.player.play()

    def capture_media(self):
        # Get path of labelmap and frozen inference graph
        self.labelmap, self.frozen_graph = self.get_path()
        self.capture=cv2.VideoCapture(0)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)

        self.stop.setVisible(True)

        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            self.od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.frozen_graph, 'rb') as fid:
                self.serialized_graph = fid.read()
                self.od_graph_def.ParseFromString(self.serialized_graph)
                tf.import_graph_def(self.od_graph_def, name='')

        NUM_CLASSES = len(label_map_util.get_label_map_dict(self.labelmap))

        self.label_map = label_map_util.load_labelmap(self.labelmap)
        self.categories = label_map_util.convert_label_map_to_categories(self.label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
        self.category_index = label_map_util.create_category_index(self.categories)

        self.timer=QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(5)

    def update_frame(self):
        ret,self.image=self.capture.read()
        self.image=cv2.flip(self.image,1)

        self.detected_image=self.detect(self.image)
        self.displayImage(self.detected_image)

    def detect(self, image_np):
        with self.detection_graph.as_default():
            with tf.Session(graph=self.detection_graph) as sess:
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(image_np, axis=0)
                image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
                # Each box represents a part of the image where a particular object was detected.
                boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
                # Each score represent how level of confidence for each of the objects.
                # Score is shown on the result image, together with the class label.
                scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
                classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
                # Actual detection.
                (boxes, scores, classes, num_detections) = sess.run(
                    [boxes, scores, classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})
                # Visualization of the results of a detection.
                vis_util.visualize_boxes_and_labels_on_image_array(
                    image_np,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    self.category_index,
                    use_normalized_coordinates=True,
                    line_thickness=8)

        return image_np
    def displayImage(self,img):
        qformat=QImage.Format_Indexed8
        if len(img.shape)==3:
            if img.shape[2]==4:
                qformat=QImage.Format_RGBA8888
            else:
                qformat=QImage.Format_RGB888

        outImage=QImage(img,img.shape[1],img.shape[0],img.strides[0],qformat)
        #BGR>>RGB
        outImage=outImage.rgbSwapped()

        pixmap = QPixmap.fromImage(outImage)
        self.media_label.setPixmap(pixmap)
        self.media_label.setScaledContents(True)
        self.resize(pixmap.width(), pixmap.height())
        self.media.addWidget(self.media_label)
        self.media.setAlignment(Qt.AlignCenter)

    def on_tier_currentIndexChanged(self, index):
        # Change the models to show based on tier selected
        self.step2ChooseModelComboBox.clear()
        if str(self.step1ChooseTierComboBox.currentText()) == 'Tier 1':
            self.step2ChooseModelComboBox.addItems(['Faster RCNN Inception V2 Coco', 'SSD Inception V2 Coco'])
        elif str(self.step1ChooseTierComboBox.currentText()) == 'Tier 2':
            self.step2ChooseModelComboBox.addItems(['Faster RCNN Inception V2 Coco', 'Faster RCNN Resnet101 Kitti',
            'RFCN Resnet101 Coco', 'SSD Inception V2 Coco'])
        elif str(self.step1ChooseTierComboBox.currentText()) == 'Tier 3':
            self.step2ChooseModelComboBox.addItems(['Faster RCNN Inception V2 Coco', 'SSD V2 Coco'])
        else:
            self.step2ChooseModelComboBox.addItems(['SSD V2 Coco'])

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.showMaximized()
        MainWindow.setWindowTitle(_translate("Video and Image Detection", "Video and Image Detection"))
        self.label.setText(_translate("MainWindow", "Tiered Object Recognition - Image and Video Detection"))
        self.step1Label.setText(_translate("MainWindow", "Step 1: Choose tier!"))
        self.step1ChooseTierComboBox.setItemText(0, _translate("MainWindow", "Tier 1"))
        self.step1ChooseTierComboBox.setItemText(1, _translate("MainWindow", "Tier 2"))
        self.step1ChooseTierComboBox.setItemText(2, _translate("MainWindow", "Tier 3"))
        self.step1ChooseTierComboBox.setItemText(3, _translate("MainWindow", "Tier 4"))
        self.step2Label.setText(_translate("MainWindow", "Step 2: Choose model!"))
        self.step2ChooseModelComboBox.setItemText(0, _translate("MainWindow", "Faster RCNN Inception V2 Coco"))
        self.step2ChooseModelComboBox.setItemText(1, _translate("MainWindow", "Faster RCNN Resnet101 Kitti"))
        self.step2ChooseModelComboBox.setItemText(2, _translate("MainWindow", "RFCN Resnet101 Coco"))
        self.step2ChooseModelComboBox.setItemText(3, _translate("MainWindow", "Faster RCNN Resnet101 Kitti"))
        self.step3UploadImageOrVideoLabel.setText(_translate("MainWindow", "Step 3: Upload or capture!"))
        self.pushButton_2.setText(_translate("MainWindow", "Upload"))
        self.pushButton_3.setText(_translate("MainWindow", "Capture"))
        #self.pushButton.setText(_translate("MainWindow", "Submit"))
        self.pushButton.setText(_translate("MainWindow", "Exit"))
        self.stop.setText(_translate("MainWindow", "Stop"))

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
