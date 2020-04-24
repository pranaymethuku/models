#!/home/rkabealo/anaconda3/envs/tor/bin/python
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtPrintSupport import *
from PyQt5.QtMultimedia import *
from PyQt5.QtMultimediaWidgets import *
import sys
import os
import detection

VIDEOS = [".mov", ".mp4", ".flv", ".avi", ".ogg", ".wmv"]

class Ui_MainWindow(QWidget):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(797, 552)
        MainWindow.setWindowFlags(Qt.WindowCloseButtonHint | Qt.WindowMinimizeButtonHint)

        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        # Create horizontal widget
        self.horizontalLayoutWidget = QWidget(self.centralwidget)
        self.horizontalLayoutWidget.setGeometry(QRect(0, 0, 791, 61))
        self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")

        # Create the layout for horizontal widget
        self.main_data = QHBoxLayout(self.horizontalLayoutWidget)
        self.main_data.setContentsMargins(10, 0, 0, 0)
        self.main_data.setObjectName("main_data")

        # Populate Tier option
        self.tier = QComboBox(self.horizontalLayoutWidget)
        self.tier.addItems(["Tier 1", "Tier 2", "Tier 3", "Tier 4"])
        self.main_data.addWidget(self.tier)

        # Populate Model option
        self.model = QComboBox(self.horizontalLayoutWidget)
        self.model.addItems(['Faster RCNN Inception V2 Coco', 'SSD Inception V2 Coco'])
        self.main_data.addWidget(self.model)

        # Change Model option based on what the user selects for Tier
        self.tier.currentIndexChanged[str].connect(self.on_tier_currentIndexChanged)
        self.tier.setCurrentIndex(0)

        # Open File Button and action method
        self.open = QPushButton("Open File", self.horizontalLayoutWidget)
        self.open.setMaximumSize(QSize(100, 16777215))
        self.open.setObjectName("open")
        self.open.clicked.connect(self.open_file)

        self.main_data.addWidget(self.open)

        # Widget for displaying media
        self.media_layout = QWidget(self.centralwidget)
        self.media_layout.setGeometry(QRect(0, 60, 791, 451))
        self.media_layout.setObjectName("media_layout")

        # Create the layout
        self.media = QHBoxLayout(self.media_layout)
        self.media.setContentsMargins(0, 0, 0, 0)
        self.media.setObjectName("media")

        # Label will be used for images
        self.label = QLabel(self)
        MainWindow.setCentralWidget(self.centralwidget)

        # Menubar at top
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setGeometry(QRect(0, 0, 797, 21))
        self.menubar.setObjectName("menubar")
        self.menuFile = QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        self.menuFile.setTitle("File")
        MainWindow.setMenuBar(self.menubar)

        # File->Open
        self.actionOpen = QAction(MainWindow)
        self.actionOpen.setText("Open")
        self.actionOpen.setObjectName("actionOpen")
        self.actionOpen.triggered.connect(self.display)

        # File->Exit
        self.actionExit = QAction(MainWindow)
        self.actionExit.setObjectName("actionExit")
        self.actionExit.setShortcut("Ctrl+Q")
        self.actionExit.triggered.connect(self.exit)
        self.actionExit.setText("Exit")

        # Adding options to menu
        self.menuFile.addAction(self.actionOpen)
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.actionExit)
        self.menubar.addAction(self.menuFile.menuAction())

        QMetaObject.connectSlotsByName(MainWindow)

    def exit(self):
        sys.exit()

    def clear_screen(self):
        for i in reversed(range(self.media.count())): 
            self.media.itemAt(i).widget().setParent(None)

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
            self.label.setPixmap(pixmap)
            self.resize(pixmap.width(), pixmap.height())
            self.media.addWidget(self.label)
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
            self.player.setPosition(0) # to start at the beginning of the video every time

            self.media.addWidget(self.video)
            self.player.play()

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
            detection.video_detection(frozen_graph, labelmap, name, "predicted.mp4")
            self.display("predicted.mp4")

    def get_path(self):
        # Get path from the tier number
        tier = self.tier.currentText().split(" ")[1]
        labelmap = "../tor_results/tier_" + tier + "/labelmap.pbtxt"
        frozen_graph = ""
        if self.model.currentText() == "Faster RCNN Inception V2 Coco":
            frozen_graph = "../tor_results/tier_" + tier + "/faster_rcnn_inception_v2_coco_2018_01_28.pb"
        elif self.model.currentText() == "Faster RCNN Resnet101 Kitti":
            frozen_graph = "../tor_results/tier_" + tier + "/_faster_rcnn_resnet101_kitti_2018_01_28.pb"
        elif self.model.currentText() == "RFCN Resnet101 Coco":
            frozen_graph = "../tor_models/tier_" + tier + "/rfcn_resnet101_coco_2018_01_28.pb"
        elif self.model.currentText() == "Faster RCNN Resnet101 Kitti":
            frozen_graph = "../tor_models/tier_" + tier + "/ssd_inception_v2_coco_2018_01_28.pb"

        return labelmap, frozen_graph

    def on_tier_currentIndexChanged(self, index):
        # Change the models to show based on tier selected
        self.model.clear()
        if str(self.tier.currentText()) == 'Tier 1':
            self.model.addItems(['Faster RCNN Inception V2 Coco', 'SSD Inception V2 Coco'])
        elif str(self.tier.currentText()) == 'Tier 2':
            self.model.addItems(['Faster RCNN Inception V2 Coco', 'Faster RCNN Resnet101 Kitti', 
            'RFCN Resnet101 Coco', 'SSD Inception V2 Coco'])
        elif str(self.tier.currentText()) == 'Tier 3':
            self.model.addItems(['3 Model 1', '3 Model 2'])
        else:
            self.model.addItems(['4 Model 1', '4 Model 2'])

if __name__ == "__main__":
    app = QApplication(sys.argv)
    MainWindow = QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
