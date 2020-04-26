# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'design.ui'
#
# Created by: PyQt5 UI code generator 5.14.2
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(982, 713)
        MainWindow.setMinimumSize(QtCore.QSize(982, 713))
        font = QtGui.QFont()
        font.setFamily("consolas")
        font.setPointSize(24)
        MainWindow.setFont(font)

        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        self.gridLayout_2 = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout_2.setObjectName("gridLayout_2")

        self.horizontal_layout_title = QtWidgets.QHBoxLayout()
        self.horizontal_layout_title.setObjectName("horizontal_layout_title")
        self.title = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(36)
        font.setUnderline(False)
        self.title.setFont(font)
        self.title.setAlignment(QtCore.Qt.AlignCenter)
        self.title.setObjectName("title")
        self.horizontal_layout_title.addWidget(self.title)
        self.gridLayout_2.addLayout(self.horizontal_layout_title, 0, 0, 1, 2)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.step1Label = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setUnderline(True)
        self.step1Label.setFont(font)
        self.step1Label.setObjectName("step1Label")
        self.horizontalLayout.addWidget(self.step1Label)
        self.tier_dropdown = QtWidgets.QComboBox(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.tier_dropdown.setFont(font)
        self.tier_dropdown.setObjectName("tier_dropdown")
        self.tier_dropdown.addItem("")
        self.tier_dropdown.addItem("")
        self.tier_dropdown.addItem("")
        self.tier_dropdown.addItem("")
        self.horizontalLayout.addWidget(self.tier_dropdown)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.step_2_label = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setUnderline(True)
        self.step_2_label.setFont(font)
        self.step_2_label.setObjectName("step_2_label")
        self.horizontalLayout.addWidget(self.step_2_label)
        self.model_dropdown = QtWidgets.QComboBox(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.model_dropdown.setFont(font)
        self.model_dropdown.setObjectName("model_dropdown")
        self.model_dropdown.addItem("")
        self.model_dropdown.addItem("")
        self.model_dropdown.addItem("")
        self.model_dropdown.addItem("")
        self.horizontalLayout.addWidget(self.model_dropdown)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem1)
        self.step_3_label = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setUnderline(True)
        self.step_3_label.setFont(font)
        self.step_3_label.setObjectName("step_3_label")
        self.horizontalLayout.addWidget(self.step_3_label)
        self.upload_button = QtWidgets.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        self.upload_button.setFont(font)
        self.upload_button.setObjectName("upload_button")
        self.horizontalLayout.addWidget(self.upload_button)
        self.capture_button = QtWidgets.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(13)
        self.capture_button.setFont(font)
        self.capture_button.setObjectName("capture_button")
        self.horizontalLayout.addWidget(self.capture_button)
        self.gridLayout_2.addLayout(self.horizontalLayout, 1, 0, 1, 2)
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.graphicsView = QtWidgets.QGraphicsView(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.graphicsView.sizePolicy().hasHeightForWidth())
        self.graphicsView.setSizePolicy(sizePolicy)
        self.graphicsView.setAlignment(QtCore.Qt.AlignBottom|QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing)
        self.graphicsView.setObjectName("graphicsView")
        self.gridLayout.addWidget(self.graphicsView, 0, 0, 1, 1)
        self.gridLayout_2.addLayout(self.gridLayout, 2, 0, 1, 2)
        self.hboxlayout = QtWidgets.QHBoxLayout()
        self.hboxlayout.setSizeConstraint(QtWidgets.QLayout.SetFixedSize)
        self.hboxlayout.setObjectName("hboxlayout")
        self.label = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy)
        self.label.setMinimumSize(QtCore.QSize(141, 31))
        self.label.setMaximumSize(QtCore.QSize(141, 31))
        font = QtGui.QFont()
        font.setStrikeOut(True)
        self.label.setFont(font)
        self.label.setText("")
        self.label.setPixmap(QtGui.QPixmap(":/logo/images/zel_tech_logo.png"))
        self.label.setScaledContents(True)
        self.label.setAlignment(QtCore.Qt.AlignBottom|QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft)
        self.label.setIndent(0)
        self.label.setObjectName("label")
        self.hboxlayout.addWidget(self.label)
        self.gridLayout_2.addLayout(self.hboxlayout, 3, 0, 1, 1)
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setSizeConstraint(QtWidgets.QLayout.SetFixedSize)
        self.horizontalLayout_5.setContentsMargins(0, -1, -1, -1)
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        spacerItem2 = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        self.horizontalLayout_5.addItem(spacerItem2)
        spacerItem3 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_5.addItem(spacerItem3)
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton.sizePolicy().hasHeightForWidth())
        self.pushButton.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        self.pushButton.setFont(font)
        self.pushButton.setIconSize(QtCore.QSize(0, 0))
        self.pushButton.setObjectName("pushButton")
        self.horizontalLayout_5.addWidget(self.pushButton)
        self.gridLayout_2.addLayout(self.horizontalLayout_5, 3, 1, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.title.setText(_translate("MainWindow", "Tiered Object Recognition - Image and Video Detection"))
        self.step1Label.setText(_translate("MainWindow", "Step 1: Choose tier!"))
        self.tier_dropdown.setItemText(0, _translate("MainWindow", "Tier 1"))
        self.tier_dropdown.setItemText(1, _translate("MainWindow", "Tier 2"))
        self.tier_dropdown.setItemText(2, _translate("MainWindow", "Tier 3"))
        self.tier_dropdown.setItemText(3, _translate("MainWindow", "Tier 4"))
        self.step_2_label.setText(_translate("MainWindow", "Step 2: Choose model!"))
        self.model_dropdown.setItemText(0, _translate("MainWindow", "Faster RCNN Inception V2 Coco"))
        self.model_dropdown.setItemText(1, _translate("MainWindow", "Faster RCNN Resnet101 Kitti"))
        self.model_dropdown.setItemText(2, _translate("MainWindow", "RFCN Resnet101 Coco"))
        self.model_dropdown.setItemText(3, _translate("MainWindow", "Faster RCNN Resnet101 Kitti"))
        self.step_3_label.setText(_translate("MainWindow", "Step 3: Upload or capture!"))
        self.upload_button.setText(_translate("MainWindow", "Upload"))
        self.capture_button.setText(_translate("MainWindow", "Capture"))
        self.pushButton.setText(_translate("MainWindow", "Exit"))

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
