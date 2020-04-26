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
        MainWindow.setMaximumSize(QtCore.QSize(982, 713))
        font = QtGui.QFont()
        font.setFamily("Futura")
        font.setPointSize(24)
        MainWindow.setFont(font)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.formLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.formLayoutWidget.setGeometry(QtCore.QRect(10, 50, 996, 61))
        self.formLayoutWidget.setObjectName("formLayoutWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.formLayoutWidget)
        self.horizontalLayout.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.step1Label = QtWidgets.QLabel(self.formLayoutWidget)
        font = QtGui.QFont()
        font.setUnderline(True)
        self.step1Label.setFont(font)
        self.step1Label.setObjectName("step1Label")
        self.horizontalLayout.addWidget(self.step1Label)
        self.tier_dropdown = QtWidgets.QComboBox(self.formLayoutWidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.tier_dropdown.setFont(font)
        self.tier_dropdown.setObjectName("tier_dropdown")
        self.tier_dropdown.addItem("")
        self.tier_dropdown.addItem("")
        self.tier_dropdown.addItem("")
        self.tier_dropdown.addItem("")
        self.horizontalLayout.addWidget(self.tier_dropdown)
        self.step_2_label = QtWidgets.QLabel(self.formLayoutWidget)
        font = QtGui.QFont()
        font.setUnderline(True)
        self.step_2_label.setFont(font)
        self.step_2_label.setObjectName("step_2_label")
        self.horizontalLayout.addWidget(self.step_2_label)
        self.model_dropdown = QtWidgets.QComboBox(self.formLayoutWidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.model_dropdown.setFont(font)
        self.model_dropdown.setObjectName("model_dropdown")
        self.model_dropdown.addItem("")
        self.model_dropdown.addItem("")
        self.model_dropdown.addItem("")
        self.model_dropdown.addItem("")
        self.horizontalLayout.addWidget(self.model_dropdown)
        self.step_3_label = QtWidgets.QLabel(self.formLayoutWidget)
        font = QtGui.QFont()
        font.setUnderline(True)
        self.step_3_label.setFont(font)
        self.step_3_label.setObjectName("step_3_label")
        self.horizontalLayout.addWidget(self.step_3_label)
        self.upload_button = QtWidgets.QPushButton(self.formLayoutWidget)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        self.upload_button.setFont(font)
        self.upload_button.setObjectName("upload_button")
        self.horizontalLayout.addWidget(self.upload_button)
        self.capture_button = QtWidgets.QPushButton(self.formLayoutWidget)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.capture_button.setFont(font)
        self.capture_button.setObjectName("capture_button")
        self.horizontalLayout.addWidget(self.capture_button)
        self.horizontalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(10, 0, 961, 51))
        self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget)
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label_3 = QtWidgets.QLabel(self.horizontalLayoutWidget)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(36)
        font.setUnderline(False)
        self.label_3.setFont(font)
        self.label_3.setAlignment(QtCore.Qt.AlignCenter)
        self.label_3.setObjectName("label_3")
        self.horizontalLayout_2.addWidget(self.label_3)
        self.gridLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.gridLayoutWidget.setGeometry(QtCore.QRect(10, 110, 961, 511))
        self.gridLayoutWidget.setObjectName("gridLayoutWidget")
        self.gridLayout = QtWidgets.QGridLayout(self.gridLayoutWidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.formLayout = QtWidgets.QFormLayout()
        self.formLayout.setObjectName("formLayout")
        self.gridLayout.addLayout(self.formLayout, 0, 0, 1, 1)
        self.horizontalLayoutWidget_2 = QtWidgets.QWidget(self.centralwidget)
        self.horizontalLayoutWidget_2.setGeometry(QtCore.QRect(10, 660, 161, 41))
        self.horizontalLayoutWidget_2.setObjectName("horizontalLayoutWidget_2")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_2)
        self.horizontalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.label = QtWidgets.QLabel(self.horizontalLayoutWidget_2)
        self.label.setMinimumSize(QtCore.QSize(141, 31))
        self.label.setMaximumSize(QtCore.QSize(141, 31))
        font = QtGui.QFont()
        font.setStrikeOut(True)
        self.label.setFont(font)
        self.label.setText("")
        self.label.setPixmap(QtGui.QPixmap(":/logo/images/zel_tech_logo.png"))
        self.label.setScaledContents(True)
        self.label.setAlignment(QtCore.Qt.AlignBottom|QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing)
        self.label.setObjectName("label")
        self.horizontalLayout_4.addWidget(self.label)
        self.horizontalLayoutWidget_3 = QtWidgets.QWidget(self.centralwidget)
        self.horizontalLayoutWidget_3.setGeometry(QtCore.QRect(860, 620, 111, 51))
        self.horizontalLayoutWidget_3.setObjectName("horizontalLayoutWidget_3")
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_3)
        self.horizontalLayout_5.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.pushButton = QtWidgets.QPushButton(self.horizontalLayoutWidget_3)
        self.pushButton.setObjectName("pushButton")
        self.horizontalLayout_5.addWidget(self.pushButton)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
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
        self.label_3.setText(_translate("MainWindow", "Tiered Object Recognition - Image and Video Detection"))
        self.pushButton.setText(_translate("MainWindow", "Exit"))
import image_rc


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())