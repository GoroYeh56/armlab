# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'armlab_gui.ui'
#
# Created by: PyQt4 UI code generator 4.12.1
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName(_fromUtf8("MainWindow"))
        MainWindow.resize(1750, 1048)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        MainWindow.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))
        self.centralwidget = QtGui.QWidget(MainWindow)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.verticalLayout_14 = QtGui.QVBoxLayout(self.centralwidget)
        self.verticalLayout_14.setObjectName(_fromUtf8("verticalLayout_14"))
        self.horizontalLayout_13 = QtGui.QHBoxLayout()
        self.horizontalLayout_13.setObjectName(_fromUtf8("horizontalLayout_13"))
        self.OutputFrame = QtGui.QFrame(self.centralwidget)
        self.OutputFrame.setEnabled(True)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.OutputFrame.sizePolicy().hasHeightForWidth())
        self.OutputFrame.setSizePolicy(sizePolicy)
        self.OutputFrame.setFrameShape(QtGui.QFrame.StyledPanel)
        self.OutputFrame.setFrameShadow(QtGui.QFrame.Raised)
        self.OutputFrame.setObjectName(_fromUtf8("OutputFrame"))
        self.verticalLayout_5 = QtGui.QVBoxLayout(self.OutputFrame)
        self.verticalLayout_5.setObjectName(_fromUtf8("verticalLayout_5"))
        self.JointCoordLabel = QtGui.QLabel(self.OutputFrame)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.JointCoordLabel.sizePolicy().hasHeightForWidth())
        self.JointCoordLabel.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.JointCoordLabel.setFont(font)
        self.JointCoordLabel.setObjectName(_fromUtf8("JointCoordLabel"))
        self.verticalLayout_5.addWidget(self.JointCoordLabel)
        self.horizontalLayout_3 = QtGui.QHBoxLayout()
        self.horizontalLayout_3.setObjectName(_fromUtf8("horizontalLayout_3"))
        self.verticalLayout_9 = QtGui.QVBoxLayout()
        self.verticalLayout_9.setObjectName(_fromUtf8("verticalLayout_9"))
        self.BLabel = QtGui.QLabel(self.OutputFrame)
        font = QtGui.QFont()
        font.setFamily(_fromUtf8("Ubuntu Mono"))
        font.setPointSize(12)
        self.BLabel.setFont(font)
        self.BLabel.setObjectName(_fromUtf8("BLabel"))
        self.verticalLayout_9.addWidget(self.BLabel, 0, QtCore.Qt.AlignRight)
        self.SLabel = QtGui.QLabel(self.OutputFrame)
        font = QtGui.QFont()
        font.setFamily(_fromUtf8("Ubuntu Mono"))
        font.setPointSize(12)
        self.SLabel.setFont(font)
        self.SLabel.setObjectName(_fromUtf8("SLabel"))
        self.verticalLayout_9.addWidget(self.SLabel, 0, QtCore.Qt.AlignRight)
        self.ELabel = QtGui.QLabel(self.OutputFrame)
        font = QtGui.QFont()
        font.setFamily(_fromUtf8("Ubuntu Mono"))
        font.setPointSize(12)
        self.ELabel.setFont(font)
        self.ELabel.setObjectName(_fromUtf8("ELabel"))
        self.verticalLayout_9.addWidget(self.ELabel, 0, QtCore.Qt.AlignRight)
        self.WALabel = QtGui.QLabel(self.OutputFrame)
        font = QtGui.QFont()
        font.setFamily(_fromUtf8("Ubuntu Mono"))
        font.setPointSize(12)
        self.WALabel.setFont(font)
        self.WALabel.setObjectName(_fromUtf8("WALabel"))
        self.verticalLayout_9.addWidget(self.WALabel, 0, QtCore.Qt.AlignRight)
        self.WRLabel = QtGui.QLabel(self.OutputFrame)
        font = QtGui.QFont()
        font.setFamily(_fromUtf8("Ubuntu Mono"))
        font.setPointSize(12)
        self.WRLabel.setFont(font)
        self.WRLabel.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.WRLabel.setObjectName(_fromUtf8("WRLabel"))
        self.verticalLayout_9.addWidget(self.WRLabel)
        self.horizontalLayout_3.addLayout(self.verticalLayout_9)
        self.verticalLayout_8 = QtGui.QVBoxLayout()
        self.verticalLayout_8.setObjectName(_fromUtf8("verticalLayout_8"))
        self.rdoutBaseJC = QtGui.QLabel(self.OutputFrame)
        font = QtGui.QFont()
        font.setFamily(_fromUtf8("Ubuntu Mono"))
        font.setPointSize(12)
        self.rdoutBaseJC.setFont(font)
        self.rdoutBaseJC.setObjectName(_fromUtf8("rdoutBaseJC"))
        self.verticalLayout_8.addWidget(self.rdoutBaseJC, 0, QtCore.Qt.AlignLeft)
        self.rdoutShoulderJC = QtGui.QLabel(self.OutputFrame)
        font = QtGui.QFont()
        font.setFamily(_fromUtf8("Ubuntu Mono"))
        font.setPointSize(12)
        self.rdoutShoulderJC.setFont(font)
        self.rdoutShoulderJC.setObjectName(_fromUtf8("rdoutShoulderJC"))
        self.verticalLayout_8.addWidget(self.rdoutShoulderJC, 0, QtCore.Qt.AlignLeft)
        self.rdoutElbowJC = QtGui.QLabel(self.OutputFrame)
        font = QtGui.QFont()
        font.setFamily(_fromUtf8("Ubuntu Mono"))
        font.setPointSize(12)
        self.rdoutElbowJC.setFont(font)
        self.rdoutElbowJC.setObjectName(_fromUtf8("rdoutElbowJC"))
        self.verticalLayout_8.addWidget(self.rdoutElbowJC, 0, QtCore.Qt.AlignLeft)
        self.rdoutWristAJC = QtGui.QLabel(self.OutputFrame)
        font = QtGui.QFont()
        font.setFamily(_fromUtf8("Ubuntu Mono"))
        font.setPointSize(12)
        self.rdoutWristAJC.setFont(font)
        self.rdoutWristAJC.setObjectName(_fromUtf8("rdoutWristAJC"))
        self.verticalLayout_8.addWidget(self.rdoutWristAJC, 0, QtCore.Qt.AlignLeft)
        self.rdoutWristRJC = QtGui.QLabel(self.OutputFrame)
        font = QtGui.QFont()
        font.setFamily(_fromUtf8("Ubuntu Mono"))
        font.setPointSize(12)
        self.rdoutWristRJC.setFont(font)
        self.rdoutWristRJC.setObjectName(_fromUtf8("rdoutWristRJC"))
        self.verticalLayout_8.addWidget(self.rdoutWristRJC)
        self.horizontalLayout_3.addLayout(self.verticalLayout_8)
        self.verticalLayout_5.addLayout(self.horizontalLayout_3)
        self.WorldCoordLabel = QtGui.QLabel(self.OutputFrame)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.WorldCoordLabel.setFont(font)
        self.WorldCoordLabel.setObjectName(_fromUtf8("WorldCoordLabel"))
        self.verticalLayout_5.addWidget(self.WorldCoordLabel)
        self.horizontalLayout_4 = QtGui.QHBoxLayout()
        self.horizontalLayout_4.setObjectName(_fromUtf8("horizontalLayout_4"))
        self.verticalLayout_13 = QtGui.QVBoxLayout()
        self.verticalLayout_13.setObjectName(_fromUtf8("verticalLayout_13"))
        self.XLabel = QtGui.QLabel(self.OutputFrame)
        font = QtGui.QFont()
        font.setFamily(_fromUtf8("Ubuntu Mono"))
        font.setPointSize(12)
        self.XLabel.setFont(font)
        self.XLabel.setScaledContents(False)
        self.XLabel.setObjectName(_fromUtf8("XLabel"))
        self.verticalLayout_13.addWidget(self.XLabel, 0, QtCore.Qt.AlignRight)
        self.YLabel = QtGui.QLabel(self.OutputFrame)
        font = QtGui.QFont()
        font.setFamily(_fromUtf8("Ubuntu Mono"))
        font.setPointSize(12)
        self.YLabel.setFont(font)
        self.YLabel.setObjectName(_fromUtf8("YLabel"))
        self.verticalLayout_13.addWidget(self.YLabel, 0, QtCore.Qt.AlignRight)
        self.ZLabel = QtGui.QLabel(self.OutputFrame)
        font = QtGui.QFont()
        font.setFamily(_fromUtf8("Ubuntu Mono"))
        font.setPointSize(12)
        self.ZLabel.setFont(font)
        self.ZLabel.setObjectName(_fromUtf8("ZLabel"))
        self.verticalLayout_13.addWidget(self.ZLabel, 0, QtCore.Qt.AlignRight)
        self.PhiLabel = QtGui.QLabel(self.OutputFrame)
        self.PhiLabel.setEnabled(True)
        font = QtGui.QFont()
        font.setFamily(_fromUtf8("Ubuntu Mono"))
        font.setPointSize(12)
        self.PhiLabel.setFont(font)
        self.PhiLabel.setObjectName(_fromUtf8("PhiLabel"))
        self.verticalLayout_13.addWidget(self.PhiLabel, 0, QtCore.Qt.AlignRight)
        self.ThetaLabel = QtGui.QLabel(self.OutputFrame)
        self.ThetaLabel.setEnabled(True)
        font = QtGui.QFont()
        font.setFamily(_fromUtf8("Ubuntu Mono"))
        font.setPointSize(12)
        self.ThetaLabel.setFont(font)
        self.ThetaLabel.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.ThetaLabel.setObjectName(_fromUtf8("ThetaLabel"))
        self.verticalLayout_13.addWidget(self.ThetaLabel)
        self.PsiLabel = QtGui.QLabel(self.OutputFrame)
        self.PsiLabel.setEnabled(True)
        font = QtGui.QFont()
        font.setFamily(_fromUtf8("Ubuntu Mono"))
        font.setPointSize(12)
        self.PsiLabel.setFont(font)
        self.PsiLabel.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.PsiLabel.setObjectName(_fromUtf8("PsiLabel"))
        self.verticalLayout_13.addWidget(self.PsiLabel)
        self.horizontalLayout_4.addLayout(self.verticalLayout_13)
        self.verticalLayout_12 = QtGui.QVBoxLayout()
        self.verticalLayout_12.setObjectName(_fromUtf8("verticalLayout_12"))
        self.rdoutX = QtGui.QLabel(self.OutputFrame)
        font = QtGui.QFont()
        font.setFamily(_fromUtf8("Ubuntu Mono"))
        font.setPointSize(12)
        self.rdoutX.setFont(font)
        self.rdoutX.setObjectName(_fromUtf8("rdoutX"))
        self.verticalLayout_12.addWidget(self.rdoutX, 0, QtCore.Qt.AlignLeft)
        self.rdoutY = QtGui.QLabel(self.OutputFrame)
        font = QtGui.QFont()
        font.setFamily(_fromUtf8("Ubuntu Mono"))
        font.setPointSize(12)
        self.rdoutY.setFont(font)
        self.rdoutY.setObjectName(_fromUtf8("rdoutY"))
        self.verticalLayout_12.addWidget(self.rdoutY, 0, QtCore.Qt.AlignLeft)
        self.rdoutZ = QtGui.QLabel(self.OutputFrame)
        font = QtGui.QFont()
        font.setFamily(_fromUtf8("Ubuntu Mono"))
        font.setPointSize(12)
        self.rdoutZ.setFont(font)
        self.rdoutZ.setObjectName(_fromUtf8("rdoutZ"))
        self.verticalLayout_12.addWidget(self.rdoutZ, 0, QtCore.Qt.AlignLeft)
        self.rdoutPhi = QtGui.QLabel(self.OutputFrame)
        font = QtGui.QFont()
        font.setFamily(_fromUtf8("Ubuntu Mono"))
        font.setPointSize(12)
        self.rdoutPhi.setFont(font)
        self.rdoutPhi.setObjectName(_fromUtf8("rdoutPhi"))
        self.verticalLayout_12.addWidget(self.rdoutPhi, 0, QtCore.Qt.AlignLeft)
        self.rdoutTheta = QtGui.QLabel(self.OutputFrame)
        font = QtGui.QFont()
        font.setFamily(_fromUtf8("Ubuntu Mono"))
        font.setPointSize(12)
        self.rdoutTheta.setFont(font)
        self.rdoutTheta.setObjectName(_fromUtf8("rdoutTheta"))
        self.verticalLayout_12.addWidget(self.rdoutTheta)
        self.rdoutPsi = QtGui.QLabel(self.OutputFrame)
        font = QtGui.QFont()
        font.setFamily(_fromUtf8("Ubuntu Mono"))
        font.setPointSize(12)
        self.rdoutPsi.setFont(font)
        self.rdoutPsi.setObjectName(_fromUtf8("rdoutPsi"))
        self.verticalLayout_12.addWidget(self.rdoutPsi)
        self.horizontalLayout_4.addLayout(self.verticalLayout_12)
        self.verticalLayout_5.addLayout(self.horizontalLayout_4)
        self.Group2 = QtGui.QVBoxLayout()
        self.Group2.setMargin(10)
        self.Group2.setObjectName(_fromUtf8("Group2"))
        self.btnUser1 = QtGui.QPushButton(self.OutputFrame)
        self.btnUser1.setObjectName(_fromUtf8("btnUser1"))
        self.Group2.addWidget(self.btnUser1)
        self.btnUser2 = QtGui.QPushButton(self.OutputFrame)
        self.btnUser2.setObjectName(_fromUtf8("btnUser2"))
        self.Group2.addWidget(self.btnUser2)
        self.btnUser3 = QtGui.QPushButton(self.OutputFrame)
        self.btnUser3.setObjectName(_fromUtf8("btnUser3"))
        self.Group2.addWidget(self.btnUser3)
        self.btnUser4 = QtGui.QPushButton(self.OutputFrame)
        self.btnUser4.setObjectName(_fromUtf8("btnUser4"))
        self.Group2.addWidget(self.btnUser4)
        self.btnUser5 = QtGui.QPushButton(self.OutputFrame)
        self.btnUser5.setObjectName(_fromUtf8("btnUser5"))
        self.Group2.addWidget(self.btnUser5)
        self.btnUser6 = QtGui.QPushButton(self.OutputFrame)
        self.btnUser6.setObjectName(_fromUtf8("btnUser6"))
        self.Group2.addWidget(self.btnUser6)
        self.btnUser7 = QtGui.QPushButton(self.OutputFrame)
        self.btnUser7.setObjectName(_fromUtf8("btnUser7"))
        self.Group2.addWidget(self.btnUser7)
        self.btnUser8 = QtGui.QPushButton(self.OutputFrame)
        self.btnUser8.setObjectName(_fromUtf8("btnUser8"))
        self.Group2.addWidget(self.btnUser8)
        self.btnUser9 = QtGui.QPushButton(self.OutputFrame)
        self.btnUser9.setObjectName(_fromUtf8("btnUser9"))
        self.Group2.addWidget(self.btnUser9)
        self.btnUser10 = QtGui.QPushButton(self.OutputFrame)
        self.btnUser10.setAutoRepeatDelay(300)
        self.btnUser10.setObjectName(_fromUtf8("btnUser10"))
        self.Group2.addWidget(self.btnUser10)
        self.btnUser11 = QtGui.QPushButton(self.OutputFrame)
        self.btnUser11.setAutoRepeatDelay(300)
        self.btnUser11.setObjectName(_fromUtf8("btnUser11"))
        self.Group2.addWidget(self.btnUser11)
        self.btnUser12 = QtGui.QPushButton(self.OutputFrame)
        self.btnUser12.setAutoRepeatDelay(300)
        self.btnUser12.setObjectName(_fromUtf8("btnUser12"))
        self.Group2.addWidget(self.btnUser12)


        self.btnUser13 = QtGui.QPushButton(self.OutputFrame)
        self.btnUser13.setAutoRepeatDelay(300)
        self.btnUser13.setObjectName(_fromUtf8("btnUser13"))
        self.Group2.addWidget(self.btnUser13)
        self.btnUser14 = QtGui.QPushButton(self.OutputFrame)
        self.btnUser14.setAutoRepeatDelay(300)
        self.btnUser14.setObjectName(_fromUtf8("btnUser14"))
        self.Group2.addWidget(self.btnUser14)

        self.btnUser15 = QtGui.QPushButton(self.OutputFrame)
        self.btnUser15.setAutoRepeatDelay(300)
        self.btnUser15.setObjectName(_fromUtf8("btnUser15"))
        self.Group2.addWidget(self.btnUser15)

        self.btnUser16 = QtGui.QPushButton(self.OutputFrame)
        self.btnUser16.setAutoRepeatDelay(300)
        self.btnUser16.setObjectName(_fromUtf8("btnUser16"))
        self.Group2.addWidget(self.btnUser16)




        self.verticalLayout_5.addLayout(self.Group2)
        spacerItem = QtGui.QSpacerItem(20, 40, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding)
        self.verticalLayout_5.addItem(spacerItem)
        self.horizontalLayout_13.addWidget(self.OutputFrame)
        self.verticalLayout_4 = QtGui.QVBoxLayout()
        self.verticalLayout_4.setObjectName(_fromUtf8("verticalLayout_4"))
        self.horizontalLayout_12 = QtGui.QHBoxLayout()
        self.horizontalLayout_12.setObjectName(_fromUtf8("horizontalLayout_12"))
        spacerItem1 = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_12.addItem(spacerItem1)
        self.videoDisplay = QtGui.QLabel(self.centralwidget)
        self.videoDisplay.setMinimumSize(QtCore.QSize(1280, 720))
        self.videoDisplay.setMaximumSize(QtCore.QSize(1280, 720))
        self.videoDisplay.setCursor(QtGui.QCursor(QtCore.Qt.CrossCursor))
        self.videoDisplay.setMouseTracking(True)
        self.videoDisplay.setFrameShape(QtGui.QFrame.Box)
        self.videoDisplay.setObjectName(_fromUtf8("videoDisplay"))
        self.horizontalLayout_12.addWidget(self.videoDisplay)
        spacerItem2 = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_12.addItem(spacerItem2)
        self.verticalLayout_4.addLayout(self.horizontalLayout_12)
        self.horizontalLayout_2 = QtGui.QHBoxLayout()
        self.horizontalLayout_2.setObjectName(_fromUtf8("horizontalLayout_2"))
        self.chk_directcontrol = QtGui.QCheckBox(self.centralwidget)
        self.chk_directcontrol.setChecked(False)
        self.chk_directcontrol.setObjectName(_fromUtf8("chk_directcontrol"))
        self.horizontalLayout_2.addWidget(self.chk_directcontrol)
        self.radioVideo = QtGui.QRadioButton(self.centralwidget)
        self.radioVideo.setChecked(True)
        self.radioVideo.setAutoExclusive(True)
        self.radioVideo.setObjectName(_fromUtf8("radioVideo"))
        self.horizontalLayout_2.addWidget(self.radioVideo)
        self.radioDepth = QtGui.QRadioButton(self.centralwidget)
        self.radioDepth.setObjectName(_fromUtf8("radioDepth"))
        self.horizontalLayout_2.addWidget(self.radioDepth)
        self.radioUsr1 = QtGui.QRadioButton(self.centralwidget)
        self.radioUsr1.setObjectName(_fromUtf8("radioUsr1"))
        self.horizontalLayout_2.addWidget(self.radioUsr1)
        self.radioUsr2 = QtGui.QRadioButton(self.centralwidget)
        self.radioUsr2.setObjectName(_fromUtf8("radioUsr2"))
        self.horizontalLayout_2.addWidget(self.radioUsr2)
        spacerItem3 = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem3)
        self.PixelCoordLabel = QtGui.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.PixelCoordLabel.setFont(font)
        self.PixelCoordLabel.setObjectName(_fromUtf8("PixelCoordLabel"))
        self.horizontalLayout_2.addWidget(self.PixelCoordLabel)
        self.rdoutMousePixels = QtGui.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily(_fromUtf8("Ubuntu Mono"))
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.rdoutMousePixels.setFont(font)
        self.rdoutMousePixels.setTextFormat(QtCore.Qt.AutoText)
        self.rdoutMousePixels.setObjectName(_fromUtf8("rdoutMousePixels"))
        self.horizontalLayout_2.addWidget(self.rdoutMousePixels)
        spacerItem4 = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem4)
        self.PixelCoordLabel_2 = QtGui.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.PixelCoordLabel_2.setFont(font)
        self.PixelCoordLabel_2.setObjectName(_fromUtf8("PixelCoordLabel_2"))
        self.horizontalLayout_2.addWidget(self.PixelCoordLabel_2)
        self.rdoutMouseWorld = QtGui.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily(_fromUtf8("Ubuntu Mono"))
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.rdoutMouseWorld.setFont(font)
        self.rdoutMouseWorld.setTextFormat(QtCore.Qt.AutoText)
        self.rdoutMouseWorld.setObjectName(_fromUtf8("rdoutMouseWorld"))
        self.horizontalLayout_2.addWidget(self.rdoutMouseWorld)
        spacerItem5 = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem5)
        self.verticalLayout_4.addLayout(self.horizontalLayout_2)
        self.SliderFrame = QtGui.QFrame(self.centralwidget)
        self.SliderFrame.setEnabled(False)
        self.SliderFrame.setFrameShape(QtGui.QFrame.StyledPanel)
        self.SliderFrame.setFrameShadow(QtGui.QFrame.Raised)
        self.SliderFrame.setLineWidth(1)
        self.SliderFrame.setObjectName(_fromUtf8("SliderFrame"))
        self.verticalLayout_16 = QtGui.QVBoxLayout(self.SliderFrame)
        self.verticalLayout_16.setObjectName(_fromUtf8("verticalLayout_16"))
        self.horizontalLayout_6 = QtGui.QHBoxLayout()
        self.horizontalLayout_6.setObjectName(_fromUtf8("horizontalLayout_6"))
        self.verticalLayout_16.addLayout(self.horizontalLayout_6)
        self.horizontalLayout_9 = QtGui.QHBoxLayout()
        self.horizontalLayout_9.setObjectName(_fromUtf8("horizontalLayout_9"))
        self.verticalLayout_2 = QtGui.QVBoxLayout()
        self.verticalLayout_2.setObjectName(_fromUtf8("verticalLayout_2"))
        self.horizontalLayout = QtGui.QHBoxLayout()
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        self.BLabelS = QtGui.QLabel(self.SliderFrame)
        self.BLabelS.setMinimumSize(QtCore.QSize(150, 0))
        self.BLabelS.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.BLabelS.setObjectName(_fromUtf8("BLabelS"))
        self.horizontalLayout.addWidget(self.BLabelS)
        self.sldrBase = QtGui.QSlider(self.SliderFrame)
        self.sldrBase.setMinimum(-179)
        self.sldrBase.setMaximum(180)
        self.sldrBase.setOrientation(QtCore.Qt.Horizontal)
        self.sldrBase.setObjectName(_fromUtf8("sldrBase"))
        self.horizontalLayout.addWidget(self.sldrBase)
        self.rdoutBase = QtGui.QLabel(self.SliderFrame)
        self.rdoutBase.setMinimumSize(QtCore.QSize(30, 0))
        self.rdoutBase.setObjectName(_fromUtf8("rdoutBase"))
        self.horizontalLayout.addWidget(self.rdoutBase)
        self.verticalLayout_2.addLayout(self.horizontalLayout)
        self.horizontalLayout_7 = QtGui.QHBoxLayout()
        self.horizontalLayout_7.setObjectName(_fromUtf8("horizontalLayout_7"))
        self.SLabelS = QtGui.QLabel(self.SliderFrame)
        self.SLabelS.setMinimumSize(QtCore.QSize(150, 0))
        self.SLabelS.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.SLabelS.setObjectName(_fromUtf8("SLabelS"))
        self.horizontalLayout_7.addWidget(self.SLabelS)
        self.sldrShoulder = QtGui.QSlider(self.SliderFrame)
        self.sldrShoulder.setMinimum(-179)
        self.sldrShoulder.setMaximum(180)
        self.sldrShoulder.setOrientation(QtCore.Qt.Horizontal)
        self.sldrShoulder.setObjectName(_fromUtf8("sldrShoulder"))
        self.horizontalLayout_7.addWidget(self.sldrShoulder)
        self.rdoutShoulder = QtGui.QLabel(self.SliderFrame)
        self.rdoutShoulder.setObjectName(_fromUtf8("rdoutShoulder"))
        self.horizontalLayout_7.addWidget(self.rdoutShoulder)
        self.verticalLayout_2.addLayout(self.horizontalLayout_7)
        self.horizontalLayout_8 = QtGui.QHBoxLayout()
        self.horizontalLayout_8.setObjectName(_fromUtf8("horizontalLayout_8"))
        self.ELabelS = QtGui.QLabel(self.SliderFrame)
        self.ELabelS.setMinimumSize(QtCore.QSize(150, 0))
        self.ELabelS.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.ELabelS.setObjectName(_fromUtf8("ELabelS"))
        self.horizontalLayout_8.addWidget(self.ELabelS)
        self.sldrElbow = QtGui.QSlider(self.SliderFrame)
        self.sldrElbow.setMinimum(-179)
        self.sldrElbow.setMaximum(180)
        self.sldrElbow.setOrientation(QtCore.Qt.Horizontal)
        self.sldrElbow.setObjectName(_fromUtf8("sldrElbow"))
        self.horizontalLayout_8.addWidget(self.sldrElbow)
        self.rdoutElbow = QtGui.QLabel(self.SliderFrame)
        self.rdoutElbow.setObjectName(_fromUtf8("rdoutElbow"))
        self.horizontalLayout_8.addWidget(self.rdoutElbow)
        self.verticalLayout_2.addLayout(self.horizontalLayout_8)
        self.horizontalLayout_11 = QtGui.QHBoxLayout()
        self.horizontalLayout_11.setObjectName(_fromUtf8("horizontalLayout_11"))
        self.WALabelS = QtGui.QLabel(self.SliderFrame)
        self.WALabelS.setMinimumSize(QtCore.QSize(150, 0))
        self.WALabelS.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.WALabelS.setObjectName(_fromUtf8("WALabelS"))
        self.horizontalLayout_11.addWidget(self.WALabelS)
        self.sldrWristA = QtGui.QSlider(self.SliderFrame)
        self.sldrWristA.setMinimum(-179)
        self.sldrWristA.setMaximum(180)
        self.sldrWristA.setOrientation(QtCore.Qt.Horizontal)
        self.sldrWristA.setObjectName(_fromUtf8("sldrWristA"))
        self.horizontalLayout_11.addWidget(self.sldrWristA)
        self.rdoutWristA = QtGui.QLabel(self.SliderFrame)
        self.rdoutWristA.setObjectName(_fromUtf8("rdoutWristA"))
        self.horizontalLayout_11.addWidget(self.rdoutWristA)
        self.verticalLayout_2.addLayout(self.horizontalLayout_11)
        self.horizontalLayout_15 = QtGui.QHBoxLayout()
        self.horizontalLayout_15.setObjectName(_fromUtf8("horizontalLayout_15"))
        self.WRLabelS = QtGui.QLabel(self.SliderFrame)
        self.WRLabelS.setMinimumSize(QtCore.QSize(150, 0))
        self.WRLabelS.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.WRLabelS.setObjectName(_fromUtf8("WRLabelS"))
        self.horizontalLayout_15.addWidget(self.WRLabelS)
        self.sldrWristR = QtGui.QSlider(self.SliderFrame)
        self.sldrWristR.setMinimum(-179)
        self.sldrWristR.setMaximum(180)
        self.sldrWristR.setOrientation(QtCore.Qt.Horizontal)
        self.sldrWristR.setObjectName(_fromUtf8("sldrWristR"))
        self.horizontalLayout_15.addWidget(self.sldrWristR)
        self.rdoutWristR = QtGui.QLabel(self.SliderFrame)
        self.rdoutWristR.setObjectName(_fromUtf8("rdoutWristR"))
        self.horizontalLayout_15.addWidget(self.rdoutWristR)
        self.verticalLayout_2.addLayout(self.horizontalLayout_15)
        self.horizontalLayout_9.addLayout(self.verticalLayout_2)
        self.verticalLayout_10 = QtGui.QVBoxLayout()
        self.verticalLayout_10.setObjectName(_fromUtf8("verticalLayout_10"))
        self.MoveTimeLabel = QtGui.QLabel(self.SliderFrame)
        self.MoveTimeLabel.setObjectName(_fromUtf8("MoveTimeLabel"))
        self.verticalLayout_10.addWidget(self.MoveTimeLabel)
        self.sldrMoveTime = QtGui.QSlider(self.SliderFrame)
        self.sldrMoveTime.setMaximum(100)
        self.sldrMoveTime.setProperty("value", 30)
        self.sldrMoveTime.setOrientation(QtCore.Qt.Vertical)
        self.sldrMoveTime.setObjectName(_fromUtf8("sldrMoveTime"))
        self.verticalLayout_10.addWidget(self.sldrMoveTime)
        self.rdoutMoveTime = QtGui.QLabel(self.SliderFrame)
        self.rdoutMoveTime.setObjectName(_fromUtf8("rdoutMoveTime"))
        self.verticalLayout_10.addWidget(self.rdoutMoveTime)
        self.horizontalLayout_9.addLayout(self.verticalLayout_10)
        self.verticalLayout_7 = QtGui.QVBoxLayout()
        self.verticalLayout_7.setObjectName(_fromUtf8("verticalLayout_7"))
        self.AccelTimeLabel = QtGui.QLabel(self.SliderFrame)
        self.AccelTimeLabel.setObjectName(_fromUtf8("AccelTimeLabel"))
        self.verticalLayout_7.addWidget(self.AccelTimeLabel)
        self.sldrAccelTime = QtGui.QSlider(self.SliderFrame)
        self.sldrAccelTime.setMaximum(100)
        self.sldrAccelTime.setProperty("value", 20)
        self.sldrAccelTime.setSliderPosition(20)
        self.sldrAccelTime.setOrientation(QtCore.Qt.Vertical)
        self.sldrAccelTime.setObjectName(_fromUtf8("sldrAccelTime"))
        self.verticalLayout_7.addWidget(self.sldrAccelTime)
        self.rdoutAccelTime = QtGui.QLabel(self.SliderFrame)
        self.rdoutAccelTime.setObjectName(_fromUtf8("rdoutAccelTime"))
        self.verticalLayout_7.addWidget(self.rdoutAccelTime)
        self.horizontalLayout_9.addLayout(self.verticalLayout_7)
        self.verticalLayout_16.addLayout(self.horizontalLayout_9)
        self.horizontalLayout_10 = QtGui.QHBoxLayout()
        self.horizontalLayout_10.setObjectName(_fromUtf8("horizontalLayout_10"))
        self.verticalLayout_16.addLayout(self.horizontalLayout_10)
        self.verticalLayout_4.addWidget(self.SliderFrame)
        self.horizontalLayout_14 = QtGui.QHBoxLayout()
        self.horizontalLayout_14.setObjectName(_fromUtf8("horizontalLayout_14"))
        self.label_3 = QtGui.QLabel(self.centralwidget)
        self.label_3.setMaximumSize(QtCore.QSize(125, 16777215))
        self.label_3.setObjectName(_fromUtf8("label_3"))
        self.horizontalLayout_14.addWidget(self.label_3)
        self.rdoutStatus = QtGui.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(9)
        self.rdoutStatus.setFont(font)
        self.rdoutStatus.setTextFormat(QtCore.Qt.AutoText)
        self.rdoutStatus.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.rdoutStatus.setWordWrap(True)
        self.rdoutStatus.setObjectName(_fromUtf8("rdoutStatus"))
        self.horizontalLayout_14.addWidget(self.rdoutStatus)
        self.verticalLayout_4.addLayout(self.horizontalLayout_14)
        self.horizontalLayout_13.addLayout(self.verticalLayout_4)
        self.verticalLayout_11 = QtGui.QVBoxLayout()
        self.verticalLayout_11.setObjectName(_fromUtf8("verticalLayout_11"))
        self.btn_estop = QtGui.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.btn_estop.setFont(font)
        self.btn_estop.setObjectName(_fromUtf8("btn_estop"))
        self.verticalLayout_11.addWidget(self.btn_estop)
        self.btn_init_arm = QtGui.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.btn_init_arm.setFont(font)
        self.btn_init_arm.setObjectName(_fromUtf8("btn_init_arm"))
        self.verticalLayout_11.addWidget(self.btn_init_arm)
        self.btn_sleep_arm = QtGui.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.btn_sleep_arm.setFont(font)
        self.btn_sleep_arm.setObjectName(_fromUtf8("btn_sleep_arm"))
        self.verticalLayout_11.addWidget(self.btn_sleep_arm)
        self.btn_torq_off = QtGui.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.btn_torq_off.setFont(font)
        self.btn_torq_off.setObjectName(_fromUtf8("btn_torq_off"))
        self.verticalLayout_11.addWidget(self.btn_torq_off)
        self.btn_torq_on = QtGui.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.btn_torq_on.setFont(font)
        self.btn_torq_on.setObjectName(_fromUtf8("btn_torq_on"))
        self.verticalLayout_11.addWidget(self.btn_torq_on)
        self.btn_calibrate = QtGui.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.btn_calibrate.setFont(font)
        self.btn_calibrate.setObjectName(_fromUtf8("btn_calibrate"))
        self.verticalLayout_11.addWidget(self.btn_calibrate)
        self.btn_task1 = QtGui.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.btn_task1.setFont(font)
        self.btn_task1.setObjectName(_fromUtf8("btn_task1"))
        self.verticalLayout_11.addWidget(self.btn_task1)
        self.btn_task2 = QtGui.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.btn_task2.setFont(font)
        self.btn_task2.setObjectName(_fromUtf8("btn_task2"))
        self.verticalLayout_11.addWidget(self.btn_task2)
        self.btn_task3 = QtGui.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.btn_task3.setFont(font)
        self.btn_task3.setObjectName(_fromUtf8("btn_task3"))
        self.verticalLayout_11.addWidget(self.btn_task3)
        self.btn_task4 = QtGui.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.btn_task4.setFont(font)
        self.btn_task4.setObjectName(_fromUtf8("btn_task4"))
        self.verticalLayout_11.addWidget(self.btn_task4)
        self.btn_task5 = QtGui.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.btn_task5.setFont(font)
        self.btn_task5.setObjectName(_fromUtf8("btn_task5"))
        self.verticalLayout_11.addWidget(self.btn_task5)
        spacerItem6 = QtGui.QSpacerItem(20, 40, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding)
        self.verticalLayout_11.addItem(spacerItem6)
        self.horizontalLayout_13.addLayout(self.verticalLayout_11)
        self.verticalLayout_14.addLayout(self.horizontalLayout_13)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtGui.QStatusBar(MainWindow)
        self.statusbar.setObjectName(_fromUtf8("statusbar"))
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow", None))
        self.JointCoordLabel.setText(_translate("MainWindow", "Joint Coordinates", None))
        self.BLabel.setText(_translate("MainWindow", "B:", None))
        self.SLabel.setText(_translate("MainWindow", "S:", None))
        self.ELabel.setText(_translate("MainWindow", "E:", None))
        self.WALabel.setText(_translate("MainWindow", "WA:", None))
        self.WRLabel.setText(_translate("MainWindow", "WR:", None))
        self.rdoutBaseJC.setText(_translate("MainWindow", "0", None))
        self.rdoutShoulderJC.setText(_translate("MainWindow", "0", None))
        self.rdoutElbowJC.setText(_translate("MainWindow", "0", None))
        self.rdoutWristAJC.setText(_translate("MainWindow", "0", None))
        self.rdoutWristRJC.setText(_translate("MainWindow", "0", None))
        self.WorldCoordLabel.setText(_translate("MainWindow", "End Effector Location", None))
        self.XLabel.setText(_translate("MainWindow", "X:", None))
        self.YLabel.setText(_translate("MainWindow", "Y:", None))
        self.ZLabel.setText(_translate("MainWindow", "Z:", None))
        self.PhiLabel.setText(_translate("MainWindow", "Phi:", None))
        self.ThetaLabel.setText(_translate("MainWindow", "Theta:", None))
        self.PsiLabel.setText(_translate("MainWindow", "Psi:", None))
        self.rdoutX.setText(_translate("MainWindow", "0", None))
        self.rdoutY.setText(_translate("MainWindow", "0", None))
        self.rdoutZ.setText(_translate("MainWindow", "0", None))
        self.rdoutPhi.setText(_translate("MainWindow", "0", None))
        self.rdoutTheta.setText(_translate("MainWindow", "0", None))
        self.rdoutPsi.setText(_translate("MainWindow", "0", None))
        self.btnUser1.setText(_translate("MainWindow", "USER 1", None))
        self.btnUser2.setText(_translate("MainWindow", "USER 2", None))
        self.btnUser3.setText(_translate("MainWindow", "USER 3", None))
        self.btnUser4.setText(_translate("MainWindow", "USER 4", None))
        self.btnUser5.setText(_translate("MainWindow", "USER 5", None))
        self.btnUser6.setText(_translate("MainWindow", "USER 6", None))
        self.btnUser7.setText(_translate("MainWindow", "USER 7", None))
        self.btnUser8.setText(_translate("MainWindow", "USER 8", None))
        self.btnUser9.setText(_translate("MainWindow", "USER 9", None))
        self.btnUser10.setText(_translate("MainWindow", "USER 10", None))
        self.btnUser11.setText(_translate("MainWindow", "USER 11", None))
        self.btnUser12.setText(_translate("MainWindow", "USER 12", None))
        self.videoDisplay.setText(_translate("MainWindow", "Video Display", None))
        self.chk_directcontrol.setText(_translate("MainWindow", "Direct Control", None))
        self.radioVideo.setText(_translate("MainWindow", "Video", None))
        self.radioDepth.setText(_translate("MainWindow", "Depth", None))
        self.radioUsr1.setText(_translate("MainWindow", "User 1", None))
        self.radioUsr2.setText(_translate("MainWindow", "User 2", None))
        self.PixelCoordLabel.setText(_translate("MainWindow", "Mouse Coordinates:", None))
        self.rdoutMousePixels.setText(_translate("MainWindow", "(U,V,D)", None))
        self.PixelCoordLabel_2.setText(_translate("MainWindow", "World Coordinates [mm]:", None))
        self.rdoutMouseWorld.setText(_translate("MainWindow", "(X,Y,Z)", None))
        self.BLabelS.setText(_translate("MainWindow", "Base", None))
        self.rdoutBase.setText(_translate("MainWindow", "0", None))
        self.SLabelS.setText(_translate("MainWindow", "Shoulder", None))
        self.rdoutShoulder.setText(_translate("MainWindow", "0", None))
        self.ELabelS.setText(_translate("MainWindow", "Elbow", None))
        self.rdoutElbow.setText(_translate("MainWindow", "0", None))
        self.WALabelS.setText(_translate("MainWindow", "Wrist Angle", None))
        self.rdoutWristA.setText(_translate("MainWindow", "0", None))
        self.WRLabelS.setText(_translate("MainWindow", "Wrist Rotate", None))
        self.rdoutWristR.setText(_translate("MainWindow", "0", None))
        self.MoveTimeLabel.setText(_translate("MainWindow", "MoveTime", None))
        self.rdoutMoveTime.setText(_translate("MainWindow", "0", None))
        self.AccelTimeLabel.setText(_translate("MainWindow", "AccelTime", None))
        self.rdoutAccelTime.setText(_translate("MainWindow", "0", None))
        self.label_3.setText(_translate("MainWindow", "Status:", None))
        self.rdoutStatus.setText(_translate("MainWindow", "Waiting for Inputs", None))
        self.btn_estop.setText(_translate("MainWindow", "EMERGENCY STOP", None))
        self.btn_init_arm.setText(_translate("MainWindow", "INITIALIZE ARM", None))
        self.btn_sleep_arm.setText(_translate("MainWindow", "SLEEP ARM", None))
        self.btn_torq_off.setText(_translate("MainWindow", "TORQUE OFF", None))
        self.btn_torq_on.setText(_translate("MainWindow", "TORQUE ON", None))
        self.btn_calibrate.setText(_translate("MainWindow", "CALIBRATE", None))
        self.btn_task1.setText(_translate("MainWindow", "TASK 1", None))
        self.btn_task2.setText(_translate("MainWindow", "TASK 2", None))
        self.btn_task3.setText(_translate("MainWindow", "TASK 3", None))
        self.btn_task4.setText(_translate("MainWindow", "TASK 4", None))
        self.btn_task5.setText(_translate("MainWindow", "TASK 5", None))

