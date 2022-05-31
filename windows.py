import sys
from PyQt5.QtWidgets import QLabel, QMainWindow, QApplication, QWidget, QPushButton, QSpinBox, QComboBox, QLineEdit, QCheckBox, QVBoxLayout, QHBoxLayout
from PyQt5.QtGui import QPixmap, QPainter, QBrush, QPen, QColor, QIntValidator
from PyQt5.QtCore import Qt, QThread

class Visualiser(QMainWindow):
    def __init__(self, path):
        super().__init__()
        self.trainingBegin = False
        self.path = path
        self.label = QLabel(self)
        self.label.move(0,30)

        self.stepLabel = QLabel('step label',self)
        self.stepLabel.setFocusPolicy(Qt.NoFocus)
        # self.stepLabel.resize(100, 50)
        label = self.addToolBar('&stepLabel')
        label.setFixedHeight(30)
        label.addWidget(self.stepLabel)

        # select view
        self.confidenceOverlay = QCheckBox('Confidence',self)
        self.confidenceOverlay.setFocusPolicy(Qt.NoFocus)
        self.confidenceOverlay.stateChanged.connect(self.displayChanged)
        disp_select = self.addToolBar('&displaySelect')
        disp_select.addWidget(self.confidenceOverlay)


    def updateImage(self, training, text):
        self.setWindowTitle('Network Prediction')
        self.stepLabel.setText(text)
        # self.update()
        self.trainingBegin = True
        if self.confidenceOverlay.isChecked():
            self.image = QPixmap(self.path+'/confidenceprediction.png')
        else:
            self.image = QPixmap(self.path+'/prediction.png')
        self.label.setPixmap(self.image)
        self.label.resize(self.image.width(), self.image.height())
        self.setGeometry(30+self.image.width(), 30, self.image.width(), self.image.height()+30)

    def displayChanged(self):
        if self.confidenceOverlay.isChecked():
            self.image = QPixmap(self.path+'/confidenceprediction.png')
        else:
            self.image = QPixmap(self.path+'/prediction.png')
        self.label.setPixmap(self.image)

    # def cursorPos(self, x, y):
    #     qp = QPainter(self)
    #     self.resize(self.image.width(), self.image.height())
    #     # self.cursorLabel.resize(self.image.width(), self.image.height())
    #     qp.drawPixmap(self.rect(), self.image)
    #     br = QBrush(QColor(200,10,10,30))
    #     pen = QPen(Qt.yellow, 1)

    #     pen_colours = [Qt.red, Qt.blue, Qt.green, Qt.magenta]
    #     brush_colours = [QColor(200,10,10,30), QColor(10,10,200,30), QColor(10,200,10,30), QColor(200,10,200,30)]

    #     qp.drawLine(x, 0, x, self.image.height())
            

class Options(QWidget):
    def __init__(self):
        super().__init__()
        self.options_set = False
        self.epochs = 8000
        self.max_time = 300
        self.n_gpu = 0
        self.offline = True
        self.overwrite = False

        self.setWindowTitle("Options")

        layout = QVBoxLayout()
        self.onlyInt = QIntValidator()

        epochLayout = QHBoxLayout()
        self.epochLabel = QLabel(self)
        self.epochLabel.setText('epochs:')
        self.epochLineEdit = QLineEdit(self)
        self.epochLineEdit.setValidator(self.onlyInt)
        self.epochLineEdit.setText(str(self.epochs))
        self.epochLineEdit.textChanged.connect(self.epochChanged)
        epochLayout.addWidget(self.epochLabel)
        epochLayout.addWidget(self.epochLineEdit)
        layout.addLayout(epochLayout)

        timeLayout = QHBoxLayout()
        self.timeLabel = QLabel(self)
        self.timeLabel.setText('max runtime (s):')
        self.timeLineEdit = QLineEdit(self)
        self.timeLineEdit.setValidator(self.onlyInt)
        self.timeLineEdit.setText(str(self.max_time))
        self.timeLineEdit.textChanged.connect(self.timeChanged)
        timeLayout.addWidget(self.timeLabel)
        timeLayout.addWidget(self.timeLineEdit)
        layout.addLayout(timeLayout)

        n_gpuLayout = QHBoxLayout()
        self.n_gpuLabel = QLabel(self)
        self.n_gpuLabel.setText('no. GPUs:')
        self.n_gpuLineEdit = QLineEdit(self)
        self.n_gpuLineEdit.setValidator(self.onlyInt)
        self.n_gpuLineEdit.setText(str(self.n_gpu))
        self.n_gpuLineEdit.textChanged.connect(self.gpuChanged)
        n_gpuLayout.addWidget(self.n_gpuLabel)
        n_gpuLayout.addWidget(self.n_gpuLineEdit)
        layout.addLayout(n_gpuLayout)

        overwriteLayout = QHBoxLayout()
        self.overwriteLabel = QLabel(self)
        self.overwriteLabel.setText('overwrite:')
        self.overwriteCheckBox = QCheckBox(self)
        self.overwriteCheckBox.toggled.connect(self.overwriteToggled)
        overwriteLayout.addWidget(self.overwriteLabel)
        overwriteLayout.addWidget(self.overwriteCheckBox)
        layout.addLayout(overwriteLayout)

        wandbLayout = QHBoxLayout()
        self.wandbLabel = QLabel(self)
        self.wandbLabel.setText('log on WandB:')
        self.wandbCheckBox = QCheckBox(self)
        self.wandbCheckBox.toggled.connect(self.wandbToggled)
        wandbLayout.addWidget(self.wandbLabel)
        wandbLayout.addWidget(self.wandbCheckBox)
        layout.addLayout(wandbLayout)

        self.setLayout(layout)
    
    def epochChanged(self):
        self.epochs = int(self.epochLineEdit.text())
    
    def timeChanged(self):
        self.max_time = int(self.timeLineEdit.text())

    def gpuChanged(self):
        self.n_gpu = int(self.n_gpuLineEdit.text())
        
    def overwriteToggled(self):
        self.overwrite = self.overwriteCheckBox.isChecked()

    def wandbToggled(self):
        if self.wandbCheckBox.isChecked():
            self.offline = False
        else:
            self.offline = True

def main():
    app = QApplication(sys.argv)
    window = Options()
    app.exec_()
    
if __name__ == '__main__':
    main()