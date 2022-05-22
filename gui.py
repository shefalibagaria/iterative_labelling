import sys
from PyQt5.QtWidgets import QLabel, QMainWindow, QApplication, QWidget, QAction, QPushButton, QSpinBox, QComboBox, QLineEdit, QCheckBox, QFileDialog
from PyQt5.QtGui import QPixmap, QPainter, QBrush, QPen, QColor, QPolygon, QCursor
from PyQt5.QtCore import Qt, QThread
from torch import true_divide
from config import Config
from src.train_worker import TrainWorker
import src.util as util
from src.networks import make_nets
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.path import Path


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Segmentatron 9000')
        self.painterWidget = Painter(self)
        self.setCentralWidget(self.painterWidget)
        self.setGeometry(30,30,self.painterWidget.image.width(),self.painterWidget.image.height()+30)

        #save map
        mainMenu = self.menuBar()
        fileMenu = mainMenu.addMenu("file")
        saveAction = QAction("save", self)
        saveAction.setShortcut("Ctrl+S")
        fileMenu.addAction(saveAction)
        saveAction.triggered.connect(self.save)
        self.show()
    
    def save(self):
        print('....saving')
        filePath = 'data/label.png'
        label_map = self.painterWidget.labels.transpose((1,2,0))
        padding = np.zeros((self.painterWidget.image.height(),self.painterWidget.image.width(), 4-label_map.shape[2]))
        labels = np.concatenate((label_map, padding), axis = 2)
        alpha = 0.5
        # labels[:,:,3] = alpha*np.amax(labels, axis=2)
        labels = 0.5*labels
        input_img = plt.imread(self.painterWidget.datapath)
        y = np.ones(input_img.shape)
        mask = np.where(labels==0.5, labels, y)
        final_img = input_img*mask
        
        plt.imsave(filePath, final_img)

class Painter(QWidget):
    def __init__(self, parent):
        super(Painter, self).__init__(parent)
        self.parent = parent
        self.datapath = 'data/nmc_cathode.png'
        self.temp_path = ''
        self.image = QPixmap(self.datapath)
        self.cursorLabel = QLabel(self)
        self.visualise_win = None
    
        self.shape = 'poly'
        self.polypts = []
        self.old_polys = []
        self.begin = None
        self.end = None

        self.training = False

        # Select no. of classes in image
        self.nClassesSpinBox = QSpinBox()
        self.nClassesSpinBox.setFocusPolicy(Qt.NoFocus)
        self.nClassesSpinBox.setMinimum(1)
        self.nClassesSpinBox.setMaximum(4)
        self.nClassesSpinBox.setValue(2)
        self.nClassesSpinBox.valueChanged.connect(self.nValueChange)
        nClasses = parent.addToolBar('&nClasses')
        nClasses.addWidget(self.nClassesSpinBox)
        self.n_classes = self.nClassesSpinBox.value()

        # Select which class currently being labelled
        self.classComboBox = QComboBox()
        self.classComboBox.setFocusPolicy(Qt.NoFocus)
        self.classComboBox.addItems([str(i+1) for i in range(self.n_classes)])
        self.classComboBox.activated.connect(self.classChanged)
        classSelect = parent.addToolBar('&classSelect')
        classSelect.addWidget(self.classComboBox)

        self.currentClass = 1

        # Create polygon dictionary
        self.old_polys = {i+1 : [] for i in range(self.n_classes)}

        # Drawing status label
        # self.drawStatusLabel = QLabel(self)
        # self.drawStatusLabel.setFocusPolicy(Qt.NoFocus)
        # self.drawStatusLabel.setText('not drawing')
        # drawStatus = parent.addToolBar('&drawStatus')
        # drawStatus.addWidget(self.drawStatusLabel)

        # Run tag
        self.tagLineEdit = QLineEdit(self)
        self.tagLineEdit.setText('run')
        tagInput = parent.addToolBar('&tagInput')
        tagInput.addWidget(self.tagLineEdit)

        # stop train
        self.stopTrain = QPushButton('Stop', self)
        self.stopTrain.hide()

        # train button
        self.trainButton = QPushButton('Train',self)
        self.trainButton.setFocusPolicy(Qt.NoFocus)
        self.trainButton.clicked.connect(self.onTrainClick)
        train = parent.addToolBar('&trainButton')
        train.addWidget(self.trainButton)

        self.overwriteCheckbox = QCheckBox('overwrite',self)
        self.overwriteCheckbox.setFocusPolicy(Qt.NoFocus)
        overwriteCheck = parent.addToolBar('&overwriteCheck')
        overwriteCheck.addWidget(self.overwriteCheckbox)

        self.fileButton = QPushButton('File',self)
        self.fileButton.setFocusPolicy(Qt.NoFocus)
        self.fileButton.clicked.connect(self.browseImage)
        file = parent.addToolBar('&fileButton')
        file.addWidget(self.fileButton)

        # Select view
        self.displayComboBox = QComboBox(self)
        self.displayComboBox.setFocusPolicy(Qt.NoFocus)
        self.displayComboBox.addItems(['Input','Prediction','Confidence'])
        self.displayComboBox.activated.connect(self.displayChanged)
        disp_select = parent.addToolBar('&displaySelect')
        disp_select.addWidget(self.displayComboBox)

        self.display = 'Input'

        # self.stepLabel = QLabel('step label')
        # self.stepLabel.setFocusPolicy(Qt.NoFocus)
        # label = parent.addToolBar('&stepLabel')
        # label.addWidget(self.stepLabel)

        self.labels = np.zeros((self.n_classes,self.image.height(),self.image.width()))

    def browseImage(self):
        fname = QFileDialog.getOpenFileName(self, 'Open File', 'c\\', 'Image files ( *.png *.jpg *.jpeg')
        self.datapath = fname[0]
        self.image = QPixmap(self.datapath)
        self.resize(self.image.width(), self.image.height())
        self.parent.resize(self.image.width(), self.image.height())
        self.cursorLabel.resize(self.image.width(), self.image.height())
        self.update()


    def paintEvent(self, event):
        qp = QPainter(self)
        self.resize(self.image.width(), self.image.height())
        self.cursorLabel.resize(self.image.width(), self.image.height())
        qp.drawPixmap(self.rect(), self.image)
        br = QBrush(QColor(200,10,10,30))
        pen = QPen(Qt.red, 1)

        pen_colours = [Qt.red, Qt.blue, Qt.green, Qt.magenta]
        brush_colours = [QColor(200,10,10,30), QColor(10,10,200,30), QColor(10,200,10,30), QColor(200,10,200,30)]

        if self.shape == 'poly':
            for c in self.old_polys:
                pen.setColor(pen_colours[c-1])
                br.setColor(brush_colours[c-1])
                qp.setBrush(br)
                qp.setPen(pen)
                for poly in self.old_polys[c]:
                    qp.drawPolygon(QPolygon(poly))

            pen.setColor(pen_colours[self.currentClass-1])
            br.setColor(brush_colours[self.currentClass-1])
            qp.setBrush(br)
            qp.setPen(pen)
            qp.drawPolygon(QPolygon(self.polypts))
    
    def mouseMoveEvent(self, event):
        self.end = event.pos()
        if self.shape == 'poly':
            try:
                self.polypts.append(self.end)
                self.update()
            except:
                pass

    def mousePressEvent(self, event):
        self.begin = event.pos()
        self.end = event.pos()
        self.cursorLabel.setCursor(QCursor(Qt.CrossCursor))
        if self.shape == 'poly':
            if len(self.polypts) == 0:
                self.polypts.append(self.begin)
            self.polypts.append(self.end)
            # self.setMouseTracking(True)
            # self.drawStatusLabel.setText('drawing')
            self.update()

    # def mouseReleaseEvent(self, event):
    #     if self.shape != 'poly':
    #         self.last_x, self.last_y = None, None

    def mouseDoubleClickEvent(self, event):
        # ends current shape
        self.cursorLabel.setCursor(QCursor(Qt.ArrowCursor))
        if self.shape == 'poly':
            if len(self.polypts) == 0:
                self.polypts.append(self.begin)
            self.polypts.append(self.end)
            self.setMouseTracking(False)
            # self.drawStatusLabel.setText('not drawing')
            self.old_polys[self.currentClass].append(self.polypts)
            self.updateLabels(self.polypts)
            self.polypts = []
            self.update()       

    def nValueChange(self):
        self.n_classes = self.nClassesSpinBox.value()
        self.classComboBox.clear()
        self.classComboBox.addItems([str(i+1) for i in range(self.n_classes)])
        if self.n_classes > len(self.old_polys):
            for i in range(len(self.old_polys),self.n_classes):
                self.old_polys[i+1] = []
            self.labels.resize(self.n_classes,self.image.height(), self.image.width())

    def classChanged(self):
        self.currentClass = int(self.classComboBox.currentText())

    def updateLabels(self, poly):
        new_poly = [[point.x(), point.y()] for point in poly]
        w = self.image.width()
        h = self.image.height()
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        x, y = x.flatten(), y.flatten()
        grid = np.vstack((x,y)).T
        p = Path(new_poly)
        contained_pts = p.contains_points(grid)
        self.labels[self.currentClass-1] += contained_pts.reshape(h, w)

    def onTrainClick(self):
        self.training = True
        self.trainButton.hide()
        self.stopTrain.show()

        tag = self.tagLineEdit.text()
        exists = util.check_exist(tag)
        overwrite = True
        if exists and self.overwriteCheckbox.isChecked() == False:
            overwrite = False

        util.initialise_folders(tag, overwrite)
        c = Config(tag)
        self.temp_path = c.path+'/temp'
        c.data_path = self.datapath
        c.n_phases = self.n_classes
        c.f[-1] = c.n_phases

        net = make_nets(c, overwrite, self.training)

        # 1: Create worker class (TrainWorker)
        # 2: Create QThread object
        self.thread = QThread()
        # 3: Create worker object
        self.worker = TrainWorker(c, self.labels,self.temp_path, net, overwrite)
        # 4: Move worker to thread
        self.worker.moveToThread(self.thread)
        # 5: Connect Signals and Slots
        self.thread.started.connect(self.worker.train)
        self.stopTrain.clicked.connect(lambda: self.worker.stop())
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker.finished.connect(self.stop_train)
        self.thread.finished.connect(self.thread.deleteLater)
        self.worker.progress.connect(self.progress)
        # 6: Start thread
        self.thread.start()

        self.visualiseWindow()

    def stop_train(self):
        self.training = False
        self.stopTrain.hide()
        self.trainButton.show()

    def progress(self, epoch, running_loss):
        # self.stepLabel.setText(f'epoch: {epoch}, running loss: {running_loss:.4f}')
        text = f'epoch: {epoch}, running loss: {running_loss:.4f}'
        if self.visualise_win is not None:
            self.visualise_win.updateImage('data/temp/confidence_prediction.png', text)
        self.displayChanged()
    
    def displayChanged(self):
        self.display = self.displayComboBox.currentText()
        try:
            if self.display == 'Input':
                self.image = QPixmap(self.datapath)
            if self.display == 'Prediction':
                self.image = QPixmap(self.temp_path+'/prediction_blend.png')
            if self.display == 'Confidence':
                self.image = QPixmap(self.temp_path+'/confidence_blend.png')
        except:
            print('Image read failed')
        self.update()

    def visualiseWindow(self):
        if self.visualise_win is None:
            self.visualise_win = Visualiser()
            self.visualise_win.show()


class Visualiser(QMainWindow):
    def __init__(self):
        super().__init__()
        self.trainingBegin = False

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
            self.image = QPixmap('data/temp/confidence_prediction.png')
        else:
            self.image = QPixmap('data/temp/prediction.png')
        self.label.setPixmap(self.image)
        self.label.resize(self.image.width(), self.image.height())
        self.setGeometry(30+self.image.width(), 30, self.image.width(), self.image.height()+30)

    def displayChanged(self):
        if self.confidenceOverlay.isChecked():
            self.image = QPixmap('data/temp/confidence_prediction.png')
        else:
            self.image = QPixmap('data/temp/prediction.png')
        self.label.setPixmap(self.image)
            

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    app.exec_()
    
if __name__ == '__main__':
    main()
