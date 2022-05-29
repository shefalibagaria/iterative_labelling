import sys
from PyQt5.QtWidgets import QLabel, QMainWindow, QApplication, QWidget, QShortcut, QPushButton, QSpinBox, QComboBox, QLineEdit, QCheckBox, QFileDialog
from PyQt5.QtGui import QPixmap, QPainter, QBrush, QPen, QColor, QPolygon, QCursor, QIcon, QKeySequence, QKeyEvent
from PyQt5.QtCore import Qt, QThread, QEvent
from torch import true_divide
from config import Config
from src.train_worker import TrainWorker
import src.util as util
from src.networks import make_nets
from windows import Visualiser, Options
import numpy as np
import time
import json
from matplotlib import pyplot as plt
from matplotlib.path import Path


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Labelling')
        self.painterWidget = Painter(self)
        self.setCentralWidget(self.painterWidget)
        self.setGeometry(30,30,self.painterWidget.image.width(),self.painterWidget.image.height())

        #save map
        # mainMenu = self.menuBar()
        # fileMenu = mainMenu.addMenu("file")
        # saveAction = QAction("save", self)
        # saveAction.setShortcut("Ctrl+S")
        # fileMenu.addAction(saveAction)
        # saveAction.triggered.connect(self.save)
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

        self.setFocusPolicy(Qt.StrongFocus)

        # init other windows
        self.visualise_win = None
        self.o = Options()
    
        self.shape = 'poly'
        self.polypts = []
        self.prev_polys = []
        self.begin = None
        self.end = None
        self.label_time = 0
        self.drawing = False
        self.t1 = 0
        self.t2 = 0

        self.training = 0
        self.training_iter = 0

        # Current Options
        self.epochs = 5000
        self.max_time = 120
        self.n_gpu = 0
        self.overwriteCheck = False

        # Select image file
        self.fileButton = QPushButton('File',self)
        self.fileButton.setFocusPolicy(Qt.NoFocus)
        self.fileButton.clicked.connect(self.browseImage)
        file = parent.addToolBar('&fileButton')
        file.addWidget(self.fileButton)

        # Select no. of classes in image
        self.nClassesSpinBox = QSpinBox()
        self.nClassesSpinBox.setFocusPolicy(Qt.NoFocus)
        self.nClassesSpinBox.setMinimum(1)
        self.nClassesSpinBox.setMaximum(4)
        self.nClassesSpinBox.setValue(2)
        self.nClassesSpinBox.valueChanged.connect(self.nValueChange)
        self.nClassesSpinBox.setToolTip('number of classes in your image')
        nClasses = parent.addToolBar('&nClasses')
        nClasses.addWidget(self.nClassesSpinBox)
        self.n_classes = self.nClassesSpinBox.value()

        # Select which class currently being labelled
        self.classComboBox = QComboBox()
        self.classComboBox.setFocusPolicy(Qt.NoFocus)
        self.classComboBox.addItems([str(i+1) for i in range(self.n_classes)])
        self.classComboBox.activated.connect(self.classChanged)
        self.classComboBox.setToolTip('select class to label')
        classSelect = parent.addToolBar('&classSelect')
        classSelect.addWidget(self.classComboBox)

        self.prevClasses = []
        self.currentClass = 1

        # Create polygon dictionary
        self.old_polys = {i+1 : [] for i in range(self.n_classes)}

        # Undo last poly
        self.undoShortcut = QShortcut(QKeySequence("Ctrl+Z"), self)
        self.undoShortcut.activated.connect(self.onUndo)

        # Run tag
        self.tagLineEdit = QLineEdit(self)
        self.tagLineEdit.setText('run')
        self.tagLineEdit.setFocusPolicy(Qt.ClickFocus)
        # self.tagLineEdit.focusOutEvent(QEvent())
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

        # Select view
        self.displayComboBox = QComboBox(self)
        self.displayComboBox.setFocusPolicy(Qt.NoFocus)
        self.displayComboBox.addItems(['Input','Prediction','Confidence'])
        self.displayComboBox.activated.connect(self.displayChanged)
        disp_select = parent.addToolBar('&displaySelect')
        disp_select.addWidget(self.displayComboBox)

        self.display = 'Input'

        # show visualiser
        self.visualiseButton = QPushButton(self)
        self.visualiseButton.setFocusPolicy(Qt.NoFocus)
        self.visualiseButton.setIcon(QIcon('icons/eye.png'))
        self.visualiseButton.clicked.connect(self.visualiseWindow)

        # show options
        self.optionsButton = QPushButton(self)
        self.optionsButton.setFocusPolicy(Qt.NoFocus)
        self.optionsButton.setIcon(QIcon('icons/settings.png'))
        self.optionsButton.clicked.connect(self.optionsWindow)

        options = parent.addToolBar('&options')
        options.addWidget(self.optionsButton)
        options.addWidget(self.visualiseButton)

        self.labels = np.zeros((self.n_classes,self.image.height(),self.image.width()))

        self.labelling_data = {
            'times' : [],
            'training iter' : []
        }
        

    def browseImage(self):
        fname = QFileDialog.getOpenFileName(self, 'Open File', 'c\\', 'Image files ( *.png *.jpg *.jpeg)')
        self.datapath = fname[0]
        self.image = QPixmap(self.datapath)
        self.resize(self.image.width(), self.image.height())
        self.parent.resize(self.image.width(), self.image.height())
        self.cursorLabel.resize(self.image.width(), self.image.height())
        self.labels = np.zeros((self.n_classes,self.image.height(),self.image.width()))
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
        if not self.drawing:
            self.t1 = time.time()
        self.drawing = True
        self.begin = event.pos()
        self.end = event.pos()
        self.cursorLabel.setCursor(QCursor(Qt.PointingHandCursor))
        if self.shape == 'poly':
            if len(self.polypts) == 0:
                self.polypts.append(self.begin)
            self.polypts.append(self.end)
            self.update()

    def mouseDoubleClickEvent(self, event):
        # ends current shape
        self.cursorLabel.setCursor(QCursor(Qt.ArrowCursor))
        self.t2 = time.time()
        self.label_time += self.t2-self.t1
        if self.shape == 'poly':
            if len(self.polypts) == 0:
                self.polypts.append(self.begin)
            self.polypts.append(self.end)
            self.setMouseTracking(False)
            self.old_polys[self.currentClass].append(self.polypts)
            self.updateLabels(self.polypts)
            self.prev_polys.append(self.polypts)
            self.prevClasses.append(self.currentClass)
            self.polypts = []
            self.update()
        self.drawing = False       

    def onUndo(self):
        try:
            self.old_polys[self.prevClasses[-1]].remove(self.prev_polys[-1])
            self.prevClasses.pop()
            self.prev_polys.pop()
            self.update()
        except:
            pass

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
        self.training_iter += 1

        tag = self.tagLineEdit.text()
        exists = util.check_exist(tag)
        overwrite = True
        if exists and self.o.overwrite == False:
            overwrite = False

        util.initialise_folders(tag, overwrite)

        c = Config(tag)
        self.temp_path = c.path+'/temp'
        self.data_path = c.path+'/data'
        if overwrite:
            with open(self.data_path+'/label_data.json', 'w', encoding='utf-8') as fp:
                json.dump(self.labelling_data, fp, sort_keys=True, indent=4)
        else:
            with open(self.data_path+'/label_data.json', 'r') as fp:
                self.labelling_data = json.load(fp)

        # Update config based on options
        c.data_path = self.datapath
        c.n_phases = self.n_classes
        c.f[-1] = c.n_phases
        c.num_epochs = self.o.epochs
        c.ngpu = self.o.n_gpu
        c.update_device()
        offline = self.o.offline
        max_time = self.o.max_time
        
        net = make_nets(c, overwrite, self.training)

        # 1: Create worker class (TrainWorker)
        # 2: Create QThread object
        self.thread = QThread()
        # 3: Create worker object
        self.worker = TrainWorker(c, self.labels, self.temp_path, self.data_path, net, max_time, overwrite, offline)
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
        self.labelling_data['times'].append(self.label_time)
        self.labelling_data['training iter'].append(self.training_iter)
        with open(self.data_path+'/label_data.json', 'w', encoding='utf-8') as fp:
            json.dump(self.labelling_data, fp, sort_keys=True, indent=4)

    def progress(self, epoch, running_loss):
        # self.stepLabel.setText(f'epoch: {epoch}, running loss: {running_loss:.4f}')
        text = f'epoch: {epoch}, running loss: {running_loss:.4f}'
        if self.visualise_win is not None:
            self.visualise_win.updateImage(True, text)
        self.displayChanged()
    
    def displayChanged(self):
        self.display = self.displayComboBox.currentText()
        # try:
        if self.display == 'Input':
            self.image = QPixmap(self.datapath)
        if self.temp_path != '':
            if self.display == 'Prediction':
                self.image = QPixmap(self.temp_path+'/predictionblend.png')
            if self.display == 'Confidence':
                self.image = QPixmap(self.temp_path+'/confidenceblend.png')
        # except:
        #     pass
        # finally:
        #     print('failed')
        #     self.image = QPixmap(self.datapath)
        self.update()

    def visualiseWindow(self):
        if self.visualise_win is None:
            self.visualise_win = Visualiser(self.temp_path)
        self.visualise_win.show()
    
    def optionsWindow(self):
        if self.o.isVisible():
            self.o.hide()
        else:
            self.o.show()
            self.o.activateWindow()
            

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    app.exec_()
    
if __name__ == '__main__':
    main()
