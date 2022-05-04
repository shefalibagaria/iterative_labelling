import sys
from PyQt5.QtWidgets import QLabel, QMainWindow, QApplication, QWidget, QVBoxLayout, QHBoxLayout, QMenu, QAction, QPushButton, QSpinBox, QComboBox
from PyQt5.QtGui import QPixmap, QPainter, QBrush, QPen, QColor, QPolygon
from PyQt5.QtCore import Qt
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.path import Path

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Net Trainer 2 Electric Boogaloo')
        self.painterWidget = Painter(self)
        self.setCentralWidget(self.painterWidget)
        self.setGeometry(30,30,self.painterWidget.image.width(),self.painterWidget.image.height())

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
        filePath = 'app/label.jpg'
        label_map = np.argmax(self.painterWidget.labels, axis=0)
        plt.imsave(filePath, label_map, cmap='gray')

class Painter(QWidget):
    def __init__(self, parent):
        super(Painter, self).__init__(parent)
        self.parent = parent
        
        # Set up canvas
        # self.label = QLabel(self)
        self.image = QPixmap('datasets/Figure_1.png')
        # self.label.setPixmap(self.image)
        # self.resize(self.image.width(), self.image.height())

        self.shape = 'poly'
        self.polypts = []
        self.old_polys = []
        self.begin = None
        self.end = None

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
        self.drawStatusLabel = QLabel()
        self.drawStatusLabel.setFocusPolicy(Qt.NoFocus)
        self.drawStatusLabel.setText('not drawing')
        drawStatus = parent.addToolBar('&drawStatus')
        drawStatus.addWidget(self.drawStatusLabel)

        self.labels = np.zeros((self.n_classes+1,self.image.height(), self.image.width()))


    def paintEvent(self, event):
        qp = QPainter(self)
        self.resize(self.image.width(), self.image.height())
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
        if self.shape == 'poly':
            if len(self.polypts) == 0:
                self.polypts.append(self.begin)
            self.polypts.append(self.end)
            # self.setMouseTracking(True)
            self.drawStatusLabel.setText('drawing')
            self.update()

    # def mouseReleaseEvent(self, event):
    #     if self.shape != 'poly':
    #         self.last_x, self.last_y = None, None

    def mouseDoubleClickEvent(self, event):
        # ends current shape
        if self.shape == 'poly':
            if len(self.polypts) == 0:
                self.polypts.append(self.begin)
            self.polypts.append(self.end)
            self.setMouseTracking(False)
            self.drawStatusLabel.setText('not drawing')
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
            self.labels.resize(self.n_classes+1,self.image.height(), self.image.width())

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
        self.labels[self.currentClass] += contained_pts.reshape(h, w)
        print(self.labels[self.currentClass])

    

app = QApplication(sys.argv)
window = MainWindow()
app.exec_()
