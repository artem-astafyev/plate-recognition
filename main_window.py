# coding=utf-8
import Queue
import os
import shutil
import sys
import time

import cv2
import numpy as np
from PyQt4 import QtCore
from PyQt4 import QtGui, uic
from PyQt4 import QtWebKit
from PyQt4.Qt import QUrl, QString, QSize

import helper as hp
from service_capturing import Capture
from service_recognition import Recognition


class MyWindow(QtGui.QMainWindow):
    __dir = 'plates/present'
    __zam1 = "file:///{0}/view/".format(os.getcwd().replace("\\", "/"))

    start_capturing = QtCore.pyqtSignal(str)

    def __init__(self):
        super(MyWindow, self).__init__()
        uic.loadUi('mybrow.ui', self)
        self.vieww = QtWebKit.QWebView(self)
        # self.vieww.load(QUrl("http://google.com"))
        self.__fill_page('loader.html')
        self.vieww.load(self.__get_uri('loader.html'))

        self.setCentralWidget(self.vieww)
        self.connect(self.my_run, QtCore.SIGNAL('triggered()'), QtCore.SLOT('run_capturing_service()'))
        self.connect(self.my_close, QtCore.SIGNAL('triggered()'), QtCore.SLOT('close()'))
        self.connect(self.my_open, QtCore.SIGNAL('triggered()'), QtCore.SLOT('show_dialog()'))

    def __get_uri(self, path):
        return QUrl("file:///{0}/view/{1}".format(os.getcwd().replace("\\", "/").replace("\\", "/"), path))

    def __fill_page(self, path):
        with open('view/tmp/' + path, 'r') as myfile:
            self.__main_tmp = myfile.read().replace('\n', '')
        self.__main_tmp = self.__main_tmp.format("", self.__zam1)
        with open("view/" + path, 'wb') as temp_file:
            temp_file.write(self.__main_tmp)

    def closeEvent(self, event):
        while os.path.exists(u'view/img'):
            shutil.rmtree(u'view/img')
        if not os.path.exists(u"view/img"):
            os.mkdir(u"view/img")
        QtGui.QMainWindow.closeEvent(self, event)

    @QtCore.pyqtSlot(str)
    def redraw(self, text):
        self.vieww.setHtml(text)

    @QtCore.pyqtSlot()
    def run_capturing_service(self):
        self.start_capturing.emit(self.__dir)

    @QtCore.pyqtSlot()
    def show_dialog(self):
        fd = QtGui.QFileDialog(self)
        fd.resize(QSize(640, 480))
        self.__dir = fd.getExistingDirectory(fd, QString(u"Открыть папку с изображениями:"),
                                             os.getcwd().replace("\\", "/"),
                                             QtGui.QFileDialog.DontResolveSymlinks)

    @QtCore.pyqtSlot()
    def unblock_window(self):
        self.my_run.setEnabled(True)
        self.my_close.setEnabled(True)
        self.my_open.setEnabled(True)
        self.__fill_page('start.html')
        self.vieww.load(self.__get_uri('start.html'))


class Presentation(QtCore.QThread):
    __counter = 0
    __zam0 = "{0}"
    __zam1 = "file:///{0}/view/".format(os.getcwd().replace("\\", "/"))

    redraw = QtCore.pyqtSignal(str)

    __queue = Queue.Queue(-1)

    def __init__(self, parent=None):
        super(Presentation, self).__init__(parent)
        while os.path.exists(u'view/img'):
            shutil.rmtree(u'view/img')
        if not os.path.exists(u"view/img"):
            os.mkdir(u"view/img")

        with open('view/tmp/main.html', 'r') as myfile:
            self.__main_tmp = myfile.read().replace('\n', '')
        self.__main_tmp = self.__main_tmp.format("{0}", self.__zam1)

    def __tworaws2onerows(self, plate):
        h, w = plate.shape
        double = np.zeros((h / 2, w * 2), dtype=np.uint8)
        double[0:h / 2, 0:w] = plate[0:h / 2, 0:w]
        double[0:h / 2, w:2 * w] = plate[h / 2:h, 0:w]
        return double

    def __line_present(self, meta, name):
        td_good = "<td>{0}</td>"
        td_wrong = """<td class="wrong">{0}</td>"""
        td_wrong_lighting = """<td class="wrong-lighting">{0}</td>"""

        row = ""
        if ''.join(meta[name]) != ''.join(meta['rec']):
            for counter, (x, y) in enumerate(zip(meta[name], meta['rec']), start=0):
                if x != y:
                    row += td_wrong_lighting.format(meta[name][counter]) + "\n"
                else:
                    row += td_wrong.format(meta[name][counter]) + "\n"
        else:
            row = '\n'.join([td_good.format(x) for x in meta[name]])

        return row

    def __make_template(self, meta):
        if not os.path.exists("view/img/{0}".format(meta['id'])):
            os.mkdir("view/img/{0}".format(meta['id']))
            os.mkdir("view/img/{0}/plate".format(meta['id']))
            os.mkdir("view/img/{0}/sym".format(meta['id']))

        if meta['rows'] == 2:
            meta['plate'] = self.__tworaws2onerows(meta['plate'])
            meta['blur'] = self.__tworaws2onerows(meta['blur'])
            meta['canny'] = self.__tworaws2onerows(meta['canny'])
            meta['thresh'] = self.__tworaws2onerows(meta['thresh'])
            meta['morph'] = self.__tworaws2onerows(meta['morph'])

        hp.write_image("view/img", "{0}/plate/blur.jpg".format(meta['id']), meta['blur'])
        hp.write_image("view/img", "{0}/plate/canny.jpg".format(meta['id']), meta['canny'])
        hp.write_image("view/img", "{0}/plate/morph.jpg".format(meta['id']), meta['morph'])
        hp.write_image("view/img", "{0}/plate/thresh.jpg".format(meta['id']), meta['thresh'])
        cv2.imwrite("view/img/{0}/plate/plate.jpg".format(meta['id']), meta['plate'])

        for counter, sign in enumerate(meta['signs'], start=1):
            cv2.imwrite('view/img/{0}/sym/{1}.jpg'.format(meta['id'], counter), sign)

        ann_row = self.__line_present(meta, 'ann')
        knn_row = self.__line_present(meta, 'knn')
        svm_row = self.__line_present(meta, 'svm')
        msp_row = self.__line_present(meta, 'msp')
        rec_row = self.__line_present(meta, 'rec')

        with open('view/tmp/item.html', 'r') as myfile:
            item = myfile.read().replace('\n', '')
        item = item.format(self.__zam0, self.__zam1, meta['id'], meta['result'], ann_row, knn_row, svm_row, msp_row,
                           rec_row)

        with open("view/img/{0}/{0}.html".format(meta['id']), 'wb') as temp_file:
            temp_file.write(item)

        self.__counter += 1

        return item

    def run(self):
        start = time.time()
        while True:
            if not self.__queue.empty():
                item = self.__make_template(self.__queue.get())
                self.__main_tmp = self.__main_tmp.format(item)

                with open("view/main-work.html", 'wb') as temp_file:
                    temp_file.write(self.__main_tmp)

                if time.time() - start > 0.65:
                    self.redraw.emit(self.__main_tmp.decode('utf-8'))
                    start = time.time()
                self.__queue.task_done()

    @QtCore.pyqtSlot(dict)
    def presentation(self, meta):
        self.__queue.put(meta)


if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)

    window = MyWindow()
    window.show()

    recognition = Recognition()
    present = Presentation()
    capture = Capture()

    window.start_capturing.connect(capture.start_capturing)
    recognition.plate_recognized.connect(present.presentation)
    recognition.unlock_interface.connect(window.unblock_window)
    capture.plate_captured.connect(recognition.process)
    present.redraw.connect(window.redraw)

    capture.start()
    recognition.start()
    present.start()

    sys.exit(app.exec_())
