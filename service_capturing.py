import Queue
import random

from PyQt4 import QtCore
from PyQt4.QtCore import QObject

import helper as hp


class Capture(QtCore.QThread):
    plate_captured = QtCore.pyqtSignal(dict)

    __queue = Queue.Queue(-1)

    def __init__(self, parent=None):
        super(Capture, self).__init__(parent)

    @QtCore.pyqtSlot(str)
    def start_capturing(self, dir_in):
        self.__queue.put(dir_in)
        print 'get new dir'

    def run(self):
        while True:
            if not self.__queue.empty():
                print 'process dir'
                dir_in = self.__queue.get()
                paths = hp.get_paths(dir_in)
                for num, path in enumerate(paths, start=1):
                    meta = {'id': random.randint(1000, 1000000000), 'plate': hp.get_image(dir_in, path)}
                    self.plate_captured.emit(meta)
                    # print path
                self.__queue.task_done()


class Show(QObject):
    def __init__(self, parent=None):
        QObject.__init__(self, parent)

    @QtCore.pyqtSlot(dict)
    def show(self, meta):
        hp.show(meta['plate'])
        #print meta['id']


def example():
    capture = Capture()
    show = Show()

    capture.plate_captured.connect(show.show)
    capture.start_capturing(dir_in="plates/present")


if __name__ == "__main__":
    pass
