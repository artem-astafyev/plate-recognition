from PyQt4 import QtCore
from PyQt4.QtCore import QObject

import helper as hp


class Capture(QObject):
    plate_captured = QtCore.pyqtSignal(dict)

    def __init__(self, parent=None):
        QObject.__init__(self, parent)

    @QtCore.pyqtSlot(str)
    def start_capturing(self, dir_in):
        paths = hp.get_paths(dir_in)
        for num, path in enumerate(paths, start=1):
            meta = {'id': num, 'plate': hp.get_image(dir_in, path)}
            self.plate_captured.emit(meta)
            print path


class Show(QObject):
    def __init__(self, parent=None):
        QObject.__init__(self, parent)

    @QtCore.pyqtSlot(dict)
    def show(self, meta):
        hp.show(meta['plate'])
        print meta['id']


def example():
    capture = Capture()
    show = Show()

    capture.plate_captured.connect(show.show)
    capture.start_capturing(dir_in="plates/present")


if __name__ == "__main__":
    pass
