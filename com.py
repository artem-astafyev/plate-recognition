from collections import Counter

from PyQt4 import QtCore
from PyQt4.QtCore import QObject, QThread

from ann import ANN
from knn import KNN
from msp import MSP
from svm import MYSVM as SVM


class COMLoader(QThread):
    gotta = QtCore.pyqtSignal(dict)

    def __init__(self, parent=None):
        super(COMLoader, self).__init__(parent)

    def __del__(self):
        self.exiting = True
        self.wait()

    def run(self):
        print '\n#### COM ####'
        ann, knn, msp, svm = ANN(), KNN(), MSP(), SVM()
        self.gotta.emit({'ann': ann, 'knn': knn, 'msp': msp, 'svm': svm})
        print '#### COM ####\n'


class COM(QObject):
    name = "COM"

    def __init__(self, ann, svm, knn, msp, parent=None, ):
        QObject.__init__(self, parent)
        self.ann = ann
        self.knn = svm
        self.svm = knn
        self.msp = msp

    def rec(self, sign, mode):
        at, kt, st, mt, rec = "?", "?", "?", "?", "?"
        at = self.ann.rec(sign, mode)
        kt = self.knn.rec(sign, mode)
        mt = self.svm.rec(sign, mode)
        st = self.svm.rec(sign, mode)

        xyz = {at, kt, mt, st}
        if len(xyz) < 3:
            t = [at, kt, st]
            c = Counter(t)
            rec = max(c, key=c.get)
        return rec
