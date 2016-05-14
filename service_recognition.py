from collections import Counter

import cv2
from PyQt4 import QtCore
from PyQt4.QtCore import QObject, SIGNAL

import helper as hp
from ann import ANN
from knn import KNN
from msp import MSP
from svm import MYSVM


class RecognitionLoader(QtCore.QThread):

    on_loaded = QtCore.pyqtSignal(dict)

    def __init__(self, parent=None):
        super(RecognitionLoader, self).__init__(parent)

    def __del__(self):
        self.exiting = True
        self.wait()

    def run(self):
        ann = ANN()
        knn = KNN()
        msp = MSP()
        svm = MYSVM()
        print 'run'

        self.on_loaded.emit({'ann':ann, 'knn':knn, 'msp' : msp, 'svm':svm})


class Recognition(QObject):
    plate_recognized = QtCore.pyqtSignal(dict)
    unlock_interface = QtCore.pyqtSignal()

    __rec_meta = {}
    __empty = ""

    def __init__(self, parent=None):
        QObject.__init__(self, parent)
        self.thread = RecognitionLoader()
        self.thread.on_loaded.connect(self.get_ocr)
        self.connect(self.thread, SIGNAL("finished()"), self.end)
        self.thread.start()

        self.__empty = cv2.imread("view/empty.jpg")
        self.__drop_rec_meta()

    def __drop_rec_meta(self):
        self.__rec_meta.clear()
        self.__rec_meta['signs'] = [self.__empty] * 8
        self.__rec_meta['ann'] = ["&nbsp;"] * 8
        self.__rec_meta['knn'] = ["&nbsp;"] * 8
        self.__rec_meta['svm'] = ["&nbsp;"] * 8
        self.__rec_meta['msp'] = ["&nbsp;"] * 8
        self.__rec_meta['rec'] = ["&nbsp;"] * 8
        self.__rec_meta['result'] = ""

    def __preprocessing(self, plate):
        morph_size = 4
        m1 = (2 * morph_size + 1, 2 * morph_size - 1)
        m2 = (morph_size, morph_size)
        h, w = plate.shape[:2]
        rows, size = (1, (230, 50)) if w / h > 2 else (2, (108, 78))
        gray = cv2.cvtColor(plate, cv2.COLOR_RGB2GRAY)
        res = cv2.resize(gray, size, interpolation=cv2.INTER_LINEAR)  # INTER_LINEAR
        blur = cv2.blur(res, (5, 5))
        canny = cv2.Canny(res, 50, 200)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, m1, m2)
        morph = cv2.morphologyEx(blur, cv2.MORPH_BLACKHAT, kernel)
        thr = cv2.adaptiveThreshold(morph, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 0)
        cv2.rectangle(thr, (0, 0), (size[0], size[1]), (255, 255, 255), 7)

        self.__rec_meta['plate'] = res
        self.__rec_meta['blur'] = blur
        self.__rec_meta['canny'] = canny
        self.__rec_meta['morph'] = morph
        self.__rec_meta['thresh'] = thr
        self.__rec_meta['rows'] = rows

        return rows, thr

    def __make_pattern(self, rects, rows):
        lr = len(rects)
        pattern = "p" * lr
        if rows == 2 and lr:
            t = len([r[4] for r in rects if r[4] < 0])
            b = len(rects) - t

            print "top, bot: ", t, b, lr

            if t == 3 and b == 5:
                pattern = "dddddsss"
            elif t == 2 and b == 4:
                pattern = "ddddss"
            elif t == 4 and b == 2:
                pattern = "ddddss"
            elif t == 4 and b == 3:
                pattern = "sdddsss"
            elif t == 3 and b == 4:
                x, y, w, h, z = rects[0]
                tavg = (rects[0][2] * rects[0][3] + rects[1][2] * rects[1][3] + rects[2][2] * rects[2][3]) / 3
                bavg = (rects[5][2] * rects[5][3] + rects[6][2] * rects[6][3]) / 2
                pattern = "dddddss" if tavg > bavg else "sssdddd"
        elif lr > 3:
            if lr == 6:
                pattern = "sdddss"
            elif lr == 8:
                pattern = "dddsssdd"
            elif lr == 4:
                pattern = "ddss"
            elif lr == 7:
                fl = rects[0][2] + rects[0][0]
                sr = rects[1][0]
                dist = rects[0][2] / 4
                if sr - fl > dist:
                    pattern = "sdddsss"
                else:
                    pattern = "dddssdd"
        return pattern

    def __segmentation(self, plate, rows):
        signs, rects = [], []
        h, w = plate.shape[:2]
        line = [255] * h
        for i in xrange(w):
            if sum(plate[0:h, i]) > 12100:
                plate[0:h, i] = line

        canny = plate.copy()
        im2, contours, hierarchy = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for cnt, hh in zip(contours, hierarchy[0]):
            x, y, w, h = cv2.boundingRect(cnt)
            if w * h == 0:
                break

            w, h = map(float, (w, h))
            koi = h / w < 3 and w / h < 1.05
            keff = 1500 > w * h > (350 - (rows - 1) * 100)
            z = x - 1000 if (y + h / 2) / plate.shape[0] < 0.5 and rows == 2 else x
            if koi and keff:
                img = hp.resize_to_small(plate[y:y + h, x: x + w])
                signs.append(img)
                rects.append([x, y, w, h, z])

        srt = zip(rects, signs)
        srt.sort(key=lambda x: x[0][4])
        rects, signs = [sr[0] for sr in srt], [sr[1] for sr in srt]
        map(hp.resize_to_small, signs)
        pattern = self.__make_pattern(rects, rows)
        return pattern, signs

    def __recognition(self, pat, sign, counter):

        mode = 'num' if pat == 'd' else 'sym'
        at, kt, st, mt, rec = "?", "?", "?", "?", "?"

        if mode == "num" or mode == "sym":
            at, _ = self.ann.rec(sign, mode)
            kt, _ = self.knn.rec(sign, mode)
            st, _ = self.svm.rec(sign, mode)
            mt, _ = self.msp.rec(sign, mode)

            xyz = {at, kt, st}
            if len(xyz) < 3:
                t = [at, kt, st]
                c = Counter(t)
                rec = max(c, key=c.get)

        self.__rec_meta['signs'][counter] = sign
        self.__rec_meta['ann'][counter] = at
        self.__rec_meta['knn'][counter] = kt
        self.__rec_meta['svm'][counter] = st
        self.__rec_meta['msp'][counter] = mt
        self.__rec_meta['rec'][counter] = rec

        return rec

    @QtCore.pyqtSlot(dict)
    def process(self, meta):
        print meta['id']
        self.__drop_rec_meta()
        self.__rec_meta['id'] = meta['id']

        rows, plate = self.__preprocessing(meta['plate'])
        pattern, signs = self.__segmentation(plate, rows)

        for counter, (pat, sign) in enumerate(zip(pattern, signs), start=0):
            self.__rec_meta['result'] += self.__recognition(pat, sign, counter)

        self.plate_recognized.emit(self.__rec_meta)

    @QtCore.pyqtSlot(dict)
    def get_ocr(self, ocr):
        self.ann = ocr['ann']
        self.knn = ocr['knn']
        self.svm = ocr['svm']
        self.msp = ocr['msp']
        print 'end run'

    QtCore.pyqtSlot()
    def end(self):
        self.unlock_interface.emit()





