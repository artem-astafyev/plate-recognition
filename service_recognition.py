import Queue
from collections import Counter

import cv2
from PyQt4 import QtCore
from PyQt4.QtCore import SIGNAL

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
        ann, knn, msp, svm = ANN(), KNN(), MSP(), MYSVM()

        print 'run'

        self.on_loaded.emit({'ann':ann, 'knn':knn, 'msp' : msp, 'svm':svm})


class Recognition(QtCore.QThread):
    plate_recognized = QtCore.pyqtSignal(dict)
    unlock_interface = QtCore.pyqtSignal()

    __empty = ""
    __queue = Queue.Queue(-1)

    def __init__(self, parent=None):
        super(Recognition, self).__init__(parent)
        self.thread = RecognitionLoader()
        self.thread.on_loaded.connect(self.get_ocr)
        self.connect(self.thread, SIGNAL("finished()"), self.end)
        self.__empty = cv2.imread("view/empty.jpg")
        self.thread.start()

    def run(self):
        while True:
            if not self.__queue.empty():
                meta = self.__queue.get()
                meta['signs'] = [self.__empty] * 8
                meta['ann'] = ["&nbsp;"] * 8
                meta['knn'] = ["&nbsp;"] * 8
                meta['svm'] = ["&nbsp;"] * 8
                meta['msp'] = ["&nbsp;"] * 8
                meta['rec'] = ["&nbsp;"] * 8
                meta['result'] = ""
                print "preprocessing", meta['id']
                rows, plate = self.__preprocessing(meta)
                print "segmentation", meta['id']
                pattern, signs = self.__segmentation(meta)
                print "recognition", meta['id']
                for counter, (pat, sign) in enumerate(zip(pattern, signs), start=0):
                    meta['result'] += self.__recognition(pat, sign, counter, meta)

                self.plate_recognized.emit(meta)

                self.__queue.task_done()

    def __preprocessing(self, meta):
        morph_size = 4
        plate = meta['plate']
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

        meta['plate'] = res
        meta['blur'] = blur
        meta['canny'] = canny
        meta['morph'] = morph
        meta['thresh'] = thr
        meta['rows'] = rows

        return rows, thr

    def __segmentation(self, meta):
        signs, rects = [], []
        plate = meta['thresh']
        rows = meta['rows']
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

    def __recognition(self, pat, sign, counter, meta):

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

        print at, kt, st, mt

        meta['signs'][counter] = sign
        meta['ann'][counter] = at
        meta['knn'][counter] = kt
        meta['svm'][counter] = st
        meta['msp'][counter] = mt
        meta['rec'][counter] = rec

        return rec

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

    @QtCore.pyqtSlot(dict)
    def process(self, meta):
        self.__queue.put(meta)
        print  'qsize', self.__queue.qsize()

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