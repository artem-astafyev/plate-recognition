import time

import cv2
import numpy as np
from numpy.linalg import norm

import helper as hp


class StatModel(object):
    def load(self, fn):
        self.model.load(fn)

    def save(self, fn):
        self.model.save(fn)


class SVM(StatModel):
    def __init__(self, C=1, gamma=0.5):
        self.model = cv2.ml.SVM_create()
        self.model.setGamma(gamma)
        self.model.setC(C)
        self.model.setKernel(cv2.ml.SVM_RBF)
        self.model.setType(cv2.ml.SVM_C_SVC)

    def train(self, samples, responses):
        self.model.train(samples, cv2.ml.ROW_SAMPLE, responses)

    def predict(self, samples):
        return self.model.predict(samples)[1].ravel()


svm_num_input_data_file = "svm/num.data"
svm_sym_input_data_file = "svm/sym.data"


class MYSVM:
    def __init__(self):
        self.svm_num = hp.train_num
        self.svm_sym = hp.train_sym
        self.samples_num_file = "svm/samples-num.npy"
        self.labels_num_file = "svm/labels-num.npy"
        self.samples_sym_file = "svm/samples-sym.npy"
        self.labels_sym_file = "svm/labels-sym.npy"
        self.num_model = self.__make_svm_from_npy("num")
        self.sym_model = self.__make_svm_from_npy("sym")
        print('loading SVM...')

    def __buddy_hog(self, img):
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
        mag, ang = cv2.cartToPolar(gx, gy)
        bin_n = 16
        bin = np.int32(bin_n * ang / (2 * np.pi))
        bin_cells = bin[:10, :10], bin[10:, :10], bin[:10, 10:], bin[10:, 10:]
        mag_cells = mag[:10, :10], mag[10:, :10], mag[:10, 10:], mag[10:, 10:]
        hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
        hist = np.hstack(hists)

        # transform to Hellinger kernel
        eps = 1e-7
        hist /= hist.sum() + eps
        hist = np.sqrt(hist)
        hist /= norm(hist) + eps
        return np.float32(hist)

    def __make_svm_from_images(self, dir_train, mode):
        samples, labels = [], []
        paths = hp.get_paths(dir_train)
        modifier = 0
        if mode == "num":
            modifier = 0
        elif mode == "sym":
            modifier = 10
        for path in paths:
            img = hp.get_image(dir_train, path)
            sample = self.__buddy_hog(img)
            label = hp.get_name(path) - modifier
            labels.append(label)
            samples.append(sample)

        samples = np.array(samples)
        labels = np.array(labels)

        rand = np.random.RandomState(321)
        shuffle = rand.permutation(len(samples))
        samples, labels = samples[shuffle], labels[shuffle]

        samples = np.float32(samples)

        model = SVM(C=2.67, gamma=5.383)
        model.train(samples, labels)
        return model

    def __make_svm_from_npy(self, mode):
        samples, labels = [], []
        if mode == "num":
            samples = np.load(self.samples_num_file)
            labels = np.load(self.labels_num_file)
        elif mode == "sym":
            samples = np.load(self.samples_sym_file)
            labels = np.load(self.labels_sym_file)


        model = SVM(C=2.67, gamma=5.383)
        model.train(samples, labels)
        return model

    def make_npy(self):
        base = self.svm_num
        samples, labels = [], []
        paths = hp.get_paths(base)
        for path in paths:
            img = hp.get_image(base, path)
            sample = self.__buddy_hog(img)
            label = hp.get_name(path)
            labels.append(label)
            samples.append(sample)

        samples = np.float32(samples)
        labels = np.array(labels)
        np.save(self.samples_num_file, samples)
        np.save(self.labels_num_file, labels)

        base = self.svm_sym
        samples, labels = [], []
        paths = hp.get_paths(base)
        for path in paths:
            img = hp.get_image(base, path)
            sample = self.__buddy_hog(img)
            label = hp.get_name(path) - 10
            labels.append(label)
            samples.append(sample)

        samples = np.float32(samples)
        labels = np.array(labels)
        np.save(self.samples_sym_file, samples)
        np.save(self.labels_sym_file, labels)

    def rec(self, img, mode):
        data = self.__buddy_hog(img)
        data2 = np.float32([np.float32(data)])

        rec = -1
        if mode == "sym":
            rec = np.int32(self.sym_model.predict(data2))
        elif mode == "num":
            rec = np.int32(self.num_model.predict(data2))
        return hp.ann_get_lit(rec, mode), rec


def testing_svm_from_image_base(input_dir, set_name, mode, svm):
    success, error = "", ""
    counter, err = 1, 0
    paths = hp.get_paths(input_dir)

    for path in paths:
        # recognition
        img = hp.get_image(input_dir, path)
        lit, i = svm.rec(img, mode)

        # test
        tlit, ti = hp.get_test(path, mode)
        if tlit != lit:
            err += 1
            error += "{0}\n".format(path)
            hp.write_image("try/error", path, img)
        else:
            success += "{0}\n".format(path)
        counter += 1

    hp.print_result(counter, err, set_name, mode)


if __name__ == "__main__":

    N = 1

    svm = MYSVM()
    start = time.time()

    if N == 1:
        testing_svm_from_image_base(hp.small_sym, "small", hp.mode_sym, svm)

    if N == 1:
        testing_svm_from_image_base(hp.small_num, "small", hp.mode_num, svm)

    if N == 1:
        testing_svm_from_image_base(hp.big_sym, "big", hp.mode_sym, svm)

    if N == 1:
        testing_svm_from_image_base(hp.big_num, "big", hp.mode_num, svm)
    print 'time: %f' % (time.time() - start)