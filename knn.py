import cv2
import numpy as np
from numpy.linalg import norm

import helper as hp


class StatModel(object):
    def load(self, fn):
        self.model.load(fn)  # Known bug: https://github.com/Itseez/opencv/issues/4969

    def save(self, fn):
        self.model.save(fn)


class KNearest(StatModel):
    def __init__(self, k=3):
        self.k = k
        self.model = cv2.ml.KNearest_create()

    def train(self, samples, responses):
        self.model.train(samples, cv2.ml.ROW_SAMPLE, responses)

    def predict(self, samples):
        retval, results, neigh_resp, dists = self.model.findNearest(samples, self.k)
        return results.ravel()


knn_num_input_data_file = "knn/num.data"
knn_sym_input_data_file = "knn/sym.data"


class KNN():
    def __init__(self):
        self.knn_num = hp.train_num
        self.knn_sym = hp.train_sym
        self.samples_num_file = "knn/samples-num.npy"
        self.labels_num_file = "knn/labels-num.npy"
        self.samples_sym_file = "knn/samples-sym.npy"
        self.labels_sym_file = "knn/labels-sym.npy"

        self.num_model = self.__make_knn_from_npy("num")
        self.sym_model = self.__make_knn_from_npy("sym")

        print('loading KNN...')


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
        # print hist
        return np.float32(hist)

    def make_npy(self):
        base = self.knn_num
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

        base = self.knn_sym
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

    def __make_knn_from_images(self, mode):
        base = ""
        modifier = 0
        samples, labels = [], []
        if mode == "num":
            base = self.knn_num
            modifier = 0
        elif mode == "sym":
            base = self.knn_sym
            modifier = 10
        paths = hp.get_paths(base)

        for path in paths:
            img = hp.get_image(base, path)
            sample = self.__buddy_hog(img)
            label = hp.get_name(path) - modifier
            labels.append(label)
            samples.append(sample)

        samples = np.float32(samples)
        labels = np.array(labels)

        model = KNearest(k=5)
        model.train(samples, labels)

        return model

    def __make_knn_from_npy(self, mode):
        samples, labels = [], []
        if mode == "num":
            samples = np.load(self.samples_num_file)
            labels = np.load(self.labels_num_file)
        elif mode == "sym":
            samples = np.load(self.samples_sym_file)
            labels = np.load(self.labels_sym_file)

        model = KNearest(k=5)
        model.train(samples, labels)
        model.save("knn/knn-{0}.dat".format(mode))

        return model

    def rec(self, img, mode):
        # data = hp.get_int_array_from_image(img)
        data = self.__buddy_hog(img)
        data2 = np.float32([np.float32(data)])

        rec = -1
        if mode == "sym":
            rec = np.int32(self.sym_model.predict(data2))
        elif mode == "num":
            rec = np.int32(self.num_model.predict(data2))
        return hp.ann_get_lit(rec, mode), rec


def testing_knn_from_image_base(input_dir, set_name, mode):
    # init block
    knn = KNN()
    success, error = "", ""
    counter, err = 1, 0
    paths = hp.get_paths(input_dir)
    for path in paths:
        # recognition
        img = hp.get_image(input_dir, path)
        lit, i = knn.rec(img, mode)

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
    #hp.write_report("test/knn", set_name, mode, success)
    #hp.write_error("test/knn", set_name, mode, error)


if __name__ == "__main__":
    N = 1

    if N == 1:
        testing_knn_from_image_base(hp.small_sym, "small", hp.mode_sym)

    if N == 1:
        testing_knn_from_image_base(hp.small_num, "small", hp.mode_num)

    if N == 1:
        testing_knn_from_image_base(hp.big_sym, "big", hp.mode_sym)

    if N == 1:
        testing_knn_from_image_base(hp.big_num, "big", hp.mode_num)
