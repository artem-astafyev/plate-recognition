import os
import re
import time

import cv2
import numpy as np
from fann2 import libfann as lf
from numpy.linalg import norm

import helper as hp

make_num_file_dir_in = "img/train/nums-15x20"
make_num_file_dir_out =  "/ann/train/"
mane_num_file_name = "train-nums.dat"

make_sym_file_dir_in = "img/train/syms-15x20"
make_sym_file_dir_out=  "/ann/train/"
mane_sym_file_name = "train-syms.dat"

nn_input_num_data_file = "ann/input/input-nums.data"
nn_input_sym_data_file = "ann/input/input-syms.data"

nn_num_test_data_file = "ann/test/test-nums.dat"
nn_sym_test_data_file = "ann/test/test-syms.dat"


class ANN:
    name = "ANN"
    def __init__(self):
        self.hog_num_file="ann/input/hog-nums.ann"
        self.hog_sym_file = "ann/input/hog-syms.ann"
        self.num_ann = lf.neural_net()
        self.sym_ann = lf.neural_net()
        self.num_ann.create_from_file(self.hog_num_file)
        self.sym_ann.create_from_file(self.hog_sym_file)
        print 'loading ANN...'

    # return num name, num
    def rec(self, img, mode):
        if mode == hp.mode_num:
            vec = list(self.__buddy_hog(img))
            rec = hp.get_max_from_int_array(self.num_ann.run(vec))
            return str(rec)
        elif mode == hp.mode_sym:
            vec = list(self.__buddy_hog(img))
            rec = hp.get_max_from_int_array(self.sym_ann.run(vec))
            return hp.ann_get_lit(rec, mode)

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


def make_input_text_vector(img, test=False):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, img = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)
    w, h = img.shape[:2]
    vector = ""
    for i in range(0, w):
        for j in range(0, h):
            if img[i][j] == 255:
                vector += "0 "
            else:
                vector += "1 "
    return vector + "\n"


def make_output_text_vector(k, n):
    assert k < n, "k > n"
    assert k > -1 and n > 0, "k | n < 0"
    out = [0] * n
    out[k] = 1
    out = map(str, out)
    return " ".join(out) + "\n"


def make_num_file(dir_in, dir_out, filename):

    paths = os.listdir(dir_in)

    out_file = "{0} {1} {2}\n".format(len(paths), 300, 10)

    for path in paths:
        result = re.match(r"(\d+)\.(\d+)\.(png|jpg)", path)
        if result is not None:
            key = result.groups()[0]
            img = cv2.imread("{0}/{1}".format(dir_in, path))
            in_vector = make_input_text_vector(img)
            out_vector = make_output_text_vector(int(key), 10)
            out_file += in_vector + out_vector

    with open(os.path.join(dir_out, filename), 'wb') as temp_file:
        temp_file.write(out_file)


def make_sym_file(dir_in, dir_out, filename):
    paths = os.listdir(dir_in)
    out_file = "{0} {1} {2}\n".format(len(paths), 300, 22)

    for path in paths:
        result = re.match(r"(\d+)\.(\d+)\.(png|jpg)", path)
        if result is not None:
            key = result.groups()[0]
            img = cv2.imread("{0}/{1}".format(dir_in, path))
            in_vector = make_input_text_vector(img)
            out_vector = make_output_text_vector(int(key) - 10, 22)
            out_file += in_vector + out_vector

    with open(os.path.join(dir_out, filename), 'wb') as temp_file:
        temp_file.write(out_file)


def testing_from_file(ann_data_file, ann_test_data_file, dir_report):
    ann = lf.neural_net()
    ann.create_from_file(ann_data_file)

    with open(ann_test_data_file) as f:
        content = f.readlines()

    count, err = len(content) // 2, 0
    for i in range(1, len(content)):
        if i % 2 == 1:
            # input vector for ann
            vec = hp.get_int_array_from_string(content[i][0:-2])
            rec = hp.get_max_from_int_array(ann.run(vec))

            # control vector
            vec = hp.get_int_array_from_string(content[i + 1][0:-2])
            num = hp.get_max_from_int_array(vec)

            if num != rec:
                print (rec, num)
                err += 1

    print "total:", count
    print "error:", err
    print "p={0:.3f}%".format(100 - float(err) / float(count) * 100)


def buddy_hog(self, img):
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


def make_hog_sym_file():
    base = hp.small_test_sym_images_15x20
    samples, labels = [], []
    paths = hp.get_paths(base)
    for path in paths:
        img = hp.get_image(base, path)
        sample = buddy_hog(img)
        label = hp.get_name(path) - 10
        labels.append(label)
        samples.append(sample)

    samples = np.float32(samples)
    labels = np.array(labels)
    labels = list(labels)

    # head
    document = ""
    head1 = str(len(samples)) + " "
    head2 = str(len(samples[0])) + " "
    head3 = str("22") + "\n"
    head = head1 + head2 + head3
    document += head
    for sample, label in zip(samples, labels):
         input = ' '.join(map(str, list(sample)))
         output = make_output_text_vector(int(label), 22)
         document += input + "\n" + output
    print document
    with open(os.path.join("ann/test/", "test-hog-syms.dat"), 'wb') as temp_file:
        temp_file.write(document)


def make_hog_num_file():
    base = hp.small_test_num_images_15x20
    samples, labels = [], []
    paths = hp.get_paths(base)
    for path in paths:
        img = hp.get_image(base, path)
        sample = buddy_hog(img)
        label = hp.get_name(path)
        labels.append(label)
        samples.append(sample)

    samples = np.float32(samples)
    labels = np.array(labels)
    labels = list(labels)

    # head
    document = ""
    head1 = str(len(samples)) + " "
    head2 = str(len(samples[0])) + " "
    head3 = str("10") + "\n"
    head = head1 + head2 + head3
    document += head
    for sample, label in zip(samples, labels):
         input = ' '.join(map(str, list(sample)))
         output = make_output_text_vector(int(label), 10)
         document += input + "\n" + output
    print document
    with open(os.path.join("ann/test/", "test-hog-nums.dat"), 'wb') as temp_file:
        temp_file.write(document)


def testing_from_image_base(input_dir, set_name, mode, ann):
    #init block
    success, error = "", ""
    counter, err = 1, 0
    paths = hp.get_paths(input_dir)
    for path in paths:
        # recognition
        img = hp.get_image(input_dir, path)
        lit, i = ann.rec(img, mode)
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
    ann = ANN()

    start = time.time()
    if N == 1:
        testing_from_image_base(hp.small_sym, "small", hp.mode_sym, ann)

    if N == 1:
        testing_from_image_base(hp.small_num, "small", hp.mode_num, ann)

    if N == 1:
        testing_from_image_base(hp.big_sym, "big", hp.mode_sym, ann)

    if N == 1:
        testing_from_image_base(hp.big_num, "big", hp.mode_num, ann)

    print 'time: %f' % (time.time() - start)









