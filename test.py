import time
import pickle

import os
import helper as hp
from ann import ANN
from com import COM
from knn import KNN
from msp import MSP
from svm import MYSVM as SVM

num_images = []
num_labels = []
sym_images = []
sym_labels = []


def load(dir_num, dir_sym):
    print '\n##############'
    print 'loading nums...'
    num_paths = hp.get_paths(dir_num)
    for path in num_paths:
        num_images.append(hp.get_gray_image(dir_num, path))
        num_labels.append(hp.get_test(path, "num")[0])
    print 'loading syms...'
    sym_paths = hp.get_paths(dir_sym)
    for path in sym_paths:
        sym_images.append(hp.get_gray_image(dir_sym, path))
        sym_labels.append(hp.get_test(path, "sym")[0])
    print '##############'


def save_pickle(dir_out):
    print '\n##############'
    print 'saving pickle...'
    with open("{0}num_images.dat".format(dir_out), "wb") as f:
        pickle.dump(num_images, f)
    with open("{0}num_labels.dat".format(dir_out), "wb") as f:
        pickle.dump(num_labels, f)
    with open("{0}sym_images.dat".format(dir_out), "wb") as f:
        pickle.dump(sym_images, f)
    with open("{0}sym_labels.dat".format(dir_out), "wb") as f:
        pickle.dump(sym_labels, f)
    print '##############'


def load_pickle(dir_in):
    print '\n##############'
    print 'loading pickle...'
    print 'loading nums...'
    global num_labels, num_images, sym_images, sym_labels
    with open("{0}num_images.dat".format(dir_in), "rb") as f:
        num_images = pickle.load(f)
    with open("{0}num_labels.dat".format(dir_in), "rb") as f:
        num_labels = pickle.load(f)
    print 'loading syms...'
    with open("{0}sym_images.dat".format(dir_in), "rb") as f:
        sym_images = pickle.load(f)
    with open("{0}sym_labels.dat".format(dir_in), "rb") as f:
        sym_labels = pickle.load(f)
    print '##############'

def test(method):
    print '\n##############'
    print 'method: ' + method.name + '\n'
    total_num = len(num_images)
    error_num = total_num
    print 'start num test...'
    start = time.time()
    for (num, label) in zip(num_images, num_labels):
        lit = method.rec(num, "num")
        if lit == label:
            error_num -= 1

    num_time = time.time() - start
    num_prob = float(total_num - error_num) / float(total_num) * 100

    print 'mode: ', "num"
    print 'time: ', "{0:.2f}s".format(num_time)
    print 'total: ', total_num
    print 'error: ', error_num
    print 'result: ', "{0:.2f}%".format(float(total_num - error_num) / float(total_num) * 100)
    print 'end num test.'

    total_sym = len(sym_images)
    error_sym = total_sym
    print '\nstart sym test...'
    start = time.time()
    for (sym, label) in zip(sym_images, sym_labels):
        lit = method.rec(sym, "sym")
        if lit == label:
            error_sym -= 1

    sym_time = time.time() - start
    sym_prob = float(total_sym - error_sym) / float(total_sym) * 100
    print 'method: ', method.name
    print 'mode: ', "sym"
    print 'time: ', "{0:.2f}s".format(sym_time)
    print 'total: ', total_sym
    print 'error: ', error_sym
    print 'result: ', "{0:.2f}%".format(sym_prob)
    print 'end sym test.\n'

    total_time = sym_time + num_time
    total_chars = len(num_labels) + len(sym_labels)
    print 'total chars', total_chars
    print 'total time:', "{0:.2f}s".format(total_time)
    print '1 char time: ', "{0:.3f}ms".format(float(total_time) / float(total_chars) * 1000)
    print 'num result', "{0:.2f}%".format(num_prob)
    print 'sym result', "{0:.2f}%".format(sym_prob)
    print '##############'


if __name__ == "__main__":
    num_path = "images/test/set3/num"
    sym_path = "images/test/set3/sym"
    pickle_path = "test/set3/"
    PICKLE = True

    print '##############'
    ann = ANN()
    svm = SVM()
    msp = MSP()
    knn = KNN()
    com = COM(ann, svm, knn, msp)
    print '##############'

    if not PICKLE:
        load(num_path, sym_path)
        save_pickle(pickle_path)
    else:
        load_pickle(pickle_path)

    N = 0
    if N == 1 or N == 0:
        test(ann)
    if N == 2 or N == 0:
        test(svm)
    if N == 3 or N == 0:
        test(msp)
    if N == 4 or N == 0:
        test(knn)
    if N == 5 or N == 0:
        test(com)
