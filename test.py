import time

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
    print '##############'
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

    print '##############'
    ann = ANN()
    svm = SVM()
    msp = MSP()
    knn = KNN()
    com = COM(ann, svm, knn, msp)
    print '##############\n'

    load(num_path, sym_path)

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
