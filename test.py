# coding=cp1251
import os
import time
from collections import Counter

import cv2

import helper as hp
from ann import ANN
from knn import KNN
from msp import MSP
from svm import MYSVM

count_dict = {}
svm_dict = {}
ann_dict = {}
knn_dict = {}
msp_dict = {}
common_dict = {}


def up_dict(dict, lit, tlit):
    if tlit == lit:
        if lit not in dict.keys():
            dict[lit] = 1
        else:
            dict[lit] += 1


def up_count_dict(tlit):
    if tlit not in count_dict.keys():
        count_dict[tlit] = 1
    else:
        count_dict[tlit] += 1


def dict_sort(dict):
    keys = dict.keys()
    keys.sort()
    values = [0] * len(dict.keys())
    counter = 0
    for key in keys:
        values[counter] = dict[key]
        counter += 1
    print keys
    print values
    return keys, values


def global_testing(input_dir, set_name, mode, ann, knn, msp, svm):
    print "Start recognition:"
    print "run test", (set_name, mode)

    abc = {3: 0, 2: 0, 1: 0}
    problem = 0
    paths = hp.get_paths(input_dir)
    total, err = len(paths), len(paths)
    for path in paths:
        # recognition
        img = hp.get_image(input_dir, path)

        at, ann_i = ann.rec(img, mode)
        kt, knn_i = knn.rec(img, mode)
        mt, msp_i = msp.rec(img, mode)
        st, svm_i = svm.rec(img, mode)
        # test
        tlit, ti = hp.get_test(path, mode)

        up_count_dict(tlit)
        up_dict(ann_dict, at, tlit)
        up_dict(knn_dict, kt, tlit)
        up_dict(msp_dict, mt, tlit)
        up_dict(svm_dict, st, tlit)

        rec = None
        xyz = {at, kt, st}
        if len(xyz) < 3:
            t = [at, kt, st]
            c = Counter(t)
            rec = max(c, key=c.get)
            if tlit == rec:
                up_dict(common_dict, rec, tlit)
                err -= 1
            else:
                rec = None

        if rec is None:
            abc[len(xyz)] += 1
            if len(xyz) == 1:
                cv2.imwrite("try/awesome/" + str([at, kt, st]) + "__" + str(tlit) + "__" + path, img)

    hp.print_result(total, err, set_name, mode)

    print "awesome:", abc[1]
    print "great:", abc[2]
    print "fuck this shit: ", abc[3]
    print "                 "


def dict_present():
    print "count:\n"
    keys, values = dict_sort(count_dict)
    print "\nmsp:\n"
    keys, values = dict_sort(msp_dict)
    print "\nann:\n"
    keys, values = dict_sort(ann_dict)
    print "\nknn:\n"
    keys, values = dict_sort(knn_dict)
    print "\nsvm:\n"
    keys, values = dict_sort(svm_dict)
    print "\ncommon:\n"
    keys, values = dict_sort(common_dict)


def make_csv_report(dict, method):
    text = "Символ;Всего символов;Распознано символов;Процент успешно распознанных символов\n"

    kk, vv = dict_sort(count_dict)
    keys, values = dict_sort(dict)
    for key, v, value in zip(keys, vv, values):
        text += str(key) + ";" + str(v) + ";" + str(value) + ";" + "{0:.2f}%".format(float(value) / float(v) * 100) + "\n"
    text += ";Общее число тестовых изображений символов;Общее число распознанных изображений символов;Процент распознанных тестовых изображений\n"
    text += ";" + str(sum(vv)) + ";" + str(sum(values)) + ";"  + "{0:.2f}%".format(
        float(sum(values)) / float(sum(vv)) * 100) + "\n"

    write = False
    if write:
        with open(os.path.join("test/report", "{0}.csv".format(method)), 'wb') as temp_file:
            temp_file.write(text)
    else:
        print "method: ",  method
        print "num: ", "{0:.2f}".format(float(sum(values[:10])) / float(sum(vv[:10])) * 100)
        print "sym: ", "{0:.2f}".format(float(sum(values[10:])) / float(sum(vv[10:])) * 100)
        print "all: ", "{0:.2f}".format(float(sum(values)) / float(sum(vv)) * 100)

        print len(vv[10:]), vv[:10], "\n"


def make_csv_reports():
    make_csv_report(msp_dict, "msp")
    make_csv_report(ann_dict, "ann")
    make_csv_report(knn_dict, "knn")
    make_csv_report(svm_dict, "svm")
    make_csv_report(common_dict, "common")

if __name__ == "__main__":

    N = 2

    print "Initializin..."
    ann = ANN()
    knn = KNN()
    msp = MSP()
    svm = MYSVM()
    print "Ending..."
    print "               "
    print "Start recognition:"


    if N == 2:
        global_testing(hp.small_sym, "small", hp.mode_sym, ann, knn, msp, svm)
        global_testing(hp.small_num, "small", hp.mode_num, ann, knn, msp, svm)

    if N == 2:
        global_testing(hp.big_sym, "big", hp.mode_sym, ann, knn, msp, svm)
        global_testing(hp.big_num, "big", hp.mode_num, ann, knn, msp, svm)

    if N == 4:
        global_testing("test/all/{0}/success".format("sym"), "all", hp.mode_sym, ann, knn, msp, svm)
        global_testing("test/all/{0}/success".format("num"), "all", hp.mode_num, ann, knn, msp, svm)

    start = time.time()
    make_csv_reports()
    print 'time: %f' % (time.time() - start)