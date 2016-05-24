import sys

import cv2

import helper as hp


class MSP():
    name = "MSP"
    def __init__(self):
        self.__patterns_num = []
        self.__patterns_sym = []
        self.__labels_num = []
        self.__labels_sym = []
        msp_num, msp_sym = "msp/num", "msp/sym"
        self.__load_num_patterns(msp_num)
        self.__load_sym_patterns(msp_sym)
        print 'loading MSP...'

    def __load_num_patterns(self, input_dir):
        paths = hp.get_paths(input_dir)
        self.__patterns_num = [hp.get_gray_image(input_dir, path) for path in paths]
        self.__labels_num = [hp.get_test(path, "num")[0] for path in paths]

    def __load_sym_patterns(self, input_dir):
        paths = hp.get_paths(input_dir)
        self.__patterns_sym = [hp.get_gray_image(input_dir, path) for path in paths]
        self.__labels_sym = [hp.get_test(path, "sym")[0] for path in paths]

    def __get_mode(self, mode):
        if mode == "num":
            return self.__labels_num, self.__patterns_num
        elif mode == "sym":
            return self.__labels_sym, self.__patterns_sym

    def rec(self, img, mode):
        tmp_max, tmp, rec = sys.maxint, 0, 0
        labels, patterns = self.__get_mode(mode)
        for pattern, label in zip(patterns, labels):
            tmp = cv2.countNonZero(pattern - img)
            if tmp < tmp_max: tmp_max, rec = tmp, label
        return rec
