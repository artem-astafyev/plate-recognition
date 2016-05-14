import time

import cv2

import helper as hp

class MSP:
    def __init__(self):
        self.__patterns_num = []
        self.__patterns_sym = []
        self.__load_num_patterns(hp.msp_num)
        self.__load_sym_patterns(hp.msp_sym)
        print 'loading MSP...'

    def __load_num_patterns(self, input_dir):
        paths = hp.get_paths(input_dir)
        for path in paths:
            img = hp.get_image(input_dir, path)
            self.__patterns_num.append(hp.get_int_array_from_image(img))

    def __load_sym_patterns(self, input_dir):
        paths = hp.get_paths(input_dir)
        for path in paths:
            img = cv2.imread("{0}/{1}".format(input_dir, path))
            self.__patterns_sym.append(hp.get_int_array_from_image(img))

    def __get_mode(self, mode):
        if mode == "num":
            return self.__patterns_num, [0] * 10
        elif mode == "sym":
            return self.__patterns_sym, [0] * 22

    def rec(self, img, mode):
        patterns, num = self.__get_mode(mode)
        rec = hp.get_int_array_from_image(img)
        for i in range(0, len(patterns)):
            for j in range(0, len(rec)):
                if rec[j] == patterns[i][j]:
                    num[i] += 1
        rec = hp.get_max_from_int_array(num)
        return hp.ann_get_lit(rec, mode), rec


def testing_simple_pattern_from_image_base(input_dir, set_name, mode, msp):
    paths = hp.get_paths(input_dir)
    success, error = "", ""
    counter, err = 1, 1
    for path in paths:
        # recognition
        img = hp.get_image(input_dir, path)
        lit, i = msp.rec(img, mode)
        # test
        tlit, ti = hp.get_test(path, mode)

        if tlit != lit:
            err += 1
            error += "{0}\n".format(path)
        else:
            success += "{0}\n".format(path)
        counter += 1

    hp.print_result(counter, err, set_name, mode)
    #hp.write_report("test/msp", set_name, mode, success)
    #hp.write_error("test/msp", set_name, mode, error)


if __name__ == "__main__":
    N = 2
    msp = MSP()
    start = time.time()
    if N == 1:
        testing_simple_pattern_from_image_base(hp.small_sym, "small", hp.mode_sym, msp)
        testing_simple_pattern_from_image_base(hp.small_num, "small", hp.mode_num, msp)


    if N == 2:
        testing_simple_pattern_from_image_base(hp.big_sym, "big", hp.mode_sym, msp)

    print time.time() - start

    if N == 3:
        testing_simple_pattern_from_image_base(hp.big_num, "big", hp.mode_num, msp)

    if N == 5:
        testing_simple_pattern_from_image_base("try/error/num", "trash", hp.mode_num, msp)
    if N == 5:
            testing_simple_pattern_from_image_base("try/error/sym", "trash", hp.mode_sym, msp)