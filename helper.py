# usr/bin/python2
import os
import re

import cv2

ann_num_test_data_file = "ann/test/test-nums.dat"
ann_sym_test_data_file = "ann/test/test-syms.dat"

dir_test_num_images_30x40 = "images/test/set2/30x40/nums"
dir_test_sym_images_30x40 = "images/test/set2/30x40/syms"
big_test_num_images_15x20 = "images/test/set2/15x20/nums"
big_test_sym_images_15x20 = "images/test/set2/15x20/syms"

small_test_num_images_15x20 = "images/test/set1/15x20/nums"
small_test_sym_images_15x20 = "images/test/set1/15x20/syms"

train_num = "images/train/nums-15x20"
train_sym = "images/train/syms-15x20"

small_num = small_test_num_images_15x20
small_sym = small_test_sym_images_15x20

big_num = big_test_num_images_15x20
big_sym = big_test_sym_images_15x20

mode_num = "num"
mode_sym = "sym"

lit = ["0123456789", "ABCDEFHKLMNOPRSTUVWXYZ"]


def get_int_array_from_string(line):
    vector = map(int, line.split(" "))
    return vector


def to_bin(img):
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    return img


def get_int_array_from_image(img):
    vector = []
    h, w, img = to_bin(img)
    map(vector.extend, img[0:h])
    map(lambda x: 0 if x == 255 else 1, vector)
    return vector


def get_max_from_int_array(mas):
    return mas.index(max(mas))


pattern = re.compile(r'(\d+)\.(\d+)\.(png|jpg)', re.IGNORECASE)


def get_paths(input_dir):
    paths = os.listdir(input_dir)
    valid = [path for path in paths if path.endswith(('.jpg', '.png', '.JPEG', '.PNG', '.JPG'))]
    return valid


def show(image, display=True):
    if display:
        cv2.imshow("Show: ", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return


def get_image(dir_in, name):
    return cv2.imread("{0}/{1}".format(dir_in, name))


def get_gray_image(dir_in, name):
    return cv2.imread("{0}/{1}".format(dir_in, name), 0)


def write_image(dir_in, name, img):
    cv2.imwrite("{0}/{1}".format(dir_in, name), img)


def ann_get_lit(num, mode):
    num = int(num)
    if mode == mode_num:
        return str(lit[0][num])
    elif mode == mode_sym:
        return str(lit[1][num])


def get_name(path):
    return int(re.match(pattern, path).group(1))


def get_test(path, mode):
    num = int(re.match(pattern, path).group(1))

    if mode == mode_num:
        tlit = str(lit[0][num])
    elif mode == mode_sym:
        tlit = str(lit[1][num - 10])
    else:
        tlit = "error"
    return tlit, num


def write_report(path, set_name, mode, text):
    with open(os.path.join(path, "{0}-{1}-report.txt".format(mode, set_name)), 'wb') as temp_file:
        temp_file.write(text)


def write_error(path, set_name, mode, text):
    with open(os.path.join(path, "{0}-{1}-error.txt".format(mode, set_name)), 'wb') as temp_file:
        temp_file.write(text)


def print_result(counter, err, set_name, mode):
    print ("set name"), set_name
    print ("mode"), mode
    print ("total:"), counter
    print ("error:"), err
    print ("p={0:.2f}%".format(100 - float(err) / float(counter) * 100))


def rename_base_sym(dir_in, dir_out, start=1, step=1):
    dict = {"-1": 1}
    paths = get_paths(dir_in)
    for path in paths:
        result = re.match(r"(\d+)\.(\d+)\.(png|jpg)", path)
        key = result.groups()[0]
        if key not in dict:
            dict[key] = start
        else:
            dict[key] += step
        img = cv2.imread("{0}/{1}".format(dir_in, path))
        filename = "{0}/{1}.{2}.png".format(dir_out, key, dict[key])
        cv2.imwrite(filename, img)


def resize_to_small(sym):
    h, w = 15, 20
    res = cv2.resize(sym, (h, w), interpolation=cv2.INTER_CUBIC)
    res = to_bin(res)
    return res


def resize_to_big(sym):
    h, w = 30, 40
    res = cv2.resize(sym, (h, w), interpolation=cv2.INTER_CUBIC)
    res = to_bin(res)
    return res
