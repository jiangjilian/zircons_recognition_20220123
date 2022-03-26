import math
import numpy as np
from scipy.stats import stats
from sklearn import preprocessing
import time
import sys
import os

def row_ln(row):
    #
    gmean = stats.gmean(np.array(row))
    func = lambda x: math.log(x / gmean, math.e)
    new_row = row.apply(func)
    return new_row


def CLR(x):
    percent_x = (x.T / x.sum(axis=1)).T
    nomalized_x = percent_x.apply(row_ln, axis=1)
    return nomalized_x


def preprocess_data(x_train, x, scaling=True):
    """
    Transform the dataset into standard uniform
    """
    pro_x = CLR(x)
    pro_x_train = CLR(x_train)
    if scaling:
        scaler = preprocessing.StandardScaler().fit(pro_x_train)
        X = scaler.transform(pro_x)
    else:
        X = pro_x
    return X


def make_print_to_file(path='./', cv=0):
    """
    path， it is a path for save your log about fuction print
    example:
    use  make_print_to_file()   and the   all the information of funtion print , will be write in to a log file
    :return:
    """

    class Logger(object):
        def __init__(self, filename="Default.log", stream=sys.stdout):
            self.terminal = stream
            # self.terminal = sys.stdout
            self.log = open(filename, "a")
            # encoding='utf8')

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)

        def flush(self):
            pass

    fileName = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))

    if not os.path.exists(path):
        os.makedirs(path)
    sys.stdout = Logger(path + str(cv) + "_" + fileName + '.log', sys.stdout)

    print("The fold of cross validation: " + str(cv) + ".")
    print(fileName.center(60, '*'))
    # print("______---------------------------------")
    # return fileName
