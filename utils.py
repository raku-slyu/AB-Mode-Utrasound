import os, sys
import random
import logging
import importlib

def error_logger(txt):
    logging.error(txt)
    sys.exit()

def load_params():
    if len(sys.argv) != 2:
        error_logger("Include Parameter file.")
    param_path = sys.argv[1]
    if not os.path.isfile(param_path):
        error_logger("Parameter file not found.")
    return importlib.import_module(param_path[:-3]).PARAMS

class data_split():
	def __init__(self, val_ratio, test_ratio):		
		self.val_ratio = val_ratio
		self.test_ratio = test_ratio
		self.random_seed = 42
	
	def splitter(self, Xd, Yd):
		data_stamp = [x for x in range(Xd.shape[0])]
		random.seed(self.random_seed)
		random.shuffle(data_stamp)

		train_stamp = data_stamp[:int(len(data_stamp) * (1-(self.val_ratio+self.test_ratio))) ]
		val_stamp = data_stamp[int(len(data_stamp) * (1-(self.val_ratio+self.test_ratio))):int(len(data_stamp) * (1-self.test_ratio)) ]
		test_stamp = data_stamp[int(len(data_stamp) * (1-self.test_ratio)):]

		x_train = Xd[train_stamp]
		x_val = Xd[val_stamp]
		x_test = Xd[test_stamp]

		y_train = Yd[train_stamp]
		y_val = Yd[val_stamp]
		y_test = Yd[test_stamp]
		
		return x_train, x_val, x_test, y_train, y_val, y_test

