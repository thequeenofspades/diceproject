import numpy as np
from value_nn import ValueNN
from config import config
from utils import load_train_data, load_dev_data, load_test_data

img_path = 'examples/images/'
label_path = 'examples/labels/'
eval_test = False

if __name__ == '__main__':
	nn = ValueNN(config)
	X_train, Y_train = load_train_data(img_path, label_path, 'val')
	X_dev, Y_dev = load_dev_data(img_path, label_path, 'val')
	if eval_test:
		X_test, Y_test = load_test_data(img_path, label_path, 'val')
	_, train_acc = nn.validate(X_train, Y_train)
	_, dev_acc = nn.validate(X_dev, Y_dev)
	print "Train accuracy: %f" % train_acc
	print "Dev accuracy: %f" % dev_acc
	if eval_test:
		_, test_acc = nn.validate(X_test, Y_test)
		print "Test accuracy: %f" % test_acc