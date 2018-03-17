import numpy as np
from value_nn import ValueNN
from config import config
from utils import load_train_data, load_dev_data

img_path = 'examples/images/'
label_path = 'examples/labels/'

if __name__ == '__main__':
	nn = ValueNN(config)
	X_train, Y_train = load_train_data(img_path, label_path, 'val')
	X_dev, Y_dev = load_dev_data(img_path, label_path, 'val')
	nn.train(X_train, Y_train, config.batches)
	_, train_acc = nn.validate(X_train, Y_train)
	_, dev_acc = nn.validate(X_dev, Y_dev)
	print "Train accuracy: %f" % train_acc
	print "Dev accuracy: %f" % dev_acc