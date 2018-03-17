import numpy as np
from value_nn import ValueNN
from config import config
from PIL import Image, ImageOps
from os import listdir
import random
from utils import load_train_data, load_dev_data, load_test_data

img_path = 'examples/images/'
label_path = 'examples/labels/'
eval_test = False

def analyze_labels(labels):
	counts = {}
	for label in labels:
		value = label
		if value in counts:
			counts[value] += 1
		else:
			counts[value] = 1
	counts = {k: counts[k] / float(len(labels)) for k in counts}
	print counts

if __name__ == '__main__':
	nn = ValueNN(config)
	X_train, Y_train = load_train_data(img_path, label_path, 'val')
	X_dev, Y_dev = load_dev_data(img_path, label_path, 'val')
	if eval_test:
		X_test, Y_test = load_test_data(img_path, label_path, 'val')
	print "Train accuracy: %f" % nn.validate(X_train, Y_train)
	print "Dev accuracy: %f" % nn.validate(X_dev, Y_dev)
	if eval_test:
		print "Test accuracy: %f" % nn.validate(X_test, Y_test)