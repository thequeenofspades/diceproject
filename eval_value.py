import numpy as np
from value_nn import ValueNN
from config import config
from utils import load_train_data, load_dev_data, load_test_data

img_path = 'examples/images/'
label_path = 'examples/labels/'
eval_test = False

def analyze_errors(labels, preds, type_labels):
	count_by_label = {}
	correct_by_label = {}
	count_by_type = {}
	correct_by_type = {}
	for i in range(len(labels)):
		label = labels[i]
		if label in count_by_label:
			count_by_label[label] += 1
		else:
			count_by_label[label] = 1
		if type_labels[i] in count_by_type:
			count_by_type[type_labels[i]] += 1
		else:
			count_by_type[type_labels[i]] = 1
		if label == preds[i]:
			if label in correct_by_label:
				correct_by_label[label] += 1
			else:
				correct_by_label[label] = 1
			if type_labels[i] in correct_by_type:
				correct_by_type[type_labels[i]] += 1
			else:
				correct_by_type[type_labels[i]] = 1
	acc_by_label = {}
	acc_by_type = {}
	for label in count_by_label:
		correct = 0
		if label in correct_by_label:
			correct = correct_by_label[label]
		acc_by_label[label] = correct / float(count_by_label[label])
	for label in count_by_type:
		correct = 0
		if label in correct_by_type[label]:
			correct = correct_by_type[label]
		acc_by_type[label] = correct / float(count_by_type[label])
	print "Accuracy by value: %s" % str(acc_by_label)
	print "Accuracy by type: %s" % str(acc_by_type)

if __name__ == '__main__':
	nn = ValueNN(config)
	X_train, Y_train, data_center, orig_X_train, orig_Y_train, type_labels_train = load_train_data(img_path, label_path, 'val', config.exclude)
	X_dev, Y_dev, type_labels_dev = load_dev_data(img_path, label_path, data_center, 'val', config.exclude)
	if eval_test:
		X_test, Y_test, type_labels_test = load_test_data(img_path, label_path, data_center, 'val', config.exclude)

	train_preds, train_acc = nn.validate(orig_X_train, orig_Y_train)
	dev_preds, dev_acc = nn.validate(X_dev, Y_dev)

	print "Train accuracy: %f" % train_acc
	analyze_errors(orig_Y_train, train_preds, type_labels_train)
	print "Dev accuracy: %f" % dev_acc
	analyze_errors(Y_dev, dev_preds, type_labels_dev)

	if eval_test:
		test_preds, test_acc = nn.validate(X_test, Y_test)
		print "Test accuracy: %f" % test_acc
		analyze_errors(Y_test, test_preds, type_labels_test)