import numpy as np
from value_nn import ValueNN
from config import config
from utils import load_train_data, load_dev_data

img_path = 'examples/images/'
label_path = 'examples/labels/'
loss_path = 'losses.txt'
eval_path = 'eval.txt'

def train(X, Y, batches):
	loss_file = open(loss_path, 'a+')
	losses, steps = nn.train(X, Y, batches)
	for i in range(len(losses)):
		loss_file.write('%d %f\n' % (steps[i], losses[i]))
	loss_file.close()
	return steps[-1]

def eval(X_train, Y_train, X_dev, Y_dev, step):
	print "Evaluating..."
	eval_file = open(eval_path, 'a+')
	_, train_acc = nn.validate(X_train, Y_train)
	_, dev_acc = nn.validate(X_dev, Y_dev)
	eval_file.write('%d %f %f\n' % (step, train_acc, dev_acc))
	eval_file.close()
	print "Train accuracy: %f" % train_acc
	print "Dev accuracy: %f" % dev_acc

if __name__ == '__main__':
	nn = ValueNN(config)
	X_train, Y_train = load_train_data(img_path, label_path, 'val')
	X_dev, Y_dev = load_dev_data(img_path, label_path, 'val')

	times_to_eval = int(config.batches / config.eval_freq)
	for i in range(times_to_eval):
		step = train(X_train, Y_train, config.eval_freq)
		eval(X_train, Y_train, X_dev, Y_dev, step)