import numpy as np
from value_nn import ValueNN
from config import config
from utils import load_train_data, load_dev_data
import cPickle as pickle
from os import listdir

img_path = 'examples/images/'
label_path = 'examples/labels/'
loss_path_pkl = 'losses_val.pkl'
eval_path_pkl = 'eval_val.pkl'

def train(nn, X, Y, batches, losses, losses_pkl={}):
	loss_file_pkl = open(loss_path_pkl, 'wb')
	losses, steps = nn.train(X, Y, batches, losses)
	losses = losses[-len(steps):]
	for i in range(len(losses)):
		losses_pkl[steps[i]] = losses[i]
	pickle.dump(losses_pkl, loss_file_pkl, pickle.HIGHEST_PROTOCOL)
	loss_file_pkl.close()
	return losses[-100:], steps[-1], losses_pkl

def eval(nn, X_train, Y_train, X_dev, Y_dev, step, eval_pkl={}):
	print "Evaluating..."
	eval_file_pkl = open(eval_path_pkl, 'wb')
	_, train_acc = nn.validate(X_train, Y_train)
	_, dev_acc = nn.validate(X_dev, Y_dev)
	eval_pkl[step] = [train_acc, dev_acc]
	pickle.dump(eval_pkl, eval_file_pkl, pickle.HIGHEST_PROTOCOL)
	eval_file_pkl.close()
	print "Train accuracy: %f" % train_acc
	print "Dev accuracy: %f" % dev_acc
	return train_acc, dev_acc, eval_pkl

if __name__ == '__main__':
	nn = ValueNN(config)
	X_train, Y_train, data_center, orig_X_train, orig_Y_train = load_train_data(img_path, label_path, 'val', config.exclude)
	X_dev, Y_dev = load_dev_data(img_path, label_path, data_center, 'val', config.exclude)

	times_to_eval = int(config.batches / config.eval_freq)
	losses = []
	if loss_path_pkl in listdir('.'):
		loss_file = open(loss_path_pkl, 'rb')
		losses_pkl = pickle.load(loss_file)
		loss_file.close()
	else:
		losses_pkl = {}
	if eval_path_pkl in listdir('.'):
		eval_file = open(eval_path_pkl, 'rb')
		eval_pkl = pickle.load(eval_file)
		eval_file.close()
	else:
		eval_pkl = {}

	_, best_dev_acc, _ = eval(nn, orig_X_train, orig_Y_train, X_dev, Y_dev, 0, eval_pkl)
	best_dev_acc_step = 0

	for i in range(times_to_eval):
		losses, step, losses_pkl = train(nn, X_train, Y_train, config.eval_freq, losses, losses_pkl)
		train_acc, dev_acc, eval_pkl = eval(nn, orig_X_train, orig_Y_train, X_dev, Y_dev, step, eval_pkl)
		if dev_acc > best_dev_acc:
			best_dev_acc = dev_acc
			best_dev_acc_step = step
			nn.save()
	print "Best dev accuracy: %f (step %d)" % (best_dev_acc, best_dev_acc_step)