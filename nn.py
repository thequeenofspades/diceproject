import tensorflow as tf
import numpy as np
import cv2
import time
import random
from skimage import color
import matplotlib
import argparse

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

n_epochs = 1000
dev_size = 30
batch_size = 16
img_size = 100
print_freq = 50
lamb = 0.1
keep_prob = 0.5
lr = 0.001

def load_data():
	image_list, Y = read_labeled_image_list('labels.txt')
	X = [cv2.imread(img) for img in image_list]

	for i in range(len(X)):
		img = X[i]
		height = img.shape[0]
		width = img.shape[1]
		min_dimen = min(height, width)
		# Crop image to square
		img = img[
			(height - min_dimen) / 2 : height - (height - min_dimen) / 2,
			(width - min_dimen) / 2 : width - (width - min_dimen) / 2,
			: ]
		# Resize image to img_size x img_size
		img = cv2.resize(img, (img_size, img_size), interpolation = cv2.INTER_AREA)
		X[i] = img

	augmented_X = []
	augmented_Y = []
	for i in range(len(X)):
		#augment image by rotating 90, 180, and 270 degrees
		img1, img2, img3, img4 = process_image(X[i])
		augmented_X = augmented_X + [img1, img2, img3, img4]
		augmented_Y = augmented_Y + [Y[i]] * 4

	# for i in random.sample(range(len(augmented_X)), 20):
	# 	print "True label: %d" % augmented_Y[i]
	# 	cv2.imshow('image', augmented_X[i])
	# 	cv2.waitKey(0)
	# 	cv2.destroyAllWindows()

	data_stats(augmented_Y)

	return augmented_X, augmented_Y

def data_stats(labels):
	# Print some basic statistics about the dataset
	unique, counts = np.unique(labels, return_counts=True)
	print "%d training examples" % len(labels)
	print "Counts for each label:"
	for i in range(len(unique)):
		print "%d: %f" % (unique[i], float(counts[i]) / np.sum(counts))

def process_image(img):
	# Normalize image and convert to grayscale
	img = (img - np.mean(img, axis=(0,1), keepdims=True)) / np.std(img, axis=(0,1))
	img = color.rgb2gray(img)
	# Rotate image 90, 180, and 270 degrees to add more training examples
	height, width = img.shape
	M = cv2.getRotationMatrix2D((height/2, width/2), 90, 1)
	img2 = cv2.warpAffine(img, M, (height, width))
	img3 = cv2.warpAffine(img2, M, (height, width))
	img4 = cv2.warpAffine(img3, M, (height, width))
	img1 = np.expand_dims(img, 2)
	img2 = np.expand_dims(img2, 2)
	img3 = np.expand_dims(img3, 2)
	img4 = np.expand_dims(img4, 2)
	return img1, img2, img3, img4

def read_labeled_image_list(image_list_file):
    """Reads a .txt file containing pathes and labeles
    Args:
       image_list_file: a .txt file with one /path/to/image per line
       label: optionally, if set label will be pasted after each line
    Returns:
       List with all filenames in file image_list_file
    """
    f = open(image_list_file, 'r')
    filenames = []
    labels = []
    for line in f:
        filename, label = line[:-1].split(' ')
        #print filename, label
        filenames.append(filename)
        labels.append(int(label))
    return filenames, labels

def split_data(X, Y):
	# Shuffle data and split into train and dev sets (no test set is necessary at this stage)
	idxs = range(len(X))
	random.shuffle(idxs)
	X = [X[i] for i in idxs]
	Y = [Y[i] for i in idxs]
	X_train = X[:len(X) - dev_size]
	Y_train = Y[:len(Y) - dev_size]
	X_dev = X[len(X) - dev_size:]
	Y_dev = Y[len(Y) - dev_size:]

	return X_train, Y_train, X_dev, Y_dev

def setup():
	# Set up the network
	X_placeholder, Y_placeholder, dropout_placeholder = add_placeholders()
	out = build_network(X_placeholder, dropout_placeholder)
	predictions = tf.nn.softmax(out)
	loss = add_loss(out, Y_placeholder)
	train_op = add_train_op(loss, lr)

	sess = tf.Session()
	init = tf.global_variables_initializer()
	sess.run(init)

	return sess, train_op, loss, X_placeholder, Y_placeholder, out, predictions, dropout_placeholder

def add_placeholders():
	# Add placeholders to the graph
	X_placeholder = tf.placeholder(tf.float32, (None, img_size, img_size, 1))
	Y_placeholder = tf.placeholder(tf.int32, (None,))
	dropout_placeholder = tf.placeholder(tf.bool, ())

	return X_placeholder, Y_placeholder, dropout_placeholder

def build_network(X, is_training, output_dimen=11, scope="scope"):
	# Basic network: (conv -> pool -> norm) x 4 -> (dense -> dropout) x 2 -> output
	# Hyperparameters chosen arbitrarily, will tune later
	regularizer = tf.contrib.layers.l2_regularizer(scale=lamb)
	conv1 = tf.contrib.layers.conv2d(
		X,
		16,
		7,
		weights_regularizer = regularizer,
		scope=scope+'/conv1')
	pool1 = tf.contrib.layers.max_pool2d(
		conv1,
		2,
		scope=scope+'/pool1')
	norm1 = tf.contrib.layers.layer_norm(
		pool1,
		scope=scope+'/norm1')
	conv2 = tf.contrib.layers.conv2d(
		norm1,
		32,
		5,
		weights_regularizer = regularizer,
		scope=scope+'/conv2')
	pool2 = tf.contrib.layers.max_pool2d(
		conv2,
		2,
		scope=scope+'/pool2')
	norm2 = tf.contrib.layers.layer_norm(
		pool2,
		scope=scope+'/norm2')
	conv3 = tf.contrib.layers.conv2d(
		norm2,
		64,
		3,
		weights_regularizer = regularizer,
		scope=scope+'/conv3')
	pool3 = tf.contrib.layers.max_pool2d(
		conv3,
		2,
		scope=scope+'/pool3')
	norm3 = tf.contrib.layers.layer_norm(
		pool3,
		scope=scope+'/norm3')
	conv4 = tf.contrib.layers.conv2d(
		norm3,
		64,
		3,
		weights_regularizer = regularizer,
		scope=scope+'/conv4')
	pool4 = tf.contrib.layers.max_pool2d(
		conv4,
		2,
		scope=scope+'/pool4')
	norm4 = tf.contrib.layers.layer_norm(
		pool4,
		scope=scope+'/norm4')
	hidden1 = tf.contrib.layers.fully_connected(
		tf.contrib.layers.flatten(norm4),
		1024,
		weights_regularizer=regularizer,
		scope=scope+'/hidden1')
	dropout1 = tf.contrib.layers.dropout(
		hidden1,
		keep_prob,
		is_training=is_training)
	hidden2 = tf.contrib.layers.fully_connected(
		dropout1,
		512,
		weights_regularizer=regularizer,
		scope=scope+'/hidden2')
	dropout2 = tf.contrib.layers.dropout(
		hidden2,
		keep_prob,
		is_training=is_training)
	out = tf.contrib.layers.fully_connected(
		dropout2,
		output_dimen,
		weights_regularizer=regularizer,
		scope=scope+'/out',
		activation_fn=None)

	return out

def add_loss(out, labels):
	# Categorical cross-entropy loss
	loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
		logits=out,
		labels=labels)
	loss = tf.reduce_mean(loss)

	return loss

def add_train_op(loss, lr=0.001):
	# Add the training op to minimize loss
	optimizer = tf.train.AdamOptimizer(learning_rate=lr)
	train_op = optimizer.minimize(loss)

	return train_op

def train(sess, X, Y, X_placeholder, Y_placeholder, train_op, loss, out, predictions, dropout_placeholder):
	# Train on a minibatch of examples
	_, cost, out, predictions = sess.run((train_op, loss, out, predictions), feed_dict={
		X_placeholder: X,
		Y_placeholder: Y,
		dropout_placeholder: True
		})

	return out, cost, predictions

def eval(sess, predictions, X_placeholder, Y_placeholder, X, Y, dropout_placeholder):
	# Get predictions for a batch of examples and return accuracy compared to true labels
	# Also return incorrect examples, labels, and predictions for review
	preds = sess.run(predictions, feed_dict={
		X_placeholder: X,
		Y_placeholder: Y,
		dropout_placeholder: False
		})
	preds = np.argmax(preds, 1)
	correctly_predicted = np.sum(np.equal(preds, np.array(Y)))
	incorrect_idxs = np.nonzero(preds != np.array(Y))[0]
	incorrect_examples = [X[i] for i in incorrect_idxs]
	incorrect_labels = [Y[i] for i in incorrect_idxs]
	incorrect_predictions = [preds[i] for i in incorrect_idxs]
	accuracy = (100. * correctly_predicted) / preds.shape[0]

	return accuracy, incorrect_examples, incorrect_labels, incorrect_predictions

def minibatch(X, Y):
	# Randomly shuffle data into minibatches and return
	idxs = range(len(X))
	random.shuffle(idxs)
	batches = []
	for i in range(len(X) / batch_size + 1):
		batch_idxs = idxs[i*batch_size : min(i*batch_size+batch_size, len(X))]
		if len(batch_idxs) > 0:
			batches.append(([X[j] for j in batch_idxs], [Y[j] for j in batch_idxs]))
	return batches

def display_images(imgs, labels, preds):
	# Display images along with their true and predicted labels (for review)
	for i in range(min(len(imgs), 10)):
		print "True label: %d, predicted label: %d" % (labels[i], preds[i])
		cv2.imshow('image', imgs[i])
		cv2.waitKey(0)
		cv2.destroyAllWindows()

def process_arguments():
	parser = argparse.ArgumentParser()
	parser.add_argument('-e', '--epochs', type=int)
	parser.add_argument('-d', '--devsize', type=int)
	parser.add_argument('-b', '--batchsize', type=int)
	parser.add_argument('-p', '--printfreq', type=int)
	args = parser.parse_args()
	if args.epochs:
		global n_epochs
		n_epochs = args.epochs
	if args.devsize:
		global dev_size
		dev_size = args.devsize
	if args.batchsize:
		global batch_size
		batch_size = args.batchsize
	if args.printfreq:
		global print_freq
		print_freq = args.printfreq

if __name__ == '__main__':
	process_arguments()

	X, Y = load_data()
	X_train, Y_train, X_dev, Y_dev = split_data(X, Y)

	sess, train_op, loss, X_placeholder, Y_placeholder, out, predictions, dropout_placeholder = setup()
	max_train_accuracy = (0.0,0)
	max_dev_accuracy = (0.0,0)

	for epoch in range(n_epochs):
		epoch_cost = 0.0

		minibatches = minibatch(X_train, Y_train)
		for X_minibatch, Y_minibatch in minibatches:
			output, cost, preds = train(sess, X_minibatch, Y_minibatch, X_placeholder, Y_placeholder, train_op, loss, out, predictions, dropout_placeholder)
			epoch_cost += cost

		epoch_cost = epoch_cost / float(len(minibatches))

		train_accuracy, incorrect_train_examples, incorrect_train_labels, incorrect_train_preds = eval(sess, predictions, X_placeholder, Y_placeholder, X_train, Y_train, dropout_placeholder)
		dev_accuracy, incorrect_dev_examples, incorrect_dev_labels, incorrect_dev_preds = eval(sess, predictions, X_placeholder, Y_placeholder, X_dev, Y_dev, dropout_placeholder)
		if train_accuracy > max_train_accuracy[0]:
			max_train_accuracy = (train_accuracy, epoch+1)
		if dev_accuracy > max_dev_accuracy[0]:
			max_dev_accuracy = (dev_accuracy, epoch+1)
		if (epoch+1) % print_freq == 0:
			print "Average cost for epoch %d: %f" % (epoch+1, epoch_cost)
			print "Train accuracy for epoch %d: %.2f%%" % (epoch+1, train_accuracy)
			print "Dev accuracy for epoch %d: %.2f%%" % (epoch+1, dev_accuracy)
	print "Maximum train accuracy: %.2f in epoch %d" % max_train_accuracy
	print "Maximum dev accuracy: %.2f in epoch %d" % max_dev_accuracy
	# print "Displaying incorrectly classified train examples..."
	# display_images(incorrect_train_examples, incorrect_train_labels, incorrect_train_preds)
	# print "Displaying incorrectly classified dev examples..."
	# display_images(incorrect_dev_examples, incorrect_dev_labels, incorrect_dev_preds)