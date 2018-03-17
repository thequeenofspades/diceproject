import numpy as np
from value_nn import ValueNN
from config import config
from PIL import Image, ImageOps
from os import listdir
import random

img_path = 'examples/images/'
label_path = 'examples/labels/'

def load_train_data(mode='val'):
	print "Loading image and label data..."
	imgs, labels = load_data('train.txt', mode)
	analyze_labels(labels)
	print "Resizing and processing images..."
	imgs = resize_imgs(imgs)
	imgs = process_image(imgs)
	print "Augmenting images..."
	imgs, labels = augment(imgs, labels)
	imgs = np.array([np.array(img).reshape(config.img_size, config.img_size, config.n_channels) for img in imgs])
	labels = np.array(labels)
	return imgs, labels

def load_dev_data(mode='val'):
	print "Loading image and label data..."
	imgs, labels = load_data('dev.txt', mode)
	print "Resizing and processing images..."
	imgs = resize_imgs(imgs)
	imgs = process_image(imgs)
	imgs = np.array([np.array(img).reshape(config.img_size, config.img_size, config.n_channels) for img in imgs])
	labels = np.array(labels)
	return imgs, labels

def load_data(path, mode='val'):
	data_file = open(path, 'r')
	imgs = []
	labels = []
	for line in data_file:
		img_name = line.split()[0]
		img = Image.open(img_path + img_name + '.jpg')
		img.load()
		imgs.append(img)
		label_file = open(label_path + img_name + '.txt', 'r')
		label = [line for line in label_file][0].split()
		label_file.close()
		if mode == 'val':
			label = config.val_class_mapping[int(label[1])]
		elif mode == 'type':
			label = config.type_class_mapping[int(label[0])]
		labels.append(label)
	data_file.close()
	return imgs, labels

def resize_imgs(imgs):
	new_imgs = [img.resize((config.img_size, config.img_size)) for img in imgs]
	return new_imgs

def process_image(imgs):
	new_imgs = [img.convert('L') for img in imgs]
	new_imgs = [ImageOps.equalize(img) for img in new_imgs]
	return new_imgs

def augment(imgs, labels):
	new_imgs = []
	new_labels = []
	for i in range(len(imgs)):
		new_imgs.append(imgs[i])
		new_imgs.append(imgs[i].rotate(90))
		new_imgs.append(imgs[i].rotate(180))
		new_imgs.append(imgs[i].rotate(270))
		new_labels = new_labels + [labels[i]] * 4
	return new_imgs, new_labels

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
	X_train, Y_train = load_train_data('val')
	X_dev, Y_dev = load_dev_data('val')
	nn.train(X_train, Y_train, config.batches)
	print "Dev accuracy: %f" % nn.validate(X_dev, Y_dev)