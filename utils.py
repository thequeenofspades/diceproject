import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from os import listdir
from config import config
import random

def analyze_labels(labels):
	print "Before augmentation: %d examples" % len(labels)
	counts = {}
	for label in labels:
		value = label
		if value in counts:
			counts[value] += 1
		else:
			counts[value] = 1
	counts = {k: counts[k] / float(len(labels)) for k in counts}
	print counts

def augment(imgs, labels):
	new_imgs = []
	new_labels = []
	for i in range(len(imgs)):
		for j in range(4):
			new_imgs.append(imgs[i].rotate(90*i))
			new_labels.append(labels[i])
	imgs = new_imgs
	labels = new_labels
	new_imgs = []
	new_labels = []
	for i in range(len(imgs)):
		new_imgs.append(imgs[i])
		new_labels.append(labels[i])
		for j in range(4):
			x1 = random.randint(-(imgs[i].width / 5), imgs[i].width / 5)
			y1 = random.randint(-(imgs[i].height / 5), imgs[i].height / 5)
			x2 = random.randint(4 * imgs[i].width / 5, 6 * imgs[i].width / 5)
			y2 = y1 + (x2 - x1)
			borders = [0,0,0,0]
			if x1 < 0:
				borders[0] = -x1
			if y1 < 0:
				borders[1] = -y1
			if x2 > imgs[i].width + 1:
				borders[2] = x2 - (imgs[i].width + 1)
			if y2 > imgs[i].height + 1:
				borders[3] = y2 - (imgs[i].height + 1)
			new_img = ImageOps.expand(imgs[i], tuple(borders))
			x1 = x1 + borders[0]
			y1 = y1 + borders[1]
			x2 = x2 + borders[0]
			y2 = y2 + borders[1]
			new_img = new_img.crop((x1, y1, x2, y2))
			assert new_img.height == new_img.width
			new_img = new_img.resize((config.img_size, config.img_size))
			# max_size = min((6 * imgs[i].width / 5) - x1, (6 * imgs[i].height / 5) - y1)
			# min_size = max(3 * imgs[i].width / 5 - x1, 3 * imgs[i].height / 5 - y1)
			# size = random.randint(min_size, max_size)
			# x2 = x1 + size
			# y2 = y1 + size
			# new_img = imgs[i].crop((x1, y1, x1 + size, y1 + size))
			# new_img = new_img.resize((config.img_size, config.img_size))
			new_imgs.append(new_img)
			new_labels.append(labels[i])
	return new_imgs, new_labels

def convert_label(img, label, fm='yolo', img_fm=None):
	if img_fm == 'PIL':
		img_width = img.width
		img_height = img.height
	else:
		img_width = img.shape[1]
		img_height = img.shape[0]
	if fm == 'yolo':
		x, y, width, height = label
		x = img_width * x
		y = img_height * y
		width = img_width * width
		height = img_height * height
		x_start = int(max(0, x - width/2))
		x_end = int(min(img_width - 1, x + width/2))
		y_start = int(max(0, y - height/2))
		y_end = int(min(img_height - 1, y + height/2))
		return [None, x_start, y_start, x_end, y_end]
	else:
		_, x_start, y_start, x_end, y_end = label
		x = x_start + (x_end - x_start) / 2
		y = y_start + (y_end - y_start) / 2
		x = x / float(img_width)
		y = y / float(img_height)
		width = (x_end - x_start) / float(img_width)
		height = (y_end - y_start) / float(img_height)
		return [x, y, width, height]

def load_train_data(img_path, label_path, mode='val', exclude=[]):
	print "Loading image and label data from train..."
	imgs, orig_labels, type_labels = load_data(img_path, label_path, 'train.txt', mode, exclude)
	print "Resizing images..."
	orig_imgs = resize_imgs(imgs)
	print "Augmenting images..."
	imgs, labels = augment(orig_imgs, orig_labels)
	print "Processing images..."
	imgs, data_center = process_image(imgs)
	orig_imgs, _ = process_image(orig_imgs)
	#preview_imgs(imgs, 20)
	labels = np.array(labels)
	print "After augmentation: %d examples" % labels.shape[0]
	return imgs, labels, data_center, orig_imgs, orig_labels, type_labels

def load_dev_data(img_path, label_path, data_center, mode='val', exclude=[]):
	print "Loading image and label data from dev..."
	imgs, labels, type_labels = load_data(img_path, label_path, 'dev.txt', mode, exclude)
	print "Resizing and processing images..."
	imgs = resize_imgs(imgs)
	imgs, _ = process_image(imgs, [data_center])
	labels = np.array(labels)
	return imgs, labels, type_labels

def load_test_data(img_path, label_path, mode='val', exclude=[]):
	print "Loading image and label data from test..."
	imgs, labels, type_labels = load_data(img_path, label_path, 'test.txt', mode, exclude)
	print "Resizing and processing images..."
	imgs = resize_imgs(imgs)
	imgs, _ = process_image(imgs)
	imgs = np.array([np.array(img).reshape(config.img_size, config.img_size, config.n_channels) for img in imgs])
	labels = np.array(labels)
	return imgs, labels, type_labels

def load_data(img_path, label_path, path, mode='val', exclude=[]):
	data_file = open(path, 'r')
	imgs = []
	labels = []
	type_labels = []
	for line in data_file:
		img_name = line.split()[0]
		img = Image.open(img_path + img_name + '.jpg')
		img.load()
		label_file = open(label_path + img_name + '.txt', 'r')
		label = [line for line in label_file][0].split()
		label_file.close()
		if mode == 'val':
			die_type = int(label[0])
			if die_type not in exclude:
				labels.append(config.val_class_mapping[int(label[1])])
				imgs.append(img)
		elif mode == 'type':
			labels.append(config.type_class_mapping[int(label[0])])
			imgs.append(img)
		type_labels.append(config.type_class_mapping[int(label[0])])
	data_file.close()
	analyze_labels(labels)
	return imgs, labels, type_labels

def nms(bboxes):
	nms_bboxes = []
	bboxes = sorted(bboxes, key=lambda x: x[0], reverse=True)
	for bbox in bboxes:
		suppress = False
		for nms_bbox in nms_bboxes:
			_, x1, y1, x2, y2 = bbox
			_, x1p, y1p, x2p, y2p = nms_bbox
			x = x1 + (x2 - x1) / 2
			y = y1 + (y2 - y1) / 2
			if x > x1p and x < x2p and y > y1p and y < y2p:
				suppress = True
		if not suppress:
			nms_bboxes.append(bbox)
	return nms_bboxes

def preview_imgs(imgs, n=10):
	preview = np.random.choice(range(len(imgs)), n, replace=False)
	preview = [imgs[i] for i in preview]
	for img in preview:
		print np.array(img).reshape(config.img_size, config.img_size, config.n_channels)
		img.show()

def process_image(imgs, data_center = None):
	new_imgs = [img.convert('L') for img in imgs]
	new_imgs = np.array([np.array(img).reshape(config.img_size, config.img_size, config.n_channels) for img in new_imgs])
	if data_center == None:
		data_center = [np.mean(new_imgs, axis=0)]
	new_imgs = new_imgs - data_center[0]
	#new_imgs = [ImageOps.equalize(img) for img in new_imgs]
	return new_imgs, data_center

def read_predictions(path, threshold):
	pred_file = open(path, 'r')
	predictions = {}
	for line in pred_file:
		pred = line.split()
		if pred[0] not in predictions:
			predictions[pred[0]] = []
		if float(pred[1]) > threshold:
			predictions[pred[0]].append([float(pred[1])] + [int(float(x)) for x in pred[2:]])
	return predictions

def resize_imgs(imgs):
	new_imgs = [img.resize((config.img_size, config.img_size)) for img in imgs]
	return new_imgs