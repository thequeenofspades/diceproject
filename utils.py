import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

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