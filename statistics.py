import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from utils import convert_label, nms, read_predictions

threshold = 0.5
pred_path = 'predictions.txt'
label_path = 'examples/examples/'
img_path = 'examples/examples/'
verbose = 1

def read_labels(name):
	path = label_path + name + '.txt'
	label_file = open(path, 'r')
	labels = []
	for line in label_file:
		label = line.split()[1:]
		labels.append([float(x) for x in label])
	return labels

def iou(bbox, label):
	_, x1, y1, x2, y2 = bbox
	_, x1p, y1p, x2p, y2p = label
	bbox_area = (x2 - x1) * (y2 - y1)
	label_area = (x2p - x1p) * (y2p - y1p)
	x1pp = max(x1, x1p)
	y1pp =  max(y1, y1p)
	x2pp = min(x2, x2p)
	y2pp = min(y2, y2p)
	intersection_area = max(0, (x2pp - x1pp) * (y2pp - y1pp))
	union_area = bbox_area + label_area - intersection_area
	intersection_over_union = intersection_area / float(union_area)
	return intersection_over_union

def center(bbox):
	_, x1, y1, x2, y2 = bbox
	x = x1 + (x2 - x1) / 2
	y = y1 + (y2 - y1) / 2
	return [x, y]

# def closest_center(x, y, labels):
# 	centers = [center(label) for label in labels]
# 	closest_dist = 10000
# 	closest_idx = None
# 	for i in range(len(centers)):
# 		xp, yp = centers[i]
# 		dist = np.sqrt((x - xp)**2 + (y - yp)**2)
# 		if dist < closest_dist:
# 			closest_dist = dist
# 			closest_idx = i
# 	return labels[closest_idx]

def closest_label(bbox, labels):
	closest_dist = 10000
	closest = None
	if verbose > 1:
		print "Finding closest label to %s" % str(bbox[1:])
	_, x1, y1, x2, y2 = bbox
	for label in labels:
		_, x1p, y1p, x2p, y2p = label
		dist = np.sqrt((x1p - x1)**2 + (y1p - y1)**2 + (x2p - x2)**2 + (y2p - y2)**2)
		if verbose > 1:
			print "--->%s: %f" % (str(label[1:]), dist)
		if dist < closest_dist:
			closest_dist = dist
			closest = label
	_, x1p, y1p, x2p, y2p = closest
	x, y = center(bbox)
	if x > x1p and x < x2p and y > y1p and y < y2p:
		fp = 0
	else:
		fp = 1
	return closest, fp

def avg_iou(bboxes, labels):
	total_iou = 0.0
	num_fp = 0
	num_tp = 0
	num_fn = 0
	closest_labels = []
	for bbox in bboxes:
		# x, y = center(bbox)
		# closest = closest_center(x, y, labels)
		closest, fp = closest_label(bbox, labels)
		closest_labels.append(closest)
		num_fp += fp
		num_tp += 1 - fp
		if verbose > 1:
			print "Found match for %s: %s" % (str(bbox[1:]), str(closest[1:]))
		total_iou += iou(bbox, closest)
	for label in labels:
		found = False
		for closest in closest_labels:
			_, x1, y1, x2, y2 = label
			_, x1p, y1p, x2p, y2p = closest
			if x1 == x1p and y1 == y1p and x2 == x2p and y2 == y2p:
				found = True
		if not found:
			num_fn += 1
	avg_precision = num_tp / float(num_fp + num_tp)
	avg_recall = num_tp / float(num_tp + num_fn)
	return total_iou / float(len(bboxes)), avg_precision, avg_recall

def img_stats(img_name, preds):
	if verbose > 0:
		print "Getting statistics for %s" % img_name
	img = plt.imread(img_path + img_name + '.jpg')
	labels = read_labels(img_name)
	abs_labels = [convert_label(img, label) for label in labels]
	nms_preds = nms(preds)
	if verbose > 0:
		print "Found %d predictions after NMS" % len(nms_preds)
	if verbose > 0:
		print "True number of objects: %d" % len(labels)
	average_iou, avg_precision, avg_recall = avg_iou(nms_preds, abs_labels)
	if verbose > 0:
		print "Average IoU: %f" % average_iou
		print "Average precision: %f" % avg_precision
		print "Average recall: %f" % avg_recall
		print ""
	return len(nms_preds), average_iou, avg_precision, avg_recall

def main():
	preds = read_predictions(pred_path, threshold)
	total_preds = 0
	total_iou = 0.0
	total_precision = 0.0
	total_recall = 0.0
	for img_name in sorted(preds.keys()):
		num_preds, average_iou, average_precision, average_recall = img_stats(img_name, preds[img_name])
		total_preds += num_preds
		total_iou += average_iou * num_preds
		total_precision += average_precision * num_preds
		total_recall += average_recall * num_preds
	average_iou = total_iou / float(total_preds)
	average_precision = total_precision / float(total_preds)
	average_recall = total_recall / float(total_preds)
	print "Total predictions: %d" % total_preds
	print "Average IoU: %f" % average_iou
	print "Average precision: %f" % average_precision
	print "Average recall: %f" % average_recall

if __name__ == '__main__':
	main()