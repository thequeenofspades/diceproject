import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from utils import convert_label, nms, read_predictions
from PIL import Image, ImageDraw, ImageOps

threshold = 0.5
pred_path = 'predictions.txt'
img_path = 'examples/examples/'
save_path = 'examples/images/'
higher_quality = True
higher_quality_path = 'examples/higher_quality_dice/'

def load_image(path):
	image = Image.open(path)
	return image

def convert_to_higher_quality(img, img_name, labels):
	yolo_labels = [convert_label(img, label, 'bbox', 'PIL') for label in labels]
	higher_q_img = load_image(higher_quality_path + img_name + '.jpg')
	labels = [convert_label(higher_q_img, label, img_fm='PIL') for label in yolo_labels]
	return higher_q_img, labels

def draw_bbox(img, x1, y1, x2, y2, width = 10):
	for i in range(width):
		ImageDraw.Draw(img).rectangle([x1 - i, y1 - i, x2 + i, y2 + i], outline=(0, 255, 0))
	return img

def draw_bboxes(img, labels, fm='yolo'):
	if fm == 'yolo':
		labels = [convert_label(img, label) for label in labels]
	for i in range(len(labels)):
		label = labels[i]
		img = draw_bbox(img, label[0], label[1], label[2], label[3])
	return img

def extract_crops(img, labels):
	crops = []
	for i in range(len(labels)):
		x1, y1, x2, y2 = labels[i]
		cropped = img.crop((x1, y1, x2, y2))
		cropped = square_crop(cropped)
		crops.append(cropped)
	return crops

def square_crop(img):
	# Make crop square by adding borders
	new_size = max(img.size)
	delta_w = new_size - img.size[0]
	delta_h = new_size - img.size[1]
	padding = (delta_w//2, delta_h//2, delta_w - delta_w//2, delta_h - delta_h//2)
	new_img = ImageOps.expand(img, padding)
	return new_img

def save_crops(crops, path):
	for i in range(len(crops)):
		crops[i].save(path + '_' + str(i) + '.jpg')

def main():
	preds = read_predictions(pred_path, threshold)
	for img_name in sorted(preds.keys()):
		img = load_image(img_path + img_name + '.jpg')
		img_preds = nms(preds[img_name])
		if higher_quality:
			img, img_preds = convert_to_higher_quality(img, img_name, img_preds)
		img_preds = [pred[1:] for pred in img_preds]
		crops = extract_crops(img, img_preds)
		save_crops(crops, save_path + img_name)
		img = draw_bboxes(img, img_preds, 'bbox')
		img.save(save_path + img_name + '.jpg')

if __name__ == '__main__':
	main()