from PIL import Image
from os import listdir


# Display each unlabeled image from the training set and accept an integer label as input
# Save labels and corresponding filename to labels.txt

if __name__ == '__main__':
	img_path = 'examples/images/'
	labels_path = 'examples/labels/'
	img_paths = sorted([path for path in listdir(img_path) if path.endswith('.jpg')])
	done_labels = [path for path in listdir(labels_path) if path.endswith('.txt')]
	for i in range(len(done_labels), len(img_paths)):
		img_filepath = img_paths[i]
		img_name = img_filepath[:-len('.jpg')]
		label_path = labels_path + img_name + '.txt'
		label_file = open(label_path, 'w+')
		img = Image.open(img_path + img_filepath)
		img.resize((100, 100)).show()
		print 'Labeling %s' % (img_name)
		die_type = raw_input('What is the type? ')
		die_value = raw_input('What is the value? ')
		label_file.write('%d %d' % (int(die_type), int(die_value)))
		img.close()
		label_file.close()