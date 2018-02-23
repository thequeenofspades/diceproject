from PIL import Image
from os import listdir

# Display each unlabeled image from the training set and accept an integer label as input
# Save labels and corresponding filename to labels.txt

if __name__ == '__main__':
	cleaned_dir = '/Users/shalinmehta/Documents/Hana/Stanford/dice_project/cleaned_dice_pics'
	labels_filename = '/Users/shalinmehta/Documents/Hana/Stanford/dice_project/labels.txt'
	labels_file = open(labels_filename, 'a+')
	labels_file_read = open(labels_filename, 'r')
	labels_idx = len(labels_file_read.readlines())+1
	labels_file_read.close()
	print "Starting at index %d" % labels_idx
	examples = sorted([f for f in listdir(cleaned_dir) if f.endswith('.png')])
	labels = []
	for i in range(labels_idx, len(examples)):
		img = Image.open(cleaned_dir + '/' + examples[i])
		img.show()
		label = int(raw_input('Label: '))
		labels.append(label)
		img.close()
	for i in range(len(labels)):
		if labels_idx+i > 0:
			labels_file.write('\n')
		labels_file.write('./cleaned_dice_pics/' + examples[labels_idx+i] + ' ' + str(labels[i]))
	labels_file.close()