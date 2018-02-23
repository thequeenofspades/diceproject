from os import listdir, rename

# Automatically move screenshots into the training data folder and rename them to their training index
# Training examples are not yet labeled at this step

if __name__ == '__main__':
	screenshots_dir = '/Users/shalinmehta/Documents/Hana/Stanford/dice_project/screenshots'
	cleaned_dir = '/Users/shalinmehta/Documents/Hana/Stanford/dice_project/cleaned_dice_pics'
	screenshots = listdir(screenshots_dir)
	cleaned_idx = len(listdir(cleaned_dir))
	for i in range(len(screenshots)):
		screenshot = screenshots[i]
		zero_padding = 5 - len(str(cleaned_idx+i))
		new_name = '0' * zero_padding + str(cleaned_idx+i) if cleaned_idx+i > 0 else '0' * 5
		new_path = cleaned_dir + '/' + new_name + '.png'
		rename(screenshots_dir + '/' + screenshot, new_path)