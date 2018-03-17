class config():
	#Data params
	dev_frac = 0.1
	test_frac = 0.05
	img_size = 100
	n_channels = 1

	#Training params
	batches = 500
	batch_size = 16
	save_freq = 100
	print_freq = 1
	eval_freq = 10
	save_path = 'weights/'

	#Type NN params
	type_class_mapping = {4: 0, 6: 1, 8: 2, 10: 3, 100: 4, 12: 5, 20: 6}
	type_num_classes = 7
	type_lamb = 0.1
	type_keep_prob = 0.5
	type_lr = 0.001
	type_layers = 3

	#Value NN params
	val_class_mapping = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9,
						11: 10, 12: 11, 13: 12, 14: 13, 15: 14, 16: 15, 17: 16,
						18: 17, 19: 18, 20: 19}
	val_num_classes = 20
	val_lamb = 0.1
	val_keep_prob = 0.8
	val_lr = 0.01
	val_layers = 3
	val_save_path = 'val/'
	val_weights_to_restore = None