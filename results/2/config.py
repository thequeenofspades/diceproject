class config():
	#Data params
	dev_frac = 0.1
	test_frac = 0.05
	img_size = 100
	n_channels = 1

	#Training params
	batches = 5000
	batch_size = 16
	save_freq = 1000
	print_freq = 10
	eval_freq = 100
	save_path = 'weights/'

	#Type NN params
	type_class_mapping = {4: 0, 6: 1, 8: 2, 10: 3, 100: 4, 12: 5, 20: 6}
	type_num_classes = 7
	type_lamb = 0.1
	type_keep_prob = 0.8
	type_lr = 0.01
	type_layers = 5
	type_layer_sizes = [16, 32, 64, 128, 512]
	type_filter_sizes = [11, 9, 7, 5, 3]
	type_save_path = 'type/'
	type_weights_to_restore = None

	#Value NN params
	val_class_mapping = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9,
						11: 10, 12: 11, 13: 12, 14: 13, 15: 14, 16: 15, 17: 16,
						18: 17, 19: 18, 20: 19}
	val_num_classes = 20
	val_lamb = None
	val_keep_prob = None
	val_lr = 0.01
	val_norm_decay = 0.9
	val_layers = 3
	val_layer_sizes = [16,32,64]#[16, 32, 64, 128, 512]
	val_filter_sizes = [5,5,5]#[11, 9, 7, 5, 3]
	val_hidden_size = 1024
	val_use_batch_norm = True
	val_save_path = 'val/'
	val_weights_to_restore = None
