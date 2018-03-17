import tensorflow as tf
import numpy as np

class ValueNN():
	def __init__(self, config):
		self.config = config
		self.sess = tf.Session()
		self.global_step = tf.Variable(tf.constant(0), trainable=False, name='global_step')
		tf.add_to_collection('global_step', self.global_step)

		#Add input and output
		self.add_placeholders()
		self.network()

		#Add loss and train op
		self.add_loss()
		self.add_train_op()

		#Save/restore or initialize variables
		self.setup()

	def add_placeholders(self):
		self.X_placeholder = tf.placeholder(tf.float32, (None, self.config.img_size, self.config.img_size, self.config.n_channels))
		self.Y_placeholder = tf.placeholder(tf.int32, (None,))
		self.dropout_placeholder = tf.placeholder(tf.bool, ())

	def network(self):
		# regularizer = tf.contrib.layers.l2_regularizer(scale=self.config.val_lamb)
		layers = [self.X_placeholder]
		layer_sizes = self.config.val_layer_sizes
		filter_sizes = self.config.val_filter_sizes
		for i in range(self.config.val_layers):
			conv_layer = tf.contrib.layers.conv2d(
				layers[i - 1],
				layer_sizes[i],
				filter_sizes[i])
				# weights_regularizer=regularizer)
			pool_layer = tf.contrib.layers.max_pool2d(
				conv_layer,
				2)
			norm_layer = tf.contrib.layers.layer_norm(
				pool_layer)
			layers.append(norm_layer)
		hidden = tf.contrib.layers.fully_connected(
			tf.contrib.layers.flatten(layers[-1]),
			512)
			# weights_regularizer=regularizer)
		# dropout = tf.contrib.layers.dropout(
		# 	hidden,
		# 	self.config.val_keep_prob,
		# 	is_training=self.dropout_placeholder)
		self.output = tf.contrib.layers.fully_connected(
			hidden,
			self.config.val_num_classes,
			# weights_regularizer=regularizer,
			activation_fn=None)
		self.preds = tf.nn.softmax(self.output)

	def add_loss(self):
		loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
			logits=self.output,
			labels=self.Y_placeholder)
		self.loss = tf.reduce_mean(loss)

	def add_train_op(self):
		optimizer = tf.train.AdamOptimizer(learning_rate=self.config.val_lr)
		self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)

	def setup(self):
		self.saver = tf.train.Saver(max_to_keep=None)
		ckpt = tf.train.get_checkpoint_state(self.config.save_path + self.config.val_save_path)
		if ckpt and ckpt.model_checkpoint_path:
			model_checkpoint_path = ckpt.model_checkpoint_path
			if self.config.val_weights_to_restore != None:
				model_checkpoint_path = self.config.save_path + self.config.val_save_path + 'model_step.ckpt-' + str(self.config.val_weights_to_restore)
			self.saver.restore(self.sess, model_checkpoint_path)
			print "Restored weights from %s" % model_checkpoint_path
			print "Global step: %d" % (tf.train.global_step(self.sess, self.global_step))
		else:
			init = tf.global_variables_initializer()
			self.sess.run(init)

	def save(self):
		print('Saving to {} with global step {}'.format(self.config.save_path + self.config.val_save_path + 'model_step.ckpt', tf.train.global_step(self.sess, self.global_step)))
		self.saver.save(self.sess, self.config.save_path + self.config.val_save_path + 'model_step.ckpt', global_step=self.global_step)

	def accuracy(self, preds, Y):
		preds = np.argmax(preds, 1)
		correctly_predicted = np.sum(np.equal(preds, np.array(Y)))
		accuracy = (100. * correctly_predicted) / preds.shape[0]
		return accuracy

	def predict(self, X):
		batches = []
		num_batches = len(X) / self.config.batch_size
		if num_batches * self.config.batch_size < len(X):
			num_batches = num_batches + 1
		preds = np.zeros((len(X), self.config.val_num_classes))
		for i in range(num_batches):
			batch = range(len(X))[i*self.config.batch_size:(i+1)*self.config.batch_size]
			batch_preds = self.sess.run((self.preds), feed_dict={
				self.X_placeholder: X[batch],
				self.dropout_placeholder: False})
			preds[batch] = batch_preds
		return preds

	def validate(self, X, Y):
		preds = self.predict(X)
		accuracy = self.accuracy(preds, Y)
		preds = np.argmax(preds, 1)
		return preds, accuracy

	def train(self, X, Y, steps=100):
		losses = []
		batch_nums = []
		for step in range(steps):
			idx = np.random.choice(range(len(X)), self.config.batch_size, replace=False)
			loss, preds, _ = self.sess.run((self.loss, self.preds, self.train_op), feed_dict={
				self.X_placeholder: X[idx],
				self.Y_placeholder: Y[idx],
				self.dropout_placeholder: True})
			losses.append(loss)
			batch_nums.append(tf.train.global_step(self.sess, self.global_step))
			avg_loss = sum(losses[-100:]) / len(losses[-100:])
			if (step + 1) % self.config.print_freq == 0:
				print "batch: %d, loss: %f, avg loss: %f, accuracy: %f" % (batch_nums[-1], loss, avg_loss, self.accuracy(preds, Y[idx]))
				preds = np.argmax(preds, 1)
				print "predictions: %s" % str(preds)
			if batch_nums[-1] % self.config.save_freq == 0:
				self.save()
		return losses, batch_nums