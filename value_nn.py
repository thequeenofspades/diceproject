import tensorflow as tf
import numpy as np
from nn import NN

class ValueNN(NN):
	def network(self):
		if self.config.val_lamb != None:	
			regularizer = tf.contrib.layers.l2_regularizer(scale=self.config.val_lamb)
		else:
			regularizer = None
		layers = [self.X_placeholder]
		layer_sizes = self.config.val_layer_sizes
		filter_sizes = self.config.val_filter_sizes
		hidden_size = self.config_val_hidden_size
		for i in range(self.config.val_layers):
			conv_layer = tf.contrib.layers.conv2d(
				layers[i - 1],
				layer_sizes[i],
				filter_sizes[i],
				weights_regularizer=regularizer)
			pool_layer = tf.contrib.layers.max_pool2d(
				conv_layer,
				2)
			norm_layer = tf.contrib.layers.batch_norm(
				pool_layer,
				decay=self.config.val_norm_decay,
				updates_collections=None,
				is_training=self.dropout_placeholder)
			layers.append(norm_layer)
		hidden = tf.contrib.layers.fully_connected(
			tf.contrib.layers.flatten(layers[-1]),
			hidden_size,
			weights_regularizer=regularizer)
		if self.config.val_dropout != None:
			dropout = tf.contrib.layers.dropout(
				hidden,
				self.config.val_keep_prob,
				is_training=self.dropout_placeholder)
		else:
			dropout = hidden
		self.output = tf.contrib.layers.fully_connected(
			hidden,
			self.config.val_num_classes,
			weights_regularizer=regularizer,
			activation_fn=None)
		self.preds = tf.nn.softmax(self.output)

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