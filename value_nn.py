import tensorflow as tf
import numpy as np
from nn import NN

class ValueNN(NN):
	def network(self):
		regularizer = tf.contrib.layers.l2_regularizer(scale=self.config.val_lamb)
		layers = [self.X_placeholder]
		for i in range(self.config.val_conv_layers):
			with tf.variable_scope('block_{}'.format(i+1)):
				out = tf.layers.conv2d(
					layers[i],
					self.config.val_layer_sizes[i],
					self.config.val_filter_sizes[i],
					padding='same',
					kernel_initializer=tf.contrib.layers.xavier_initializer(),
					kernel_regularizer=regularizer)
				if self.config.val_use_batch_norm:
					out = tf.layers.batch_normalization(
						out,
						momentum=self.config.val_norm_decay,
						training=self.dropout_placeholder)
				out = tf.nn.relu(out)
				out = tf.layers.max_pooling2d(out, 2, 2)
				layers.append(out)
		out_flat = tf.reshape(layers[-1], (-1, (self.config.img_size / (2*self.config.val_conv_layers))**2 * self.config.n_channels))
		layers = [out_flat]
		for i in range(self.config.val_fc_layers):
			with tf.variable_scope('fc_{}'.format(i+1)):
				out = tf.layers.dense(
					layers[i],
					self.config.val_fc_sizes[i],
					kernel_initializer=tf.contrib.layers.xavier_initializer(),
					kernel_regularizer=regularizer)
				if self.config.val_use_batch_norm:
					out = tf.layers.batch_normalization(
						out,
						momentum=self.config.val_norm_decay,
						training=self.dropout_placeholder)
				if self.config.val_keep_prob != None:
					out = tf.contrib.layers.dropout(
						out,
						self.config.val_keep_prob,
						is_training=self.dropout_placeholder)
				out = tf.nn.relu(out)
		with tf.variable_scope('fc2'):
			self.output = tf.layers.dense(
				layers[-1],
				self.config.val_num_classes,
				kernel_initializer=tf.contrib.layers.xavier_initializer(),
				kernel_regularizer=regularizer,
				activation=None)
			self.preds = tf.nn.softmax(self.output)

	def add_train_op(self):
		optimizer = tf.train.AdamOptimizer(learning_rate=self.config.val_lr)
		if self.config.val_use_batch_norm:
			# Add a dependency to update the moving mean and variance for batch normalization
			with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
				self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)
		else:
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