import tensorflow as tf
import numpy as np
import random

class NN:
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

		#Initialize minibatches
		self.minibatch_idx = None
		self.minibatches = []

	def get_global_step(self):
		return tf.train.global_step(self.sess, self.global_step)

	def add_placeholders(self):
		self.X_placeholder = tf.placeholder(tf.float32, (None, self.config.img_size, self.config.img_size, self.config.n_channels))
		self.Y_placeholder = tf.placeholder(tf.int32, (None,))
		self.dropout_placeholder = tf.placeholder(tf.bool, ())

	def add_loss(self):
		loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
			logits=self.output,
			labels=self.Y_placeholder)
		self.loss = tf.reduce_mean(loss) + tf.losses.get_regularization_loss()

	def accuracy(self, preds, Y):
		preds = np.argmax(preds, 1)
		correctly_predicted = np.sum(np.equal(preds, np.array(Y)))
		accuracy = (100. * correctly_predicted) / preds.shape[0]
		return accuracy

	def validate(self, X, Y):
		preds = self.predict(X)
		accuracy = self.accuracy(preds, Y)
		preds = np.argmax(preds, 1)
		return preds, accuracy

	def get_train_minibatch(self):
		num_minibatches = int(self.X.shape[0] / self.config.batch_size)
		if self.config.batch_size * num_minibatches < self.X.shape[0]:
			num_minibatches += 1
		if self.minibatch_idx == None:
			minibatch_idxs = range(self.X.shape[0])
			random.shuffle(minibatch_idxs)
			for i in range(num_minibatches):
				self.minibatches.append(minibatch_idxs[i*self.config.batch_size:(i+1)*self.config.batch_size])
			self.minibatch_idx = 0
		X = self.X[self.minibatches[self.minibatch_idx]]
		Y = self.Y[self.minibatches[self.minibatch_idx]]
		self.minibatch_idx = (self.minibatch_idx + 1) % num_minibatches
		return X, Y

	def train(self, X, Y, steps=100, losses=[]):
		self.X = X
		self.Y = Y
		batch_nums = []
		for step in range(steps):
			# idx = np.random.choice(range(len(X)), self.config.batch_size, replace=False)
			X, Y = self.get_train_minibatch()
			loss, preds, _ = self.sess.run((self.loss, self.preds, self.train_op), feed_dict={
				self.X_placeholder: X,
				self.Y_placeholder: Y,
				self.dropout_placeholder: True})
			losses.append(loss)
			batch_nums.append(tf.train.global_step(self.sess, self.global_step))
			avg_loss = sum(losses[-100:]) / len(losses[-100:])
			if (step + 1) % self.config.print_freq == 0:
				print "batch: %d, loss: %f, avg loss: %f, accuracy: %f" % (batch_nums[-1], loss, avg_loss, self.accuracy(preds, Y))
				preds = np.argmax(preds, 1)
				print "predictions: %s" % str(preds)
		return losses, batch_nums