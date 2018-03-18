import tensorflow as tf
import numpy as np

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

	def train(self, X, Y, steps=100, losses=[]):
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