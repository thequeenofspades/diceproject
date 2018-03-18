import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

if __name__ == '__main__':
	save_path = 'results/reg/'
	eval_file = open(save_path + 'eval_val.txt', 'r')
	losses_file = open(save_path + 'losses_val.txt', 'r')
	eval_batch = []
	train_eval = []
	dev_eval = []
	losses = []
	losses_batch = []
	for line in eval_file:
		batch, train_acc, dev_acc = line.split()
		eval_batch.append(int(batch))
		train_eval.append(float(train_acc))
		dev_eval.append(float(dev_acc))
	eval_file.close()
	for line in losses_file:
		batch, loss = line.split()
		losses_batch.append(int(batch))
		losses.append(float(loss))
	losses_file.close()

	# Plot accuracy
	train_accs, = plt.plot(eval_batch, train_eval, label='Train accuracy')
	dev_accs, = plt.plot(eval_batch, dev_eval, label='Dev accuracy')
	plt.legend(handles=[train_accs, dev_accs])
	plt.title('Prediction accuracy')
	plt.xlabel('Batch')
	plt.ylabel('Accuracy')
	plt.savefig(save_path + 'eval_val.png')

	plt.gcf().clear()

	# Get moving average for losses
	avg_losses = [losses[0]]
	for i in range(1, len(losses)):
		avg_losses.append(sum(losses[max(0, i-50):i]) / float(len(losses[max(0, i-50):i])))

	# Plot loss
	plt.plot(losses_batch, avg_losses)
	plt.title('Training loss')
	plt.xlabel('Batch')
	plt.ylabel('Loss (averaged over last 50)')
	plt.yscale('log')
	plt.savefig(save_path + 'losses_val.png')