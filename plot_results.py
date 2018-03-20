import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import cPickle as pickle

if __name__ == '__main__':
	save_path = 'results/5/'
	evals_pkl = pickle.load(open(save_path + 'eval_val.pkl', 'rb'))
	losses_pkl = pickle.load(open(save_path + 'losses_val.pkl', 'rb'))
	eval_batch = sorted(evals_pkl.keys())
	losses_batch = sorted(losses_pkl.keys())
	losses = []
	train_eval = []
	dev_eval = []
	for i in losses_batch:
		losses.append(losses_pkl[i])
	for i in eval_batch:
		train_eval.append(evals_pkl[i][0])
		dev_eval.append(evals_pkl[i][1])

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