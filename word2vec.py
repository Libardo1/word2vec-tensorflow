from util import *
from train import *
from crawler import *
import time
import pickle
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):

	plt.figure(figsize=(60, 60))  #in inches
	for i, label in enumerate(labels):
		x, y = low_dim_embs[i,:]
		plt.scatter(x, y)
		plt.annotate(label,
                     xy=(x, y),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom',
					 family=["NanumGothicCoding"])

	plt.savefig(filename)

def visualization(result, word_dict):

	tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=10000)
	plot_only = len(result)
	print(plot_only)

	low_dim_embs = tsne.fit_transform(result)
	plot_with_labels(low_dim_embs, word_dict)

def main():

	if not os.path.exists("log/"):
		os.makedirs("log/")
	log = open("log/log_w2v.txt", "w")
		
	args = parse_arguments()

	if args.sub == "crawl":
		lc = LawCrawler(args.domain, args.num_process, args.verbose)
		data = lc.start()

		with open(args.output, "w") as w:
			w.write("{0}" .format(data))
	
	if args.sub == "word2vec":	
		# To store result files
		if not os.path.exists("result/"):
			os.makedirs("result/")

		# Store/load preprocessed data
		try:
			with open("result/preprocess", "rb") as pre_data:
				log.write("Use preprocessd data.\n")
				word_dict = pickle.load(pre_data)
				word2idx = pickle.load(pre_data)
		except IOError as e:
			word2idx, word_dict = preprocess(args, log)
			with open("result/preprocess", "wb") as pre_data:
				pickle.dump(word_dict, pre_data)
				pickle.dump(word2idx, pre_data)

		start_time = time.time()
		result = train(word2idx, args)
		end_time = time.time()
		log.write("Train word2vec done. {0:.2f} sec.\n" .format(end_time-start_time))

		# Store trained data (word vector).
		with open("result/"+args.result, "wb") as w2v_data:
			pickle.dump(result, w2v_data)

	if args.sub == "vis":
		
		try:
			with open("result/result", "rb") as w2v_data:
				result = pickle.load(w2v_data)

			with open("result/preprocess", "rb") as pre_data:
				word_dict = pickle.load(pre_data)

		except IOError as e:
			log.write("No result files.\n")
			log.close()
		
		start_time = time.time()
		visualization(result, word_dict)
		end_time = time.time()
		log.write("t-SNE and visualization done. {0:.2f} sec.\n" .format(end_time-start_time))

	log.close()

if __name__ == "__main__":
	main()
