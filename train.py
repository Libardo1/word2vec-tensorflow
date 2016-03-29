import tensorflow as tf
import math
import time
from util import *

def train(word2idx, args):

	""" Construct data flow graph """
	train_input = tf.placeholder(tf.int32, shape=[args.batch])
	train_label = tf.placeholder(tf.int32, shape=[args.batch, 1])

	# 1st layer (projection layer).
	proj = tf.Variable(tf.random_uniform([args.voca_size, args.hidden_size], -1.0, 1.0))
	out_proj = tf.nn.embedding_lookup(proj, train_input, name="proj")

	# Hidden layer.
	distribution = tf.random_normal([args.voca_size, args.hidden_size], 
							 		stddev=1.0 / math.sqrt(args.hidden_size))
	h_W = tf.Variable(distribution)
	h_b = tf.Variable(tf.zeros([args.voca_size]))

	# Loss function
	nce_op = tf.nn.nce_loss(h_W, h_b, out_proj, train_label, 
							args.neg_sample, args.voca_size,
							name="nce")
	loss_op = tf.reduce_mean(nce_op)

	opt = tf.train.AdamOptimizer(learning_rate=args.lr).minimize(loss_op)

	log = open("log/loss.txt", "w")
	log.write("lr = {0:.4f}\n" .format(args.lr))

	""" Start tensorflow session """
	with tf.Session() as sess:
		init_op = tf.initialize_all_variables()
		sess.run(init_op)

		for epoch in range(args.num_epoch):
			batch_input, batch_label = generate_batch(word2idx, args)
			feed_dict = {train_input: batch_input, train_label: batch_label}
			
			_, loss = sess.run([opt, loss_op], feed_dict=feed_dict)

			if args.verbose and epoch % 100 == 0:
				print("Loss in {0} epochs: {1:.2f}" .format(epoch, loss))
				log.write("Loss in {0} epochs: {1:.2f}\n" .format(epoch, loss))

		result = proj.eval()
		
	log.write("Final loss: {0:.2f}\n" .format(loss))
	log.close()

	return result
