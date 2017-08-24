import tensorflow as tf
import numpy as np
#text : 'hihello'
#unique chars : h,i,e,l,o
#voc index : h:0, i:1,e:2,l:3,o;4
hidden_size =5 #output from the LSTM
input_dim = 5  #one hot size
rnn_size = 5   #one sentence
sequence_length = 6 #|ihello| ==6
batch_size = 1
#Data creation
idx2char = ['h','i','e','l','o']
x_data = [[0,1,0,2,3,3,]] #hihell
x_one_hot = [[[1,0,0,0,0],
			  [0,1,0,0,0],
			  [1,0,0,0,0],
			  [0,0,1,0,0],
			  [0,0,0,1,0],
			  [0,0,0,1,0]]]
y_data = [[1,0,2,3,3,4]]
x = tf.placeholder(tf.float32,[None,sequence_length,input_dim])
y = tf.placeholder(tf.int32,[None,sequence_length])

#creating rnn cell
cell = tf.contrib.rnn.BasicLSTMCell(num_units = hidden_size,state_is_tuple=True)
initial_state = cell.zero_state(batch_size,tf.float32)
outputs, _states = tf.nn.dynamic_rnn(cell,x,initial_state=initial_state,dtype=tf.float32)

weights = tf.ones([batch_size,sequence_length])

sequence_loss = tf.contrib.seq2seq.sequence_loss(logits = outputs,targets=y,weights = weights)
loss = tf.reduce_mean(sequence_loss)
train = tf.train.AdamOptimizer(learning_rate = 0.1).minimize(loss)

#training
prediction = tf.argmax(outputs,axis =2)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for i in range(50):
		l, _ = sess.run([loss,train],feed_dict = {x:x_one_hot,y:y_data})
		result = sess.run(prediction,feed_dict = {x:x_one_hot})
		print i,"loss:",l,"prediction:",result,"true Y:",y_data

		#print char using dic
		result_str = [idx2char[c] for c in np.squeeze(result)]
		print "\tPrediction str: ",''.join(result_str)

