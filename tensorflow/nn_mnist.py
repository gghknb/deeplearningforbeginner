from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets("MNIST_data/",one_hot = True)
nb_classes = 10
keep_prob = tf.placeholder(tf.float32)

x = tf.placeholder(tf.float32,[None,784])
y = tf.placeholder(tf.float32,[None,nb_classes])

#set nice weight 
w1 = tf.get_variable("w1",shape = [784,512],initializer = tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([512]))
L1 = tf.nn.relu(tf.matmul(x,w1)+b1)
L1 = tf.nn.dropout(L1,keep_prob = keep_prob)

w2 = tf.get_variable("w2",shape = [512,512],initializer = tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([512]))
L2 = tf.nn.relu(tf.matmul(L1,w2)+b2)
L2 = tf.nn.dropout(L2,keep_prob = keep_prob)

w3 = tf.get_variable("w3",shape = [512,512],initializer = tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([512]))
L3 = tf.nn.relu(tf.matmul(L2,w3)+b3)
L3 = tf.nn.dropout(L3,keep_prob = keep_prob)

w4 = tf.get_variable("w4",shape = [512,512],initializer = tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([512]))
L4 = tf.nn.relu(tf.matmul(L3,w4)+b4)
L4 = tf.nn.dropout(L4,keep_prob = keep_prob)

w5 = tf.get_variable("w5",shape = [512,10],initializer = tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf.random_normal([10]))

hypothesis = tf.sigmoid(tf.matmul(L4,w5)+b5)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = hypothesis,
															  labels = y))
optimizer = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(cost)
#cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(hypothesis),axis = 1))
#optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.1).minimize(cost)

is_correct = tf.equal(tf.arg_max(hypothesis,1),tf.arg_max(y,1))
accuracy = tf.reduce_mean(tf.cast(is_correct,tf.float32))

training_epochs = 15
batch_size = 100

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	
	for epoch in range(training_epochs):
		avg_cost = 0
		total_batch = int(mnist.train.num_examples / batch_size)
		
		for i in range(total_batch):
			batch_xs,batch_ys = mnist.train.next_batch(batch_size)
			c , _ = sess.run([cost,optimizer],feed_dict = {x:batch_xs,y:batch_ys,keep_prob:0.7})	
			avg_cost += c / total_batch

		print('Epoch: ', "%04d" %(epoch+1), "cost = ", "{:9f}".format(avg_cost))

		# test the model using test sets
	print("Accuracy : ",accuracy.eval(session = sess,feed_dict = { x:mnist.test.images,
		y:mnist.test.labels,keep_prob:1}))

