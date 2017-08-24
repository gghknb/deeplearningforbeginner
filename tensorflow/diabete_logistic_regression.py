import numpy as np
import tensorflow as tf

xy = np.loadtxt('data-03-diabetes.csv', delimiter = ',', dtype = np.float32)

x_data = xy[:,0:-1]
y_data = xy[:,[-1]]

x = tf.placeholder(tf.float32, shape = [None,8])
y = tf.placeholder(tf.float32, shape = [None,1])

w = tf.Variable(tf.random_normal([8,1]))
b = tf.Variable(tf.random_normal([1]))

hypothesis = tf.sigmoid(tf.matmul(x,w) + b)
cost = -tf.reduce_mean(y*tf.log(hypothesis) + (1-y) * tf.log(1-hypothesis))
train = tf.train.GradientDescentOptimizer(learning_rate = 0.01).minimize(cost)
predicted = tf.cast(hypothesis > 0.5, dtype = tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted,y), dtype = tf.float32))

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	
	feed = {x:x_data,y:y_data}
	for step in range(10001) :
		sess.run(train,feed_dict = feed)
		if step % 200 == 0 :
			print(step,sess.run(cost,feed_dict = feed))
	h,c,a = sess.run([hypothesis,predicted,accuracy],feed_dict = feed)
	print("\nHypothesis : ", h , "\nCorrect(Y): ",c , "\nAccuracy : ",a)
