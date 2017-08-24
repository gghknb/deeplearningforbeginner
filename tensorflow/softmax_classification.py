import tensorflow as tf
x_data = [[1,2,1,1],[2,1,3,2,],[3,1,3,4],[4,1,5,5],[1,7,5,5]
					,[1,2,5,6],[1,6,6,6],[1,7,7,7]]
y_data = [[0,0,1],[0,0,1],[0,0,1],[0,1,0],[0,1,0],[0,1,0],[1,0,0],[1,0,0]]
#onehotencoding label 2 		2		2		1		1		1		0		0
x = tf.placeholder("float",[None,4])
y = tf.placeholder("float",[None,3])
nb_classes = 3

w = tf.Variable(tf.random_normal([4,nb_classes]),name='weight')
b = tf.Variable(tf.random_normal([nb_classes]),name = 'bias')

hypothesis = tf.nn.softmax(tf.matmul(x,w)+b)

cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(hypothesis),axis = 1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

with tf.Session() as sess :
	sess.run(tf.global_variables_initializer())

	for step in range(2001):
		sess.run(optimizer,feed_dict={x:x_data,y:y_data})
		if step % 200 == 0:
			print(step,sess.run(cost,feed_dict = {x:x_data,y:y_data}))

	#testing & one hot encoding
	a = sess.run(hypothesis,feed_dict = {x:[[1,11,7,9]]})
	print(a,sess.run(tf.arg_max(a,1)))  # arg_max-> return max value(1)

	allo = sess.run(hypothesis,feed_dict = {x:[[1,11,7,9],[1,3,4,3],[1,1,0,1]]})
	print(allo,sess.run(tf.arg_max(allo,1)))

