#using cross_entropy, one_hot, reshape
import tensorflow as tf
import numpy as np

xy = np.loadtxt('data-04-zoo.csv',delimiter = ',', dtype = np.float32)
x_data = xy[:,0:-1]
y_data = xy[:,[-1]]

nb_classes = 7
x= tf.placeholder(tf.float32,[None,16])
y= tf.placeholder(tf.int32,[None,1]) #0~6 (n x 1)
y_one_hot = tf.one_hot(y,nb_classes)	 #to change one_hot example [[0,0,0,0,0,0,1]]
										 # one hot shape = (?,1,7)
y_one_hot = tf.reshape(y_one_hot,[-1,nb_classes])  #(?,7) ->[[0000001]]


w = tf.Variable(tf.random_normal([16,nb_classes]),name='weight')
b = tf.Variable(tf.random_normal([nb_classes]) , name = 'bias')

logits = tf.matmul(x,w) + b
hypothesis = tf.nn.softmax(logits)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits,labels = y_one_hot))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

prediction = tf.argmax(hypothesis,1) #probability->0~6
correct_prediction = tf.equal(prediction,tf.argmax(y_one_hot,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

with tf.Session() as sess :
	sess.run(tf.global_variables_initializer())
	for step in range(2001) :
		sess.run(optimizer,feed_dict={x:x_data,y:y_data})
		if step % 100 == 0 :
			loss,acc = sess.run([cost,accuracy],feed_dict={x:x_data,y:y_data})
			print("Step: {:5}\tLoss : {:3f}\tAcc:{:.2f}".format(step,loss,acc))

	pred = sess.run(prediction,feed_dict = { x:x_data})
	for p,y in zip(pred,y_data.flatten()): # flatten -> [[1],[0]] -> [1,0] 
										   # zip 
		print("[{}] prediction : {} True Y : {}".format(p == int(y),p,int(y)))

