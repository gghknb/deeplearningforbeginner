import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
# two convolution layer & pooling layer
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
x = tf.placeholder(tf.float32,[None,784])
x_img = tf.reshape(x,[-1,28,28,1]) # n 28x28 image 1 color 
y = tf.placeholder(tf.float32,[None,10])

keep_prob = tf.placeholder(tf.float32)
#L1 ImgIn shape = (n,28,28,1)
w1 = tf.Variable(tf.random_normal([3,3,1,32], stddev = 0.01)) # 3x3 32 number of filters
# conv -> (n,28,28,32)
# pool -> (n,14,14,32)
L1 = tf.nn.conv2d(x_img,w1,strides = [1,1,1,1],padding='SAME')
L1 = tf.nn.relu(L1)
L1 = tf.nn.max_pool(L1,ksize = [1,2,2,1],strides = [1,2,2,1],padding = 'SAME')
L1 = tf.nn.dropout(L1,keep_prob = keep_prob)
#L2 ImgIn shape = (n,14,14,32)
w2 = tf.Variable(tf.random_normal([3,3,32,64], stddev = 0.01)) # 3x3 64 number of filters
# conv -> (n,14,14,64)
# pool -> (n,7,7,64)
L2 = tf.nn.conv2d(L1,w2,strides = [1,1,1,1],padding='SAME')
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2,ksize = [1,2,2,1],strides = [1,2,2,1],padding = 'SAME')
#L2 = tf.reshape(L2,[-1,7*7*64]) # (n,7,7,64)->(n,3136)
L2 = tf.nn.dropout(L2,keep_prob = keep_prob)

#L3 ImgIn shape = (n,7,7,64)
w3 = tf.Variable(tf.random_normal([3,3,64,128], stddev = 0.01)) # 3x3 64 number of filters
# conv -> (n,7,7,128)
# pool -> (n,4,4,128)
# reshape -> (n,4*4*128)
L3 = tf.nn.conv2d(L2,w3,strides = [1,1,1,1],padding='SAME')
L3 = tf.nn.relu(L3)
L3 = tf.nn.max_pool(L3,ksize = [1,2,2,1],strides = [1,2,2,1],padding = 'SAME')
L3 = tf.reshape(L3,[-1,4*4*128]) # (n,7,7,64)->(n,3136)
L3 = tf.nn.dropout(L3,keep_prob = keep_prob)

#fully connected layer
w4 = tf.get_variable("w4",shape = [4*4*128,625],initializer = tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([625]))
L4 = tf.nn.relu(tf.matmul(L3,w4) + b4)
L4 = tf.nn.dropout(L4,keep_prob = keep_prob)

w5 = tf.get_variable("w5",shape = [625,10],initializer = tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf.random_normal([10]))
hypothesis = tf.matmul(L4,w5) + b5


cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis,labels = y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

training_epochs = 15
batch_size = 100

sess = tf.Session()
sess.run(tf.global_variables_initializer())

print('Learning started. It takes some time.')
for epoch in range(training_epochs):
	avg_cost = 0
	total_batch = int(mnist.train.num_examples / batch_size)
	for i in range(total_batch) :
		batch_xs,batch_ys = mnist.train.next_batch(batch_size)
		c , _ = sess.run([cost,optimizer],feed_dict = {x:batch_xs,y:batch_ys,keep_prob:0.7})
		avg_cost += c / total_batch
		print(type(c))
	
	print 'Epoch : ', '%04d' %(epoch+1), 'cost = ', '{:.9f}'.format(avg_cost)

print('Learning finished')
correct_prediction = tf.equal(tf.argmax(hypothesis,1),tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
print('Accuracy : ',sess.run(accuracy,feed_dict = {x:mnist.test.images,y:mnist.test.labels,keep_prob:1}))










	

