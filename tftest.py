# import tensorflow as tf
#
# # node1 = tf.constant(3.0,dtype=tf.float32)
# # node2 = tf.constant(4.0)
# # print(node1,node2)
# #
# sess=tf.Session()
# # print(sess.run([node1,node2]))
# #
# # node3 = tf.add(node1,node2)
# # print("",sess.run(node3))
#
# # # placeholders
# # a = tf.placeholder(tf.float32)
# # b = tf.placeholder(tf.float32)
# # add_node = a+b
# # print(sess.run(add_node,{a:3,b:4.5}))
#
# # variables
# w = tf.Variable([.3],dtype=tf.float32)
# b = tf.Variable([-.3],dtype=tf.float32)
#
# x = tf.placeholder(tf.float32)
# linear_model = w*x+b;
# init = tf.global_variables_initializer()
# sess.run(init)
#
# # print(sess.run(linear_model,{x:[1,2,3,4]}))
#
# y = tf.placeholder(tf.float32)
# squared_deltas=tf.square(linear_model-y)
# loss = tf.reduce_sum(squared_deltas)
# print(sess.run(loss,{x:[1,2,3,4],y:[0,-1,-2,-3]}))
#
# optimizer = tf.train.GradientDescentOptimizer(0.01)
# train = optimizer.minimize(loss)
# init = tf.global_variables_initializer()
# sess.run(init)
# print(sess.run([w, b]))
# for i in range(1000):
#     sess.run(train, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]})
#     print(sess.run([w, b]))
# print(sess.run([w, b]))

#
# import tensorflow as tf
# # Model parameters
# W = tf.Variable([.3], dtype=tf.float32)
# b = tf.Variable([-.3], dtype=tf.float32)
# # Model input and output
# x = tf.placeholder(tf.float32)
# linear_model = W*x + b
# y = tf.placeholder(tf.float32)
# # loss
# loss = tf.reduce_sum(tf.square(linear_model - y)) # sum of the squares
# # optimizer
# optimizer = tf.train.GradientDescentOptimizer(0.01)
# train = optimizer.minimize(loss)
#
# # training data
# x_train = [1, 2, 3, 4]
# y_train = [0, -1, -2, -3]
# # training loop
# init = tf.global_variables_initializer()
# sess = tf.Session()
# sess.run(init) # reset values to wrong
# for i in range(1000):
#     sess.run(train, {x: x_train, y: y_train})
#     print(sess.run([W,b]))
# # evaluate training accuracy
# curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: x_train, y: y_train})
# print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))


from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# construct model
x=tf.placeholder(tf.float32,[None,784])
W=tf.Variable(tf.zeros([784,10]))
b=tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)

# label
y_ = tf.placeholder(tf.float32, [None, 10])#10-d vector,但我们不知道具体放入多少

#cross_entropy0 = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y),reduction_indices=[1]))

cross_entropy =tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(labels =
y_,logits = tf.matmul(x, W) + b))


train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

