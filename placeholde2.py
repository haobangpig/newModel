import tensorflow as tf
'''
2.also the placeholder can store the arrays.
The first dimension of the placeholder is None, meaning we can have any number of rows. 
The second dimension is fixed at 3, meaning each row needs to have three columns of data.
'''

x = tf.placeholder("float", [None, 3])
y = x * 2

with tf.Session() as session:
    x_data = [[4, 3, 12],
              [4, 5, 6],]
    result = session.run(y, feed_dict={x: x_data})
    print(result)


