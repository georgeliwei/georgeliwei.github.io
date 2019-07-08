---
layout: CSDN blog转移
title:  "CNN实现mnist数据集识别"
date:   2019-07-08 00:03:20 +0800
categories: liw update
---

# 描述  
CNN模型学习，实现对mnist数据集的训练和识别。

## 创建函数，从原始mnist数据文件中读取数据，读取到的28*28数据需要归一化操作
	def load_data_from_mnist(dataname, labelname):
	    """return two matrix, one is mnist data, one is lables with one-host format"""
	    fp = open(dataname, "rb")
	    databuf = fp.read()
	    offsetidx = 0
	    magicNum, imageNum, imageRow, imageCol = struct.unpack_from('>4I', databuf, 0)
	    offsetidx = struct.calcsize('>4I')
	
	    datasize = imageNum * imageRow * imageCol
	    readfmt = ">%dB" % datasize
	    imgdata = struct.unpack_from(readfmt, databuf, offsetidx)
	    imgdata = np.array(imgdata)
	    imgdata = imgdata.astype(np.float32)
	    imgdata = imgdata.reshape(imageNum, imageRow * imageCol)
	    imgdata = imgdata/255
	
	    fp.close()
	
	    fp = open(labelname, "rb")
	    labelbuf = fp.read()
	    offsetidx = 0
	    magicNum, imageNum = struct.unpack_from('>2I', labelbuf, 0)
	    offsetidx = struct.calcsize('>2I')
	    readfmt = ">%dB" % imageNum
	    labeldata = struct.unpack_from(readfmt, labelbuf, offsetidx)
	    labeldata = np.array(labeldata)
	    labeldata = labeldata.astype(np.int8)
	    labeldata = np.eye(10)[labeldata]
	    labeldata = labeldata.astype(np.float32)
	
	    fp.close()
	    return imgdata, labeldata

## 定义预测函数（推断函数）
该函数负责根据特征数据推断出预测结果。  
该函数的实现也是训练模型实现的关键部分。  
该函数内部可以划分为两部分。所使用到的变量参数和数据之间关系的定义。  
注意：由于预测函数也会被损失函数调用，因此内部变量的定义需要进行唯一化，使用方式上可以如下所示：  
	
	def weight_variable(shape, name):
	    try:
	        with tf.variable_scope("cnn", reuse=True):
	            var = tf.get_variable(name=name, shape=shape)
	            return var
	    except ValueError:
	        with tf.variable_scope("cnn", reuse=False):
	            var = tf.get_variable(name=name, shape=shape)
	            return var

预测函数详细实现方式如下：  

	def inference(input_x, keep_probe):
	    #define varible
	    w_conv1 = weight_variable([5, 5, 1, 32])
	    b_conv1 = bias_variable([32])
	    w_conv2 = weight_variable([5, 5, 32, 64])
	    b_conv2 = bias_variable([64])
	    w_fc1 = weight_variable([7 * 7 * 64, 1024])
	    b_fc1 = bias_variable([1024])
	    w_fc2 = weight_variable([1024, 10])
	    b_fc2 = bias_variable([10])
	
	    #defin model
	    x_image = tf.reshape(input_x, [-1, 28, 28, 1])
	    h_cov1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)
	    h_pool1 = max_pool_2x2(h_cov1)
	    h_cov2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
	    h_pool2 = max_pool_2x2(h_cov2)
	    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
	    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)
	    h_fc1_drop = tf.nn.dropout(h_fc1, keep_probe)
	    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2)
	    return y_conv

## 定义损失函数
损失函数需要将真实结果与预测结果进行比较，从而以差异最小化的方式进行训练。  

	def loss(input_x, input_y):
	    y_predict = inference(input_x, 0.5)
	    cross_entropy = -tf.reduce_sum(input_y * tf.log(y_predict))
	    return cross_entropy


## 定义训练函数
训练函数主要使用一些优化器，将损失函数最小化，从而得到预先定义的权重等取值，以便用于新数据的预测。  

	def train(total_loss):
	    return tf.train.AdamOptimizer(1e-4).minimize(total_loss)

## 主流程 
主流程函数组合以上各过程，本次实现如下所示：  

	def main():
	    x_input, y_input = load_train_data()
	    data_len = len(x_input)
	    batch_size = 50
	    batch_num = data_len / batch_size
	
	    x = tf.placeholder('float', shape=[None, 784])
	    y = tf.placeholder('float', shape=[None, 10])
	    total_loss = loss(x, y)
	    train_op = train(total_loss)
	    predict_op = inference(x, 1.0)
	
	    sess = tf.Session()
	    sess.run(tf.global_variables_initializer())
	
	    for idx in range(2000):
	        batch_idx = int(idx % batch_num)
	        batch_x = x_input[batch_idx * batch_size: (batch_idx + 1) * batch_size]
	        batch_y = y_input[batch_idx * batch_size: (batch_idx + 1) * batch_size]
	
	        sess.run(train_op, feed_dict={x: batch_x, y: batch_y})
	        if idx % 100 == 0:
	            print("Train step is ", idx)
	
	    #在数据集合上使用预测函数，检查预测结果
	    tst_x_input, tst_y_input = load_test_data()
	    data_len = len(x_input)
	    same_cnt = 0
	    all_cnt = 0
	    for idx in range(data_len):
	        batch_x = tst_x_input[idx: idx + 1]
	        batch_y = tst_y_input[idx: idx + 1]
	        rs = sess.run(predict_op, feed_dict={x: batch_x, y: batch_y})
	
	        rs = soft_max_result_to_one_shot(rs)
	        if (rs == batch_y).all():
	            same_cnt = same_cnt + 1
	        all_cnt = all_cnt + 1
	
	    print("accuracy = ", same_cnt / all_cnt)



# 完整代码内容如下，预测准确率可以达到90%
	import tensorflow as tf
	import numpy as np
	import struct
	
	
	def load_data_from_mnist(dataname, labelname):
	    """return two matrix, one is mnist data, one is lables with one-host format"""
	    fp = open(dataname, "rb")
	    databuf = fp.read()
	    offsetidx = 0
	    magicNum, imageNum, imageRow, imageCol = struct.unpack_from('>4I', databuf, 0)
	    offsetidx = struct.calcsize('>4I')
	
	    datasize = imageNum * imageRow * imageCol
	    readfmt = ">%dB" % datasize
	    imgdata = struct.unpack_from(readfmt, databuf, offsetidx)
	    imgdata = np.array(imgdata)
	    imgdata = imgdata.astype(np.float32)
	    imgdata = imgdata.reshape(imageNum, imageRow * imageCol)
	    imgdata = imgdata/255
	    fp.close()
	    fp = open(labelname, "rb")
	    labelbuf = fp.read()
	    offsetidx = 0
	    magicNum, imageNum = struct.unpack_from('>2I', labelbuf, 0)
	    offsetidx = struct.calcsize('>2I')
	    readfmt = ">%dB" % imageNum
	    labeldata = struct.unpack_from(readfmt, labelbuf, offsetidx)
	    labeldata = np.array(labeldata)
	    labeldata = labeldata.astype(np.int8)
	    labeldata = np.eye(10)[labeldata]
	    labeldata = labeldata.astype(np.float32)
	    fp.close()
	    return imgdata, labeldata
	
	
	def load_train_data():
	    return load_data_from_mnist(
	        "E:\\mlDataSet\\mnist\\train-images.idx3-ubyte",
	        "E:\\mlDataSet\\mnist\\train-labels.idx1-ubyte"
	    )
	
	
	def load_test_data():
	    return load_data_from_mnist(
	        "E:\\mlDataSet\\mnist\\t10k-images.idx3-ubyte",
	        "E:\\mlDataSet\\mnist\\t10k-labels.idx1-ubyte"
	    )
	
	
	def soft_max_result_to_one_shot(input):
	    result = input
	    for idx in range(len(result)):
	        argmax =np.max(result[idx])
	        result[idx] = np.where(result[idx] == argmax, 1, 0)
	    return result
	
	
	def weight_variable(shape, name):
	    try:
	        with tf.variable_scope("cnn", reuse=True):
	            var = tf.get_variable(name=name, shape=shape)
	            return var
	    except ValueError:
	        with tf.variable_scope("cnn", reuse=False):
	            var = tf.get_variable(name=name, shape=shape)
	            return var
	
	
	def bias_variable(shape):
	    inital = tf.constant(0.1, shape=shape)
	    return inital
	
	
	def conv2d(x, W):
	    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
	
	
	def max_pool_2x2(x):
	    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
	
	
	def inference(input_x, keep_probe):
	    # define varible
	    w_conv1 = weight_variable([5, 5, 1, 32], "w_conv1")
	    b_conv1 = bias_variable([32])
	    w_conv2 = weight_variable([5, 5, 32, 64], "w_conv2")
	    b_conv2 = bias_variable([64])
	    w_fc1 = weight_variable([7 * 7 * 64, 1024], "w_fc1")
	    b_fc1 = bias_variable([1024])
	    w_fc2 = weight_variable([1024, 10], "w_fc2")
	    b_fc2 = bias_variable([10])
	
	    #defin model
	    x_image = tf.reshape(input_x, [-1, 28, 28, 1])
	    h_cov1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)
	    h_pool1 = max_pool_2x2(h_cov1)
	    h_cov2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
	    h_pool2 = max_pool_2x2(h_cov2)
	    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
	    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)
	    h_fc1_drop = tf.nn.dropout(h_fc1, keep_probe)
	    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2)
	    return y_conv
	
	
	def loss(input_x, input_y):
	    y_predict = inference(input_x, 0.5)
	    cross_entropy = -tf.reduce_sum(input_y * tf.log(y_predict))
	    return cross_entropy
	
	
	def train(total_loss):
	    return tf.train.AdamOptimizer(1e-4).minimize(total_loss)
	
	
	def main():
	    x_input, y_input = load_train_data()
	    data_len = len(x_input)
	    batch_size = 50
	    batch_num = data_len / batch_size
	
	    x = tf.placeholder('float', shape=[None, 784])
	    y = tf.placeholder('float', shape=[None, 10])
	
	    predict_op = inference(x, 1.0)
	    total_loss = loss(x, y)
	    train_op = train(total_loss)
	
	    sess = tf.Session()
	    sess.run(tf.global_variables_initializer())
	
	    for idx in range(2000):
	        batch_idx = int(idx % batch_num)
	        batch_x = x_input[batch_idx * batch_size: (batch_idx + 1) * batch_size]
	        batch_y = y_input[batch_idx * batch_size: (batch_idx + 1) * batch_size]
	
	        sess.run(train_op, feed_dict={x: batch_x, y: batch_y})
	        if idx % 100 == 0:
	            print("Train step is ", idx)
	
	    tst_x_input, tst_y_input = load_test_data()
	    data_len = len(x_input)
	    same_cnt = 0
	    all_cnt = 0
	    for idx in range(data_len):
	        batch_x = tst_x_input[idx: idx + 1]
	        batch_y = tst_y_input[idx: idx + 1]
	        rs = sess.run(predict_op, feed_dict={x: batch_x, y: batch_y})
	
	        rs = soft_max_result_to_one_shot(rs)
	        if (rs == batch_y).all():
	            same_cnt = same_cnt + 1
	        all_cnt = all_cnt + 1
	    print("accuracy = ", same_cnt / all_cnt)
	
	main()
