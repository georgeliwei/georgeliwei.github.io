---
layout: CSDN blog转移
title:  "一段将mnist数据呈现为图片的python代码"
date:   2019-05-06 00:03:20 +0800
categories: liw update
---

# 描述  
一段代码，用于从mnist数据文件中读取原始数据，并通过opencv将数据呈现为图片

	import tensorflow as tf
	import numpy as np
	import struct
	import cv2 as cv
	
	def show_mnist_picture(name):
	    """this function show's image of the mnist data"""
	    cv.namedWindow("test", 0)
	    cv.resizeWindow("test", 140, 140)
	
	    fp = open(name, "rb")
	    databuf = fp.read()
	    offsetidx = 0
	    magicNum, imageNum, imageRow, imageCol = struct.unpack_from('>4I', databuf, offsetidx)
	
	    print("magicNum=%d, imageNum=%d, imageRow=%d, imageCol=%d" % (magicNum, imageNum, imageRow, imageCol))
	
	    offsetidx = struct.calcsize('>4I')
	    for idx in range(imageNum):
	        imagesize = imageRow * imageCol
	        readfmt = ">%dB" % imagesize
	        mnistimag = struct.unpack_from(readfmt, databuf, offsetidx)
	        offsetidx += struct.calcsize(readfmt)
	
	        mnistimag = np.array(mnistimag)
	        mnistimag = mnistimag.astype(np.uint8)
	        mnistimag = mnistimag.reshape(imageRow, imageCol)
	
	        cv.imshow("test", mnistimag)
	        cv.waitKey(100)
	        if (idx + 1) % 1000 == 0:
	            break
	    fp.close()
	
	
	show_mnist_picture("E:\\mlDataSet\\mnist\\train-images.idx3-ubyte")
  




