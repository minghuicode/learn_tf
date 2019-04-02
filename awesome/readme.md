awesome data training process
=====

this is an example of tensorflow training process via own jpg files.

training:
-----
1. ```python3 mnist2awesome.py```  download mnist dateset && convert gzip to jpg

1. ```python3 generateds.py``` convert jpg files to tfrecord files

1. ```python3 backward.py``` training the model

this folder have five python files :

mnist2awesome.py
-----
	this python3 file can download mnist dataset, and then decompression it to 70,000 images(60k for training, 10k for test)

generateds.py
-----

	this python3 file can convert above 70k images to .tf(tfrecords) type file

forward.py
-----

	this file define the forward of model

backward.py
-----

	this file compute gradient and update the weights using training 60k images

test.py
-----
	this file calculate test accuracy using other 10k images

