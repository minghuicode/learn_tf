# coding:utf-8
### this script can convert mnist data from 4 files to pictures & labels
from PIL import Image
import struct
import os
import tarfile
from tensorflow.examples.tutorials.mnist import input_data

def read_image(filename,savepath):
	f = open(filename,'rb')
	index = 0
	buf = f.read()
	f.close()
	magic,images,rows,columns = struct.unpack_from('>IIII',buf,index)
	index += struct.calcsize('>IIII')

	for i in range(images):
		image = Image.new('L',(columns,rows))
		for x in range(rows):
			for y in range(columns):
				image.putpixel((y,x),int(struct.unpack_from('>B',buf,index)[0]))
				index += struct.calcsize('>B')
		print('save '+ str(i) + 'image')
		image.save(os.path.join(savepath,str(i)+'.jpg'))

def read_label(filename,saveFilename):
	savepath = saveFilename.split('.')[0]
	f = open(filename,'rb')
	index = 0
	buf = f.read()
	f.close()
	magic,labels = struct.unpack_from('>II', buf,index)
	index +=  struct.calcsize('>II')
	labelArr = [0] *labels
	for x in range(labels):
		labelArr[x] = int(struct.unpack_from('>B',buf,index)[0])
		index += struct.calcsize('>B')
	save = open(saveFilename,'w')
	for x in range(labels):
		save.write(str(x)+'.jpg '+str(labelArr[x])+'\n')
	save.close()
	print('save labels success')

def mnist2awesome_data():
	read_image('data/t10k-images-idx3-ubyte','awesome_data_jpg/test_jpg_10000')
	read_label('data/t10k-labels-idx1-ubyte','awesome_data_jpg/test_jpg_10000.txt')
	read_image('data/train-images-idx3-ubyte','awesome_data_jpg/train_jpg_60000')
	read_label('data/train-labels-idx1-ubyte','awesome_data_jpg/train_jpg_60000.txt')

def untar(filename,dirs):
	t = tarfile.open(fname)
	t.extractall(path=dirs)

def main():
	# download mnist data
	mnist = input_data.read_data_sets('data',one_hot=True)
	# untar mnist data 
	untar('data/t10k-images-idx3-ubyte.gz','data')
	untar('data/t10k-labels-idx1-ubyte.gz','data')
	untar('data/train-images-idx3-ubyte.gz','data')
	untar('data/train-labels-idx1-ubyte.gz','data')
	# convert mnist date to jpg
	os.mkdir('awesome_data_jpg')
	os.mkdir('awesome_data_jpg/test_jpg_10000')
	os.mkdir('awesome_data_jpg/train_jpg_60000')
	mnist2awesome_data()

if __name__ == '__main__':
	main()
