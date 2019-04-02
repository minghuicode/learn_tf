# learn tensorflow
Some simple code help myself learn tensorflow.

# Requirements

- Python 3
- Tensorflow
- Numpy
- Matplotlib
- Scikit-image
- Pillow
- Pylab


### Table of contents
1. [mlp](#mlp)  
1. [mnist](#mnist)  
1. [lstm](#lstm)  
1. [awesome](#awesome)  
1. [vgg16](#vgg16)  

### mlp  
This is an example of mlp(multilayer percetron) training process use tensorflow. All files are in folder *mlp/*.  
usage:  
``` python3 mlp/backward.py ```

### mnist  
This is an example of mlp training use mnist dataset. It require download mnist first.   
usage:  
download & train  
``` python3 mnist/mnist_backward.py ```  
test on dataset  
``` python3 mnist/mnist_test.py ```  
test on jpg filel(some test jpg files are placed in folder *mnist/jpg/*)  
``` python3 mnist/mnist_app.py ```  

### lstm  
This is an example of lstm(long short term memory) training process using tensorflow.  
usage:  
``` python3 lstm/backward.py ```  

### awesome  
This is an example of convert jpg files to tfrecord file, and then use it to train a model.    
usage:
download mnist dateset && convert gzip to jpg  
```python3 awesome/mnist2awesome.py```  
convert jpg files to tfrecord file  
```python3 awesome/generateds.py```  
training the model via tfrecord file  
```python3 awesome/backward.py```

### vgg16  
This is an example of using vgg16 to classify a single image.  
usage:  
download *vgg16.npy* file and put it in folder vgg16 on the internet.  
load model and test:   
```python3 vgg16/app.py```  
when vgg16 model load over, write *vgg16/pic/cat.jpg* , waiting its answer.



