# antialiased-cnns_Caffe
This repository contains a [Caffe](https://github.com/BVLC/caffe) implementation of the paper [Making Convolutional Networks Shift-Invariant Again](https://arxiv.org/pdf/1904.11486.pdf).   
## antialiased-cnns  
### Anti-aliasing common downsampling layers   
<img src="https://github.com/chaipangpang/antialiased-cnns_Caffe/blob/master/pics/Anti-aliasing%20common%20downsampling%20layers.jpg" width="750" height="135" alt="Anti-aliasing%20common%20downsampling%20layers"/>  

### An example on max-pooling  
<img src="https://github.com/chaipangpang/antialiased-cnns_Caffe/blob/master/pics/example%20on%20max-pooling.jpg" width="750" height="362" alt="example%20on%20max-pooling"/>  

### BlurPool
BlurPool combine blur filter and subsample, then blur filter use gaussian blur in [antialiased-cnns](https://github.com/adobe/antialiased-cnns/blob/master/models_lpf/__init__.py).    

## Caffe implementation  
### Subsample        
Provide [subsample layer](https://github.com/chaipangpang/antialiased-cnns_Caffe/tree/master/subsample_layer) to support antialiased-cnns.  
Forward:    
 <img src="https://github.com/chaipangpang/antialiased-cnns_Caffe/blob/master/pics/forward.jpg" width="447" height="220" alt="forward"/>    
Backward:   
<img src="https://github.com/chaipangpang/antialiased-cnns_Caffe/blob/master/pics/backward.jpg" width="447" height="220" alt="backward"/>  
### Gaussian blur  
Gaussian blur use [Deep Separable Convolution](https://arxiv.org/abs/1610.02357) whit gauss initialization, gaussian kernel obeys standard normal distribution N(0,1).

