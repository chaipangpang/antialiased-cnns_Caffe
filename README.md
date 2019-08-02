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

### Anti-aliasing common downsampling layers in caffe
* MaxPool(stride = 2)
```
layer {
  bottom: "bottom"
  top: "max_pool"
  name: "max_pool"
  type: "Pooling"
  pooling_param {
    pool: MAX
    kernel_size: 2
    stride: 1
    pad:1
  }
}
layer {
  bottom: "max_pool"
  top: "blurconv"
  name: "blurconv"
  type: "ConvolutionDepthwise"
  param {
    lr_mult: 0
    decay_mult: 0
  }
  convolution_param {
    num_output: 64
    bias_term: false
    pad: 0
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "gaussian"
      std: 1
    }
  }
}
layer {
  bottom: "blurconv"
  top: "subsample"
  name: "subsample"
  type: "Subsample"
  subsample_param {
    kernel_size: 2
    stride: 2
  }
}
```
* Conv(stride = 2)+ReLU
```
layer {
  bottom: "bottom"
  top: "conv"
  name: "conv"
  type: "Convolution"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 64
    pad: 1
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "gaussian"
      std: 0.01
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  bottom: "conv"
  top: "conv"
  name: "relu"
  type: "ReLU"
}
layer {
  bottom: "conv"
  top: "blurconv"
  name: "blurconv"
  type: "ConvolutionDepthwise"
  param {
    lr_mult: 0
    decay_mult: 0
  }
  convolution_param {
    num_output: 64
    bias_term: false
    pad: 1
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "gaussian"
      std: 1
    }
  }
}
layer {
  bottom: "blurconv"
  top: "subsample"
  name: "subsample"
  type: "Subsample"
  subsample_param {
    kernel_size: 2
    stride: 2
  }
} 
```
* AvePool(stride = 2)
```
layer {
  bottom: "bottom"
  top: "blurconv"
  name: "blurconv"
  type: "ConvolutionDepthwise"
  param {
    lr_mult: 0
    decay_mult: 0
  }
  convolution_param {
    num_output: 64
    bias_term: false
    pad: 1
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "gaussian"
      std: 1
    }
  }
}
layer {
  bottom: "blurconv"
  top: "subsample"
  name: "subsample"
  type: "Subsample"
  subsample_param {
    kernel_size: 2
    stride: 2
  }
}
```  
## Usage  
### Prerequisites
[Caffe](https://github.com/BVLC/caffe)  
[CUDA8.0](https://developer.nvidia.com/cuda-toolkit)  
[cudnn5.0](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/)    

### How to build  


