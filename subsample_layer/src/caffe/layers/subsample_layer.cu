#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/subsample_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void SubSampleForward(const int nthreads,
    const Dtype* const bottom_data, const int num, const int channels,
    const int height, const int width, const int subsample_height,
	const int subsample_width, const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w, const int pad_h, const int pad_w,
    Dtype* const top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
	  const int sw = index % subsample_width;
	  const int sh = (index / subsample_width) % subsample_height;
	  const int c = (index / subsample_width / subsample_height) % channels;
	  const int n = index / subsample_width / subsample_height / channels;
    int hstart = sh * stride_h - pad_h;
    int wstart = sw * stride_w - pad_w;
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);
    const Dtype* const bottom_slice = bottom_data + (n * channels + c) * height * width;
	top_data[index] = bottom_slice[hstart * width + wstart];
  }
}

template <typename Dtype>
void SubsampleLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  int count = top[0]->count();
   SubSampleForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data, bottom[0]->num(), channels_,
        height_, width_, subsampled_height_, subsampled_width_, kernel_h_,
        kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_, top_data);
}

template <typename Dtype>
__global__ void SubSampleBackward(const int nthreads, const Dtype* const top_diff,
    const int num, const int channels, const int height,
	const int width, const int subsample_height, const int subsample_width,
    const int kernel_h, const int kernel_w, const int stride_h,
    const int stride_w, const int pad_h, const int pad_w,
    Dtype* const bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local index
    // find out the local offset
    const int w = index % width + pad_w;
    const int h = (index / width) % height + pad_h;
    const int c = (index / width / height) % channels;
    const int n = index / width / height / channels;
    const int shstart = (h < kernel_h) ? 0 : (h - kernel_h) / stride_h + 1;
    const int swstart = (w < kernel_w) ? 0 : (w - kernel_w) / stride_w + 1;
    const Dtype* const top_diff_slice = top_diff + (n * channels + c) * subsample_height * subsample_width;
	bottom_diff[index] = top_diff_slice[shstart * subsample_width + swstart] ;

  }
}


template <typename Dtype>
void SubsampleLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  const int count = bottom[0]->count();
  caffe_gpu_set(count, Dtype(0.), bottom_diff);

  SubSampleBackward<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> >(
          count, top_diff, top[0]->num(), channels_,
		  height_, width_, subsampled_height_, subsampled_width_, kernel_h_,
          kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_, bottom_diff);
   
  CUDA_POST_KERNEL_CHECK;
}

INSTANTIATE_LAYER_GPU_FUNCS(SubsampleLayer);

}  // namespace caffe
