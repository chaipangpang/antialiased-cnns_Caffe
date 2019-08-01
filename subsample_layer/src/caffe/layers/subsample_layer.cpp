#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/subsample_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

using std::min;
using std::max;

template <typename Dtype>
void SubsampleLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  SubsampleParameter subsample_param = this->layer_param_.subsample_param();
  CHECK(!subsample_param.has_kernel_size() !=
	  !(subsample_param.has_kernel_h() && subsample_param.has_kernel_w()))
      << "Filter size is kernel_size OR kernel_h and kernel_w; not both";
  CHECK(subsample_param.has_kernel_size() ||
	  (subsample_param.has_kernel_h() && subsample_param.has_kernel_w()))
      << "For non-square filters both kernel_h and kernel_w are required.";
  
  CHECK((!subsample_param.has_pad() && subsample_param.has_pad_h()
	  && subsample_param.has_pad_w())
	  || (!subsample_param.has_pad_h() && !subsample_param.has_pad_w()))
      << "pad is pad OR pad_h and pad_w are required.";
  CHECK((!subsample_param.has_stride() && subsample_param.has_stride_h()
	  && subsample_param.has_stride_w())
	  || (!subsample_param.has_stride_h() && !subsample_param.has_stride_w()))
      << "Stride is stride OR stride_h and stride_w are required.";
  
  if (subsample_param.has_kernel_size()) {
	  kernel_h_ = kernel_w_ = subsample_param.kernel_size();
    } else {
	  kernel_h_ = subsample_param.kernel_h();
	  kernel_w_ = subsample_param.kernel_w();
    }
 
  CHECK_GT(kernel_h_, 0) << "Filter dimensions cannot be zero.";
  CHECK_GT(kernel_w_, 0) << "Filter dimensions cannot be zero.";
  if (!subsample_param.has_pad_h()) {
	  pad_h_ = pad_w_ = subsample_param.pad();
  } else {
	  pad_h_ = subsample_param.pad_h();
	  pad_w_ = subsample_param.pad_w();
  }
  if (!subsample_param.has_stride_h()) {
	  stride_h_ = stride_w_ = subsample_param.stride();
  } else {
	  stride_h_ = subsample_param.stride_h();
	  stride_w_ = subsample_param.stride_w();
  }
  if (pad_h_ != 0 || pad_w_ != 0) {
    CHECK_LT(pad_h_, kernel_h_);
    CHECK_LT(pad_w_, kernel_w_);
  }
}

template <typename Dtype>
void SubsampleLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(4, bottom[0]->num_axes()) << "Input must have 4 axes, "
      << "corresponding to (num, channels, height, width)";
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();

  subsampled_height_ = static_cast<int>(ceil(static_cast<float>(
      height_ + 2 * pad_h_ - kernel_h_) / stride_h_)) + 1;
  subsampled_width_ = static_cast<int>(ceil(static_cast<float>(
      width_ + 2 * pad_w_ - kernel_w_) / stride_w_)) + 1;

  if (pad_h_ || pad_w_) {

	  if ((subsampled_height_ - 1) * stride_h_ >= height_ + pad_h_) {
		  --subsampled_height_;
    }
	  if ((subsampled_width_ - 1) * stride_w_ >= width_ + pad_w_) {
		  --subsampled_width_;
    }
	  CHECK_LT((subsampled_height_ - 1) * stride_h_, height_ + pad_h_);
	  CHECK_LT((subsampled_width_ - 1) * stride_w_, width_ + pad_w_);
  }
  top[0]->Reshape(bottom[0]->num(), channels_, subsampled_height_,  subsampled_width_);
  if (top.size() > 1) {
    top[1]->ReshapeLike(*top[0]);
  }
}

template <typename Dtype>
void SubsampleLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int top_count = top[0]->count();

  // subsample to save time, although this results in more code.
    for (int i = 0; i < top_count; ++i) {
      top_data[i] = 0;
    }
    // The main loop
    for (int n = 0; n < bottom[0]->num(); ++n) {
      for (int c = 0; c < channels_; ++c) {
        for (int sh = 0; sh < subsampled_height_; ++sh) {
          for (int sw = 0; sw < subsampled_width_; ++sw) {
            int hstart = sh * stride_h_ - pad_h_;
            int wstart = sw * stride_w_ - pad_w_;
            hstart = max(hstart, 0);
            wstart = max(wstart, 0);
            top_data[sh * subsampled_width_ + sw]  =  bottom_data[hstart * width_ + wstart];
          }
        }
        // compute offset
        bottom_data += bottom[0]->offset(0, 1);
        top_data += top[0]->offset(0, 1);
      }
    }  
}

template <typename Dtype>
void SubsampleLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  caffe_set(bottom[0]->count(), Dtype(0), bottom_diff);

    // The main loop
    for (int n = 0; n < top[0]->num(); ++n) {
      for (int c = 0; c < channels_; ++c) {
        for (int sh = 0; sh < subsampled_height_; ++sh) {
          for (int sw = 0; sw < subsampled_width_; ++sw) {

            int hstart = sh * stride_h_ - pad_h_;
            int wstart = sw * stride_w_ - pad_w_;
            hstart = max(hstart, 0);
            wstart = max(wstart, 0); 
		   bottom_diff[hstart * width_ + wstart] = top_diff[sh * subsampled_width_ + sw] ;
         
		  }
        }
        // offset
        bottom_diff += bottom[0]->offset(0, 1);
        top_diff += top[0]->offset(0, 1);
      }
    }
}


#ifdef CPU_ONLY
STUB_GPU(SubsampleLayer);
#endif

INSTANTIATE_CLASS(SubsampleLayer);
REGISTER_LAYER_CLASS(Subsample);

}  // namespace caffe
