#ifndef KERNEL_SUB_CONV_H_
#define KERNEL_SUB_CONV_H_

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/platform/types.h"


namespace tensorflow {
namespace functor {

template <typename Device, typename T>
struct SubConvFunctor {
	void operator()(const Device& d, OpKernelContext* context,
			const T* input, int* input_dim, const T* filter, int* filter_dim,
			const int* stride, T* output, int* output_dim, const int* conv_len,
			const int* what_to, const int* where_to, const int* output_mask,
			const Tensor& output_mask_tensor, Tensor* output_tensor);
};

template <typename Device, typename T>
struct SubConvBackFilterFunctor {
	void operator()(const Device& d, OpKernelContext* context,
			const T* input,
			int* input_dim, int* filter_dim, int* output_dim,
			const int* output_mask, const T* grad_ys,
			T* output,
			const int* where_to, const int* conv_len);
};

template <typename Device, typename T>
struct SubConvBackInputFunctor {
	void operator()(const Device& d, OpKernelContext* context,
			const T* input, const T* filter,
			int* input_dim, int* filter_dim, int* output_dim,
			const int* output_mask, const T* grad_ys,
			T* output,
			const int* what_to, const int* where_to, const int* conv_len);
};


}  // namespace functor
}  // namespace tensorflow

#endif //KERNEL_SUB_CONV_H_
