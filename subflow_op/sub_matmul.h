#ifndef KERNEL_SUB_MATMUL_H_
#define KERNEL_SUB_MATMUL_H_

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/platform/types.h"


namespace tensorflow {
namespace functor {

template <typename Device, typename T>
struct SubMatmulFunctor {
	void operator()(const Device& d, OpKernelContext* context,
		int mat_a_dim[2], int mat_b_dim[2], int output_dim[2],
		const Tensor& mat_a_tensor, const Tensor& mat_btensor,
		const Tensor& output_mask_tensor, Tensor* output_tensor);
};

template <typename Device, typename T>
struct SubMatmulBackMatBFunctor {
	void operator()(const Device& d, OpKernelContext* context,
		int mat_a_dim[2], int mat_b_dim[2], int output_dim[2],
		const Tensor& mat_a_tensor, const Tensor& mat_b_tensor,
		const Tensor& output_mask_tensor, Tensor* output_tensor,
		const Tensor& grad_tensor);
};


}  // namespace functor
}  // namespace tensorflow

#endif //KERNEL_SUB_MATMUL_H_
