#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#define EIGEN_USE_THREADS
#endif  // GOOGLE_CUDA

#include "sub_matmul.h"

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/util/work_sharder.h"


namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace functor {

template <typename T>
struct SubMatmulFunctor<CPUDevice, T> {
	void operator()(const CPUDevice& d, OpKernelContext* context,
			int mat_a_dim[2], int mat_b_dim[2], int output_dim[2],
			const Tensor& mat_a_tensor, const Tensor& mat_b_tensor,
			const Tensor& output_mask_tensor, Tensor* output_tensor) {

		auto a_mat = mat_a_tensor.matrix<T>();
		auto b_mat = mat_b_tensor.matrix<T>();
		auto output_mask_flat = output_mask_tensor.flat<int>().data();

		auto output_mat = output_tensor->matrix<T>();
		output_mat.setZero();


		Eigen::array<Eigen::IndexPair<int>, 1> product_dims
				= { Eigen::IndexPair<int>(1, 0) };

		output_mat = a_mat.contract(b_mat, product_dims);

		auto output_flat = output_tensor->flat<T>().data();
		for (int i = 0, j = 0; i < output_dim[0]*output_dim[1]; i++) {
			output_flat[i] *= output_mask_flat[j++];
			if (j >= output_dim[1]) {
				j = 0;
			}
		}
	}
};

template <typename Device, typename T>
class SubMatmulOp : public OpKernel {
	public:
		explicit SubMatmulOp(OpKernelConstruction* context) : OpKernel(context) {}

		void Compute(OpKernelContext* context) override {
			const Tensor& mat_a_tensor = context->input(0);
			const Tensor& mat_b_tensor = context->input(1);
			const Tensor& output_mask_tensor = context->input(2);

			Tensor* output_tensor = NULL;
			OP_REQUIRES_OK(context, context->allocate_output(0,
						TensorShape({mat_a_tensor.dim_size(0), mat_b_tensor.dim_size(1)}), &output_tensor));

			OP_REQUIRES(context, output_mask_tensor.NumElements() == output_tensor->dim_size(1),
					errors::InvalidArgument("output_mask shape wrong"));

			int mat_a_dim[2] = { (int)mat_a_tensor.dim_size(0), (int)mat_a_tensor.dim_size(1) };
			int mat_b_dim[2] = { (int)mat_b_tensor.dim_size(0), (int)mat_b_tensor.dim_size(1) };
			int output_dim[2] = { (int)output_tensor->dim_size(0), (int)output_tensor->dim_size(1) };


			SubMatmulFunctor<Device, T>()(context->eigen_device<Device>(),
					context,
					mat_a_dim, mat_b_dim, output_dim,
					mat_a_tensor, mat_b_tensor, output_mask_tensor,
					output_tensor);
		}
};


template <typename T>
struct SubMatmulBackMatBFunctor<CPUDevice, T> {
	void operator()(const CPUDevice& d, OpKernelContext* context,
			int mat_a_dim[2], int mat_b_dim[2], int output_dim[2],
			const Tensor& mat_a_tensor, const Tensor& mat_b_tensor,
			const Tensor& output_mask_tensor, Tensor* output_tensor,
			const Tensor& grad_tensor) {

		auto a_mat = mat_a_tensor.matrix<T>();
		auto b_mat = mat_b_tensor.matrix<T>();
		auto output_mask_flat = output_mask_tensor.flat<int>().data();
		auto grad_mat = grad_tensor.matrix<T>();

		auto output_mat = output_tensor->matrix<T>();
		output_mat.setZero();

		int *valid_output_mask = (int *)malloc(sizeof(int)*output_dim[1]);
		int valid_output_mask_len = 0;

		for (int i = 0; i < output_dim[1]; i++) {
			if (output_mask_flat[i] != 0) {
				valid_output_mask[valid_output_mask_len++] = i;
			}
		}

		Eigen::array<Eigen::IndexPair<int>, 1> product_dims
				= { Eigen::IndexPair<int>(0, 0) };

		auto shard = [&a_mat, &b_mat, &output_mat, &grad_mat,
			&product_dims, &valid_output_mask] (int64 start, int64 end) {
			for (int64 i = start; i < end; i++) {
				output_mat.chip(valid_output_mask[i], 1)
					= a_mat.contract(grad_mat.chip(valid_output_mask[i], 1),
						product_dims);
			}
		};

		auto worker_threads
			= *(context->device()->tensorflow_cpu_worker_threads());
		int64 total = valid_output_mask_len;
		//int64 cost_per_unit = cost_per_unit = worker_threads.num_threads*10000/total;
		int64 cost_per_unit = 100000;
		Shard(worker_threads.num_threads, worker_threads.workers,
			total, cost_per_unit, shard);

		free(valid_output_mask);
	}
};

template <typename Device, typename T>
class SubMatmulBackMatBOp : public OpKernel {
	public:
		explicit SubMatmulBackMatBOp(OpKernelConstruction* context) : OpKernel(context) {}

		void Compute(OpKernelContext* context) override {
			const Tensor& mat_a_tensor = context->input(0);
			const Tensor& mat_b_tensor = context->input(1);
			const Tensor& output_mask_tensor = context->input(2);
			const Tensor& grad_tensor = context->input(3);

			int mat_a_dim[2] = { (int)mat_a_tensor.dim_size(0),
					(int)mat_a_tensor.dim_size(1) };
			int mat_b_dim[2] = { (int)mat_b_tensor.dim_size(0),
					(int)mat_b_tensor.dim_size(1) };
			int output_dim[2] = { mat_b_dim[0], mat_b_dim[1] };

			Tensor* output_tensor = NULL;
			OP_REQUIRES_OK(context, context->allocate_output(0,
					TensorShape({mat_b_dim[0], mat_b_dim[1]}), &output_tensor));

			OP_REQUIRES(context, output_mask_tensor.NumElements() == output_tensor->dim_size(1),
					errors::InvalidArgument("wrong output_mask shape"));

			SubMatmulBackMatBFunctor<Device, T>()(context->eigen_device<Device>(),
					context,
					mat_a_dim, mat_b_dim, output_dim,
					mat_a_tensor, mat_b_tensor, output_mask_tensor,
					output_tensor, grad_tensor);

		}
};

#define REGISTER_CPU(NAME, OP, T)					\
	REGISTER_KERNEL_BUILDER(					\
		Name(NAME).Device(DEVICE_CPU).TypeConstraint<T>("T"),	\
		OP<CPUDevice, T>);

REGISTER_CPU("SubMatmul", SubMatmulOp, float);
REGISTER_CPU("SubMatmul", SubMatmulOp, int32);
REGISTER_CPU("SubMatmulBackMatB", SubMatmulBackMatBOp, float);
REGISTER_CPU("SubMatmulBackMatB", SubMatmulBackMatBOp, int32);

#ifdef GOOGLE_CUDA
#define REGISTER_GPU(NAME, OP, FUNCTOR, T)				\
	extern template struct FUNCTOR<GPUDevice, T>;			\
REGISTER_KERNEL_BUILDER(						\
		Name(NAME).Device(DEVICE_GPU).TypeConstraint<T>("T"),	\
		OP<GPUDevice, T>);

REGISTER_GPU("SubMatmul", SubMatmulOp, SubMatmulFunctor, float);
REGISTER_GPU("SubMatmul", SubMatmulOp, SubMatmulFunctor, int32);
REGISTER_GPU("SubMatmulBackMatB", SubMatmulBackMatBOp, SubMatmulBackMatBFunctor, float);
REGISTER_GPU("SubMatmulBackMatB", SubMatmulBackMatBOp, SubMatmulBackMatBFunctor, int32);

#endif  // GOOGLE_CUDA
}

REGISTER_OP("SubMatmul")
.Attr("T: {int32, float}")
.Input("mat_a: T")
.Input("mat_b: T")
.Input("output_mask: int32")
.Output("out: T")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
	shape_inference::ShapeHandle input;
	TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &input));
	TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &input));
	c->set_output(0, c->Matrix(c->Dim(c->input(0), 0), c->Dim(c->input(1), 1)));
	return Status::OK();
});

REGISTER_OP("SubMatmulBackMatB")
.Attr("T: {int32, float}")
.Input("mat_a: T")
.Input("mat_b: T")
.Input("output_mask: int32")
.Input("grad: T")
.Output("out: T")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
	c->set_output(0, c->input(1));
	return Status::OK();
});

}  // namespace tensorflow
