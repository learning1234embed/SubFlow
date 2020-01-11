#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#define EIGEN_USE_THREADS
#endif  // GOOGLE_CUDA

#include "sub_conv2d.h"

#include <type_traits>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
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
void sub_conv_single_channel(const T *input, int input_dim[2],
		const T *filter, int filter_dim[2], const int *stride,
		T *output, int output_dim[2], const int *conv_len,
		const int *what_to, const int *where_to,
		const int *output_mask)
{
	int pos = 0;
	for (int i = 0; i < input_dim[0]*input_dim[1]; i++) {
		if (input[i] != 0) {
			for (int j = 0; j < conv_len[i]; j++) {
				output[where_to[pos+j]]
					+= input[i]*filter[what_to[pos+j]];
			}
		}
		pos += conv_len[i];
	}
}

template <typename T>
struct SubConvFunctor<CPUDevice, T> {
	void operator()(const CPUDevice& d, OpKernelContext* context,
			const T* input, int* input_dim, const T* filter, int* filter_dim,
			const int* stride, T* output, int* output_dim, const int* conv_len,
			const int* what_to, const int* where_to, const int* output_mask,
			const Tensor& output_mask_tensor, Tensor* output_tensor) {

		memset(output, 0,
			output_dim[0]*output_dim[1]*output_dim[2]*output_dim[3]*sizeof(T));

		auto shard = [&input, &input_dim, &filter, &filter_dim, &stride, &output,
			&output_dim, &conv_len, &what_to, &where_to,
			&output_mask](int64 start, int64 end) {

			int input_size = input_dim[1]*input_dim[2]*input_dim[3];
			int output_size = output_dim[1]*output_dim[2]*output_dim[3];
			int filter_size = filter_dim[1]*filter_dim[2]*filter_dim[3];

			for (int64 i = start; i < end; i++) {
				int batch = i / (filter_dim[0]*filter_dim[1]);
				int out_channels = (i - batch*filter_dim[0]*filter_dim[1]) / filter_dim[1];
				int in_channels = i % filter_dim[1];

				const T *input_pos = input + batch*input_size +
					in_channels*input_dim[2]*input_dim[3];
				const T *filter_pos = filter + out_channels*filter_size +
					in_channels*filter_dim[2]*filter_dim[3];
				T *output_pos = output + batch*output_size +
					out_channels*output_dim[2]*output_dim[3];
				const int *output_mask_pos = output_mask +
					out_channels*output_dim[2]*output_dim[3];

				sub_conv_single_channel<T>(input_pos, &input_dim[2],
					filter_pos, &filter_dim[2], stride+1,
					output_pos, &output_dim[2], conv_len,
					what_to, where_to, output_mask_pos);
			}
		};

		auto worker_threads
			= *(context->device()->tensorflow_cpu_worker_threads());
		int64 total = input_dim[0]*input_dim[1]*filter_dim[0];
		//int64 cost_per_unit = worker_threads.num_threads*10000/total;
		int64 cost_per_unit = 10000;
		Shard(worker_threads.num_threads, worker_threads.workers,
			total, cost_per_unit, shard);

		for (int i = 0, j = 0; i < output_dim[0]*output_dim[1]*output_dim[2]*output_dim[3]; i++) {
			output[i] *= output_mask[j++];
			if (j >= output_dim[1]*output_dim[2]*output_dim[3]) {
				j = 0;
			}
		}
	}
};

template <typename Device, typename T>
class SubConvOp : public OpKernel {
	public:
		explicit SubConvOp(OpKernelConstruction* context) : OpKernel(context) {}

		void Compute(OpKernelContext* context) override {

			const Tensor& input_tensor = context->input(0);
			auto input = input_tensor.flat<T>();

			OP_REQUIRES(context, input_tensor.dims() == 4,
				errors::InvalidArgument("input_tensor dim should be 4"));

			const Tensor& filter_tensor = context->input(1);
			auto filter = filter_tensor.flat<T>();

			OP_REQUIRES(context, filter_tensor.dims() == 4,
				errors::InvalidArgument("filter_tensor dim should be 4"));

			const Tensor& stride_tensor = context->input(2);
			auto stride = stride_tensor.flat<int>();

			OP_REQUIRES(context, stride_tensor.dims() == 1,
				errors::InvalidArgument("stride_tensor dim should be 1"));
			OP_REQUIRES(context, stride.size() == 4,
					errors::InvalidArgument("stride size should be 4"));

			const Tensor& what_to_tensor = context->input(3);
			auto what_to = what_to_tensor.flat<int>();

			const Tensor& where_to_tensor = context->input(4);
			auto where_to = where_to_tensor.flat<int>();

			const Tensor& conv_len_tensor = context->input(5);
			auto conv_len = conv_len_tensor.flat<int>();

			const Tensor& output_mask_tensor = context->input(6);
			auto output_mask = output_mask_tensor.flat<int>();

			OP_REQUIRES(context, conv_len.size()
				== input_tensor.dim_size(2)*input_tensor.dim_size(3),
				errors::InvalidArgument("conv_len size wrong"));

			int stride_data[4];
			
			using namespace std;
			if (std::is_same<Device, CPUDevice>::value) {
				stride_data[0] = stride.data()[0];
				stride_data[1] = stride.data()[1];
				stride_data[2] = stride.data()[2];
				stride_data[3] = stride.data()[3];
			} else {
				cudaMemcpy(stride_data, stride.data(),
					sizeof(int)*4, cudaMemcpyDeviceToHost);
			}

			int output_dim[4];
			output_dim[0] = (int)input_tensor.dim_size(0);
			output_dim[1] = (int)filter_tensor.dim_size(0);
			output_dim[2] = ((int)input_tensor.dim_size(2)
				- (int)filter_tensor.dim_size(2)) / stride_data[1] + 1;
			output_dim[3] = ((int)input_tensor.dim_size(3)
				- (int)filter_tensor.dim_size(3)) / stride_data[2] + 1;

			OP_REQUIRES(context, output_dim[0] > 0,
				errors::InvalidArgument("output_dim[0] is negative"));
			OP_REQUIRES(context, output_dim[1] > 0,
				errors::InvalidArgument("output_dim[1] is negative"));
			OP_REQUIRES(context, output_dim[2] > 0,
				errors::InvalidArgument("output_dim[2] is negative"));
			OP_REQUIRES(context, output_dim[3] > 0,
				errors::InvalidArgument("output_dim[3] is negative"));

			Tensor* output_tensor = nullptr;
			OP_REQUIRES_OK(context, context->allocate_output(0,
				TensorShape({output_dim[0], output_dim[1], output_dim[2],
				output_dim[3]}), &output_tensor));
			auto output = output_tensor->template flat<T>();

			int input_dim[4];
			input_dim[0] = (int)input_tensor.dim_size(0);
			input_dim[1] = (int)input_tensor.dim_size(1);
			input_dim[2] = (int)input_tensor.dim_size(2);
			input_dim[3] = (int)input_tensor.dim_size(3);

			int filter_dim[4];
			filter_dim[0] = (int)filter_tensor.dim_size(0);
			filter_dim[1] = (int)filter_tensor.dim_size(1);
			filter_dim[2] = (int)filter_tensor.dim_size(2);
			filter_dim[3] = (int)filter_tensor.dim_size(3);

			SubConvFunctor<Device, T>()(context->eigen_device<Device>(),
				context,
				input.data(), input_dim, filter.data(), filter_dim,
				stride.data(), output.data(), output_dim,
				conv_len.data(), what_to.data(), where_to.data(),
				output_mask.data(),
				output_mask_tensor, output_tensor);
		}
};

template <typename T>
struct SubConvBackFilterFunctor<CPUDevice, T> {
	void operator()(const CPUDevice& d, OpKernelContext* context,
			const T* input,
			int* input_dim, int* filter_dim, int* output_dim,
			const int* output_mask, const T* grad_ys,
			T* output,
			const int* where_to, const int* conv_len) {

		memset(output, 0,
			sizeof(T)*filter_dim[0]*filter_dim[1]*filter_dim[2]*filter_dim[3]);

		int *input_sequence
			= (int *)malloc(sizeof(int)*filter_dim[2]*filter_dim[3]);
		int len = 0;
		int pos = 0;

		for (int i = 0; i < input_dim[2]*input_dim[3]; i++) {
			if (where_to[pos] == 0) {
				input_sequence[len++] = i;
			}
			if (len >= filter_dim[2]*filter_dim[3]) {
				break;
			}
			pos += conv_len[i];
		}

		auto shard = [&input, &input_dim, &filter_dim, &output,	&output_dim,
			&output_mask, &grad_ys, &input_sequence](int64 start, int64 end) {

			for (int64 i = start; i < end; i++) {

				int batch = i / (filter_dim[0]*filter_dim[1]);
				int out_channels
					= (i - batch*filter_dim[0]*filter_dim[1]) / filter_dim[1];
				int in_channels = i % filter_dim[1];

				const T *input_pos = input +
					batch*input_dim[1]*input_dim[2]*input_dim[3] +
					in_channels*input_dim[2]*input_dim[3];
				T *output_pos = output +
					out_channels*filter_dim[1]*filter_dim[2]*filter_dim[3] +
					in_channels*filter_dim[2]*filter_dim[3];
				const int* output_mask_pos = output_mask
					+ out_channels*output_dim[2]*output_dim[3];
				const T* grad_ys_pos = grad_ys +
					batch*output_dim[1]*output_dim[2]*output_dim[3] +
					out_channels*output_dim[2]*output_dim[3];

				for (int j = 0; j < output_dim[2]*output_dim[3]; j++) {
					if (output_mask_pos[j] != 0) {
						int hop = j / output_dim[3];
						for (int k = 0; k < filter_dim[2]*filter_dim[3]; k++) {
							int input_idx = input_sequence[k]
								+ j + hop*(input_dim[3]-output_dim[3]);
							output_pos[k] +=
								grad_ys_pos[j]*input_pos[input_idx];
						}
					}
				}
			}
		};

		auto worker_threads
			= *(context->device()->tensorflow_cpu_worker_threads());
		int64 total = input_dim[0]*input_dim[1]*filter_dim[0];
		int64 cost_per_unit = worker_threads.num_threads*10000/total;
		Shard(worker_threads.num_threads, worker_threads.workers,
			total, cost_per_unit, shard);

		free(input_sequence);
	}
};

template <typename Device, typename T>
class SubConvBackFilterOp : public OpKernel {
	public:
		explicit SubConvBackFilterOp(OpKernelConstruction* context) : OpKernel(context) {}

		void Compute(OpKernelContext* context) override {

			const Tensor& input_tensor = context->input(0);
			auto input = input_tensor.flat<T>();

			OP_REQUIRES(context, input_tensor.dims() == 4,
				errors::InvalidArgument("input_tensor dim should be 4"));

			const Tensor& filter_tensor = context->input(1);
			auto filter = filter_tensor.flat<T>();

			OP_REQUIRES(context, filter_tensor.dims() == 4,
				errors::InvalidArgument("filter_tensor dim should be 4"));

			const Tensor& stride_tensor = context->input(2);
			auto stride = stride_tensor.flat<int>();

			OP_REQUIRES(context, stride_tensor.dims() == 1,
				errors::InvalidArgument("stride_tensor dim should be 1"));
			OP_REQUIRES(context, stride.size() == 4,
				errors::InvalidArgument("stride size should be 4"));

			const Tensor& what_to_tensor = context->input(3);
			auto what_to = what_to_tensor.flat<int>();

			const Tensor& where_to_tensor = context->input(4);
			auto where_to = where_to_tensor.flat<int>();

			const Tensor& conv_len_tensor = context->input(5);
			auto conv_len = conv_len_tensor.flat<int>();

			const Tensor& output_mask_tensor = context->input(6);
			auto output_mask = output_mask_tensor.flat<int>();

			const Tensor& grad_ys_tensor = context->input(7);
			auto grad_ys = grad_ys_tensor.flat<T>();

			OP_REQUIRES(context, conv_len.size() ==
				input_tensor.dim_size(2)*input_tensor.dim_size(3),
				errors::InvalidArgument("conv_len size wrong"));

			int stride_data[4];

			using namespace std;
			if (std::is_same<Device, CPUDevice>::value) {
				stride_data[0] = stride.data()[0];
				stride_data[1] = stride.data()[1];
				stride_data[2] = stride.data()[2];
				stride_data[3] = stride.data()[3];
			} else {
				cudaMemcpy(stride_data, stride.data(),
					sizeof(int)*4, cudaMemcpyDeviceToHost);
			}

			int input_dim[4];
			input_dim[0] = (int)input_tensor.dim_size(0);
			input_dim[1] = (int)input_tensor.dim_size(1);
			input_dim[2] = (int)input_tensor.dim_size(2);
			input_dim[3] = (int)input_tensor.dim_size(3);

			int filter_dim[4];
			filter_dim[0] = (int)filter_tensor.dim_size(0);
			filter_dim[1] = (int)filter_tensor.dim_size(1);
			filter_dim[2] = (int)filter_tensor.dim_size(2);
			filter_dim[3] = (int)filter_tensor.dim_size(3);

			int output_dim[4];
			output_dim[0] = (int)input_tensor.dim_size(0);
			output_dim[1] = (int)filter_tensor.dim_size(0);
			output_dim[2] = ((int)input_tensor.dim_size(2) -
				(int)filter_tensor.dim_size(2)) / stride_data[1] + 1;
			output_dim[3] = ((int)input_tensor.dim_size(3) -
				(int)filter_tensor.dim_size(3)) / stride_data[2] + 1;

			OP_REQUIRES(context, output_dim[0] > 0,
				errors::InvalidArgument("output_dim[0] is negative"));
			OP_REQUIRES(context, output_dim[1] > 0,
				errors::InvalidArgument("output_dim[1] is negative"));
			OP_REQUIRES(context, output_dim[2] > 0,
				errors::InvalidArgument("output_dim[2] is negative"));
			OP_REQUIRES(context, output_dim[3] > 0,
				errors::InvalidArgument("output_dim[3] is negative"));

			Tensor* output_tensor = nullptr;
			OP_REQUIRES_OK(context, context->allocate_output(0,
					filter_tensor.shape(), &output_tensor));
			auto output = output_tensor->template flat<T>();

			SubConvBackFilterFunctor<Device, T>()(context->eigen_device<Device>(),
					context,
					input.data(),
					input_dim, filter_dim, output_dim,
					output_mask.data(), grad_ys.data(),
					output.data(),
					where_to.data(), conv_len.data());
		}
};

template <typename T>
struct SubConvBackInputFunctor<CPUDevice, T> {
	void operator()(const CPUDevice& d, OpKernelContext* context,
			const T* input, const T* filter,
			int* input_dim, int* filter_dim, int* output_dim,
			const int* output_mask, const T* grad_ys,
			T* output,
			const int* what_to, const int* where_to, const int* conv_len) {
	}
};


template <typename Device, typename T>
class SubConvBackInputOp : public OpKernel {
	public:
		explicit SubConvBackInputOp(OpKernelConstruction* context) : OpKernel(context) {}

		void Compute(OpKernelContext* context) override {

			const Tensor& input_tensor = context->input(0);
			auto input = input_tensor.flat<T>();

			OP_REQUIRES(context, input_tensor.dims() == 4,
				errors::InvalidArgument("input_tensor dim should be 4"));

			const Tensor& filter_tensor = context->input(1);
			auto filter = filter_tensor.flat<T>();

			OP_REQUIRES(context, filter_tensor.dims() == 4,
				errors::InvalidArgument("filter_tensor dim should be 4"));

			const Tensor& stride_tensor = context->input(2);
			auto stride = stride_tensor.flat<int>();

			OP_REQUIRES(context, stride_tensor.dims() == 1,
				errors::InvalidArgument("stride_tensor dim should be 1"));
			OP_REQUIRES(context, stride.size() == 4,
				errors::InvalidArgument("stride size should be 4"));

			const Tensor& what_to_tensor = context->input(3);
			auto what_to = what_to_tensor.flat<int>();

			const Tensor& where_to_tensor = context->input(4);
			auto where_to = where_to_tensor.flat<int>();

			const Tensor& conv_len_tensor = context->input(5);
			auto conv_len = conv_len_tensor.flat<int>();

			const Tensor& output_mask_tensor = context->input(6);
			auto output_mask = output_mask_tensor.flat<int>();

			const Tensor& grad_ys_tensor = context->input(7);
			auto grad_ys = grad_ys_tensor.flat<T>();

			OP_REQUIRES(context, conv_len.size() ==
				input_tensor.dim_size(2)*input_tensor.dim_size(3),
				errors::InvalidArgument("conv_len size wrong"));

			int stride_data[4];

			using namespace std;
			if (std::is_same<Device, CPUDevice>::value) {
				stride_data[0] = stride.data()[0];
				stride_data[1] = stride.data()[1];
				stride_data[2] = stride.data()[2];
				stride_data[3] = stride.data()[3];
			} else {
				cudaMemcpy(stride_data, stride.data(),
					sizeof(int)*4, cudaMemcpyDeviceToHost);
			}

			int input_dim[4];
			input_dim[0] = (int)input_tensor.dim_size(0);
			input_dim[1] = (int)input_tensor.dim_size(1);
			input_dim[2] = (int)input_tensor.dim_size(2);
			input_dim[3] = (int)input_tensor.dim_size(3);

			int filter_dim[4];
			filter_dim[0] = (int)filter_tensor.dim_size(0);
			filter_dim[1] = (int)filter_tensor.dim_size(1);
			filter_dim[2] = (int)filter_tensor.dim_size(2);
			filter_dim[3] = (int)filter_tensor.dim_size(3);

			int output_dim[4];
			output_dim[0] = (int)input_tensor.dim_size(0);
			output_dim[1] = (int)filter_tensor.dim_size(0);
			output_dim[2] = ((int)input_tensor.dim_size(2) -
				(int)filter_tensor.dim_size(2)) / stride_data[1] + 1;
			output_dim[3] = ((int)input_tensor.dim_size(3) -
				(int)filter_tensor.dim_size(3)) / stride_data[2] + 1;

			OP_REQUIRES(context, output_dim[0] > 0,
				errors::InvalidArgument("output_dim[0] is negative"));
			OP_REQUIRES(context, output_dim[1] > 0,
				errors::InvalidArgument("output_dim[1] is negative"));
			OP_REQUIRES(context, output_dim[2] > 0,
				errors::InvalidArgument("output_dim[2] is negative"));
			OP_REQUIRES(context, output_dim[3] > 0,
				errors::InvalidArgument("output_dim[3] is negative"));

			Tensor* output_tensor = nullptr;
			OP_REQUIRES_OK(context, context->allocate_output(0,
					input_tensor.shape(), &output_tensor));
			auto output = output_tensor->template flat<T>();

			SubConvBackInputFunctor<Device, T>()(context->eigen_device<Device>(),
					context,
					input.data(), filter.data(),
					input_dim, filter_dim, output_dim,
					output_mask.data(), grad_ys.data(),
					output.data(),
					what_to.data(), where_to.data(), conv_len.data());


		}
};


#define REGISTER_CPU(NAME, OP, T)					\
	REGISTER_KERNEL_BUILDER(					\
		Name(NAME).Device(DEVICE_CPU).TypeConstraint<T>("T"),	\
		OP<CPUDevice, T>);

REGISTER_CPU("SubConv", SubConvOp, float);
REGISTER_CPU("SubConv", SubConvOp, int32);
REGISTER_CPU("SubConvBackFilter", SubConvBackFilterOp, float);
REGISTER_CPU("SubConvBackFilter", SubConvBackFilterOp, int32);
REGISTER_CPU("SubConvBackInput", SubConvBackInputOp, float);
REGISTER_CPU("SubConvBackInput", SubConvBackInputOp, int32);

#ifdef GOOGLE_CUDA
#define REGISTER_GPU(NAME, OP, FUNCTOR, T)				\
	extern template struct FUNCTOR<GPUDevice, T>;			\
REGISTER_KERNEL_BUILDER(						\
		Name(NAME).Device(DEVICE_GPU).TypeConstraint<T>("T"),	\
		OP<GPUDevice, T>);

REGISTER_GPU("SubConv", SubConvOp, SubConvFunctor, float);
REGISTER_GPU("SubConv", SubConvOp, SubConvFunctor, int32);
REGISTER_GPU("SubConvBackFilter", SubConvBackFilterOp, SubConvBackFilterFunctor, float);
REGISTER_GPU("SubConvBackFilter", SubConvBackFilterOp, SubConvBackFilterFunctor, int32);
REGISTER_GPU("SubConvBackInput", SubConvBackInputOp, SubConvBackInputFunctor, float);
REGISTER_GPU("SubConvBackInput", SubConvBackInputOp, SubConvBackInputFunctor, int32);


#endif  // GOOGLE_CUDA
}

REGISTER_OP("SubConv")
.Attr("T: {int32, float} = DT_FLOAT")
.Input("input: T")
.Input("filter: T")
.Input("stride: int32")
.Input("what_to: int32")
.Input("where_to: int32")
.Input("conv_len: int32")
.Input("output_mask: int32")
.Output("output: T")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
	shape_inference::ShapeHandle input;
	for (size_t i = 0; i < 2; i++) {
		TF_RETURN_IF_ERROR(c->WithRank(c->input(i), 4, &input));
	}

	shape_inference::DimensionHandle batch = c->Dim(c->input(0), 0);
	shape_inference::DimensionHandle channel = c->Dim(c->input(1), 0);
	shape_inference::DimensionHandle height;
	c->Subtract(c->Dim(c->input(0), 2), c->Dim(c->input(1), 2), &height);
	c->Divide(height, shape_inference::DimensionOrConstant(1), true, &height);
	c->Add(height, shape_inference::DimensionOrConstant(1), &height);
	shape_inference::DimensionHandle width;
	c->Subtract(c->Dim(c->input(0), 3), c->Dim(c->input(1), 3), &width);
	c->Divide(width, shape_inference::DimensionOrConstant(1), true, &width);
	c->Add(width, shape_inference::DimensionOrConstant(1), &width);
	c->set_output(0, c->MakeShape({batch, channel, height, width}));
	return Status::OK();
});

REGISTER_OP("SubConvBackFilter")
.Attr("T: {int32, float} = DT_FLOAT")
.Input("input: T")
.Input("filter: T")
.Input("stride: int32")
.Input("what_to: int32")
.Input("where_to: int32")
.Input("conv_len: int32")
.Input("output_mask: int32")
.Input("grad_ys: T")
.Output("output: T")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
	c->set_output(0, c->input(1));
	return Status::OK();
});

REGISTER_OP("SubConvBackInput")
.Attr("T: {int32, float} = DT_FLOAT")
.Input("input: T")
.Input("filter: T")
.Input("stride: int32")
.Input("what_to: int32")
.Input("where_to: int32")
.Input("conv_len: int32")
.Input("output_mask: int32")
.Input("grad_ys: T")
.Output("output: T")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
	c->set_output(0, c->input(0));
	return Status::OK();
});

}  // namespace tensorflow
