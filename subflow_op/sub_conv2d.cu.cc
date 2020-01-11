#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "sub_conv2d.h"
#include "tensorflow/core/framework/register_types.h"

namespace tensorflow {
namespace functor {

typedef Eigen::GpuDevice GPUDevice;

template <typename T>
__global__ void SubConvKernel(const T *input, int *input_dim, const T *filter,
	int *filter_dim, const int *stride, T *output, int *output_dim,
	const int *conv_len, const int *what, const int *where) {

	int input_size = input_dim[1]*input_dim[2]*input_dim[3];
	int filter_size = filter_dim[1]*filter_dim[2]*filter_dim[3];
	int output_size = output_dim[1]*output_dim[2]*output_dim[3];

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	//int k = blockIdx.z * blockDim.z + threadIdx.z;

	if (i < input_dim[0]*input_dim[1]*filter_dim[0] && j < input_dim[2]*input_dim[3]) {
		int batch = i / (filter_dim[0]*filter_dim[1]);
		int out_channels
			= (i - batch*filter_dim[0]*filter_dim[1]) / filter_dim[1];
		int in_channels = i % filter_dim[1];

		const T *input_pos
			= input + batch*input_size + in_channels*input_dim[2]*input_dim[3];
		const T *filter_pos
			= filter + out_channels*filter_size
			+ in_channels*filter_dim[2]*filter_dim[3];
		T *output_pos
			= output + batch*output_size
			+ out_channels*output_dim[2]*output_dim[3];

		if (input_pos[j] != 0) {
			int where_pos = 0;
			for (int k = 0; k < j; k++) {
				where_pos += conv_len[k];
			}
			for (int k = 0; k < conv_len[j]; k++) {
				atomicAdd(&output_pos[where[where_pos+k]],
					input_pos[j]*filter_pos[what[where_pos+k]]);
			}
		}
	}
}

template <typename T>
__global__ void SubConvOutputMaskKernel(T *output, int *output_dim,
	const int *output_mask) {

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	if (i < output_dim[0] && j < output_dim[1]*output_dim[2]*output_dim[3]) {
		T *output_pos = output + i*output_dim[1]*output_dim[2]*output_dim[3];
		output_pos[j] = (float)output_mask[j] * output_pos[j];
	}
}

template <typename T>
struct SubConvFunctor<GPUDevice, T> {
	void operator()(const GPUDevice& d, OpKernelContext* context,
			const T* input, int* input_dim, const T* filter, int* filter_dim,
			const int* stride, T* output, int* output_dim,
			const int* conv_len, const int* what, const int* where,
			const int* output_mask,
			const Tensor& output_mask_tensor, Tensor* output_tensor) {

		cudaMemset(output, 0,
			output_dim[0]*output_dim[1]*output_dim[2]*output_dim[3]*sizeof(T));

		int *input_dim_dev;
		cudaMalloc(&input_dim_dev, sizeof(int)*4);
		cudaMemcpy(input_dim_dev, input_dim, sizeof(int)*4,
			cudaMemcpyHostToDevice);

		int *filter_dim_dev;
		cudaMalloc(&filter_dim_dev, sizeof(int)*4);
		cudaMemcpy(filter_dim_dev, filter_dim, sizeof(int)*4,
			cudaMemcpyHostToDevice);

		int *output_dim_dev;
		cudaMalloc(&output_dim_dev, sizeof(int)*4);
		cudaMemcpy(output_dim_dev, output_dim, sizeof(int)*4,
			cudaMemcpyHostToDevice);

		dim3 block_size(8, 128);
		dim3 num_blocks((input_dim[0]*input_dim[1]*filter_dim[0]+block_size.x-1)/block_size.x,
			(input_dim[2]*input_dim[3]+block_size.y-1)/block_size.y);

		SubConvKernel<T><<<num_blocks, block_size>>>
			(input, input_dim_dev, filter, filter_dim_dev, stride,
			output, output_dim_dev, conv_len, what, where);
		cudaDeviceSynchronize();

		dim3 block_size2(4,256);
		dim3 num_blocks2((output_dim[0]+block_size2.x-1)/block_size2.x,
			(output_dim[1]*output_dim[2]*output_dim[3]+block_size2.y-1)/block_size2.y);

		SubConvOutputMaskKernel<T><<<num_blocks2, block_size2>>>
			(output, output_dim_dev, output_mask);

		cudaFree(output_dim_dev);
		cudaFree(filter_dim_dev);
		cudaFree(input_dim_dev);
	}
};

template struct SubConvFunctor<GPUDevice, float>;
template struct SubConvFunctor<GPUDevice, int32>;

__global__ void InputSequenceKernel(const int* where_to, const int* conv_len,
	const int* input_dim, const int* filter_dim,
	int* input_sequence) {

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
}

template <typename T>
__global__ void SubConvBackFilterKernel(const T* input,
	const int* input_dim, const int* filter_dim, const int* output_dim,
	int* input_sequence,
	const int* output_mask, const T* grad, T* output) {

	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < input_dim[0]*input_dim[1]*filter_dim[0]) {

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
		const int* output_mask_pos
			= output_mask + out_channels*output_dim[2]*output_dim[3];
		const T* grad_pos = grad +
			batch*output_dim[1]*output_dim[2]*output_dim[3] +
			out_channels*output_dim[2]*output_dim[3];

		for (int j = 0; j < output_dim[2]*output_dim[3]; j++) {
			if (output_mask_pos[j] != 0) {
				int hop = j / output_dim[3];
				for (int k = 0; k < filter_dim[2]*filter_dim[3]; k++) {
					int input_idx = input_sequence[k] + j
						+ hop*(input_dim[3]-output_dim[3]);
					atomicAdd(&output_pos[k],
						grad_pos[j]*input_pos[input_idx]);
				}
			}
		}
	}
}

template <typename T>
struct SubConvBackFilterFunctor<GPUDevice, T> {
	void operator()(const GPUDevice& d, OpKernelContext* context,
			const T* input,
			int* input_dim, int* filter_dim, int* output_dim,
			const int* output_mask, const T* grad,
			T* output,
			const int* where_to, const int* conv_len) {

		cudaMemset(output, 0,
			filter_dim[0]*filter_dim[1]*filter_dim[2]*filter_dim[3]*sizeof(T));

		int *input_sequence_dev;
		cudaMalloc(&input_sequence_dev, sizeof(int)*filter_dim[2]*filter_dim[3]);

		int *input_dim_dev;
		cudaMalloc(&input_dim_dev, sizeof(int)*4);
		cudaMemcpy(input_dim_dev, input_dim, sizeof(int)*4,
			cudaMemcpyHostToDevice);

		int *filter_dim_dev;
		cudaMalloc(&filter_dim_dev, sizeof(int)*4);
		cudaMemcpy(filter_dim_dev, filter_dim, sizeof(int)*4,
			cudaMemcpyHostToDevice);

		int *output_dim_dev;
		cudaMalloc(&output_dim_dev, sizeof(int)*4);
		cudaMemcpy(output_dim_dev, output_dim, sizeof(int)*4,
			cudaMemcpyHostToDevice);

		InputSequenceKernel<<<1, 1>>>(where_to, conv_len,
			input_dim_dev, filter_dim_dev, input_sequence_dev);

		int dev = 0;
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, dev);

		dim3 block_size(deviceProp.maxThreadsPerBlock);
		dim3 num_blocks((input_dim[0]*input_dim[1]*filter_dim[0]+block_size.x-1)/block_size.x);

		SubConvBackFilterKernel<T><<<num_blocks, block_size>>>
			(input, input_dim_dev, filter_dim_dev, output_dim_dev,
			input_sequence_dev,
			output_mask, grad, output);

		cudaFree(output_dim_dev);
		cudaFree(filter_dim_dev);
		cudaFree(input_dim_dev);
		cudaFree(input_sequence_dev);
	}
};

template struct SubConvBackFilterFunctor<GPUDevice, float>;
template struct SubConvBackFilterFunctor<GPUDevice, int32>;

template <typename T>
__global__ void SubConvBackInputKernel(const T* input, const T* filter,
	const int* output_mask, const T* grad, T* output,
	const int* input_dim, const int* filter_dim, const int* output_dim,
	const int* what_to, const int* where_to, const int* conv_len) {

	int batch = blockIdx.z * blockDim.z + threadIdx.z;
	int in = blockIdx.y * blockDim.y + threadIdx.y;
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (batch < input_dim[0] && in < filter_dim[1] && i < input_dim[2]*input_dim[3]) {
		int pos = 0;
		for (int j = 0; j < i; j++) {
			pos += conv_len[j];
		}

		T* output_pos = output + batch*input_dim[1]*input_dim[2]*input_dim[3]
			+ in*input_dim[2]*input_dim[3] + i;
		for (int len = 0; len < conv_len[i]; len++) {
			for (int out = 0; out < filter_dim[0]; out++) {
				const int* output_mask_pos = output_mask +
					out*output_dim[2]*output_dim[3] +
					where_to[pos+len];
				if (*output_mask_pos != 0) {
					const T* filter_pos = filter +
						out*filter_dim[1]*filter_dim[2]*filter_dim[3] +
						in*filter_dim[2]*filter_dim[3] +
						what_to[pos+len];
					const T* grad_pos = grad +
						batch*output_dim[1]*output_dim[2]*output_dim[3] +
						out*output_dim[2]*output_dim[3] +
						where_to[pos+len];
					*output_pos += *filter_pos * *grad_pos;
				}
			}
		}
	}
}

template <typename T>
struct SubConvBackInputFunctor<GPUDevice, T> {
	void operator()(const GPUDevice& d, OpKernelContext* context,
			const T* input, const T* filter,
			int* input_dim, int* filter_dim, int* output_dim,
			const int* output_mask, const T* grad,
			T* output,
			const int* what_to, const int* where_to, const int* conv_len) {

		cudaMemset(output, 0,
			input_dim[0]*input_dim[1]*input_dim[2]*input_dim[3]*sizeof(T));

		int *input_dim_dev;
		cudaMalloc(&input_dim_dev, sizeof(int)*4);
		cudaMemcpy(input_dim_dev, input_dim, sizeof(int)*4,
			cudaMemcpyHostToDevice);

		int *filter_dim_dev;
		cudaMalloc(&filter_dim_dev, sizeof(int)*4);
		cudaMemcpy(filter_dim_dev, filter_dim, sizeof(int)*4,
			cudaMemcpyHostToDevice);

		int *output_dim_dev;
		cudaMalloc(&output_dim_dev, sizeof(int)*4);
		cudaMemcpy(output_dim_dev, output_dim, sizeof(int)*4,
			cudaMemcpyHostToDevice);


		dim3 block_size(16, 16, 4);
		dim3 num_blocks(
			(input_dim[2]*input_dim[3]+block_size.x-1)/block_size.x,
			(filter_dim[1]+block_size.y-1)/block_size.y,
			(input_dim[0]+block_size.z-1)/block_size.z);

		SubConvBackInputKernel<T><<<num_blocks, block_size>>>
			(input, filter, output_mask, grad, output,
			input_dim_dev, filter_dim_dev, output_dim_dev,
			what_to, where_to, conv_len);

		cudaFree(output_dim_dev);
		cudaFree(filter_dim_dev);
		cudaFree(input_dim_dev);
	}
};

template struct SubConvBackInputFunctor<GPUDevice, float>;
template struct SubConvBackInputFunctor<GPUDevice, int32>;

}  // end namespace functor
}  // end namespace tensorflow

#endif  // GOOGLE_CUDA
