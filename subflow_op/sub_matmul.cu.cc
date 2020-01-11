#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "sub_matmul.h"
#include "tensorflow/core/framework/register_types.h"

namespace tensorflow {
namespace functor {

typedef Eigen::GpuDevice GPUDevice;


template <typename T>
__global__ void GetValidOutputMaskKernel(const int* output_mask,
	int *output_dim, int *valid_output_mask, int *valid_output_mask_len) {

	int len = 0;

	for (int i = 0; i < output_dim[1]; i++) {
		if (output_mask[i] != 0) {
			valid_output_mask[len++] = i;
		}
	}

	*valid_output_mask_len = len;
}

template <typename T>
__global__ void SubMatmulKernel(const T* mat_a, const T* mat_b, T* output,
	int mat_a_dim[2], int mat_b_dim[2], int output_dim[2],
	int *valid_output, int *valid_output_len) {

	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int i = blockIdx.z * blockDim.z + threadIdx.z;

	if (col < *valid_output_len && row < output_dim[0] && i < mat_b_dim[0]) {
		int row_idx = row*output_dim[1];
		int a_row_idx = row*mat_a_dim[1];
		int col_idx = valid_output[col];
		int pos = row_idx + col_idx;
		atomicAdd(&output[pos],
			mat_a[a_row_idx+i] * mat_b[i*mat_b_dim[1]+col_idx]);
	}
}

template <typename T>
struct SubMatmulFunctor<GPUDevice, T> {
	void operator()(const GPUDevice& d, OpKernelContext* context,
			int mat_a_dim[2], int mat_b_dim[2], int output_dim[2],
			const Tensor& mat_a_tensor, const Tensor& mat_b_tensor,
			const Tensor& output_mask_tensor, Tensor* output_tensor) {

		auto a_mat_flat = mat_a_tensor.flat<T>().data();
		auto b_mat_flat = mat_b_tensor.flat<T>().data();
		auto output_mask_flat = output_mask_tensor.flat<int>().data();

		auto output_flat = output_tensor->flat<T>().data();
		cudaMemset(output_flat, 0, sizeof(T)*output_dim[0]*output_dim[1]);

		int *mat_a_dim_dev;
		int *mat_b_dim_dev;
		int *output_dim_dev;

		cudaMalloc(&mat_a_dim_dev, sizeof(int)*2);
		cudaMemcpy(mat_a_dim_dev, mat_a_dim, sizeof(int)*2, cudaMemcpyHostToDevice);
		cudaMalloc(&mat_b_dim_dev, sizeof(int)*2);
		cudaMemcpy(mat_b_dim_dev, mat_b_dim, sizeof(int)*2, cudaMemcpyHostToDevice);
		cudaMalloc(&output_dim_dev, sizeof(int)*2);
		cudaMemcpy(output_dim_dev, output_dim, sizeof(int)*2, cudaMemcpyHostToDevice);

		int *valid_output_mask_dev;
		int *valid_output_mask_len_dev;

		cudaMalloc(&valid_output_mask_dev, sizeof(int)*output_dim[1]);
		cudaMalloc(&valid_output_mask_len_dev, sizeof(int));

		GetValidOutputMaskKernel<T><<<1, 1>>>
			(output_mask_flat, output_dim_dev,
			 valid_output_mask_dev, valid_output_mask_len_dev);

		int valid_output_mask_len;
		cudaMemcpy(&valid_output_mask_len, valid_output_mask_len_dev,
			sizeof(int), cudaMemcpyDeviceToHost);
		if (valid_output_mask_len <= 0) {
			cudaFree(valid_output_mask_len_dev);
			cudaFree(valid_output_mask_dev);
			cudaFree(output_dim_dev);
			cudaFree(mat_b_dim_dev);
			cudaFree(mat_a_dim_dev);
			return;
		}

		//int dev = 0;
		//cudaDeviceProp deviceProp;
		//cudaGetDeviceProperties(&deviceProp, dev);
		//printf("%d\n", deviceProp.maxThreadsPerBlock);

		dim3 block_size(16,4,16);
		dim3 num_blocks(
			(valid_output_mask_len+block_size.x-1)/block_size.x,
			(mat_a_dim[0]+block_size.y-1)/block_size.y,
			(mat_b_dim[0]+block_size.z-1)/block_size.z);

		SubMatmulKernel<T><<<num_blocks, block_size>>>
			(a_mat_flat, b_mat_flat, output_flat,
			 mat_a_dim_dev, mat_b_dim_dev, output_dim_dev,
			 valid_output_mask_dev, valid_output_mask_len_dev);

		cudaFree(valid_output_mask_len_dev);
		cudaFree(valid_output_mask_dev);
		cudaFree(output_dim_dev);
		cudaFree(mat_b_dim_dev);
		cudaFree(mat_a_dim_dev);
	}
};

template struct SubMatmulFunctor<GPUDevice, float>;
template struct SubMatmulFunctor<GPUDevice, int32>;

template <typename T>
__global__ void SubMatmulBackMatBKernel(const T* mat_a, T* output, const T* grad,
	int mat_a_dim[2], int mat_b_dim[2], int output_dim[2],
	int *valid_output, int *valid_output_len) {


	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int i = blockIdx.z * blockDim.z + threadIdx.z;

	if (col < *valid_output_len && row < output_dim[0] && i < mat_a_dim[0]) {
		int row_idx = row*output_dim[1];
		int col_idx = valid_output[col];
		int a_col_idx = row;
		atomicAdd(&output[row_idx + col_idx],
			mat_a[i*mat_a_dim[1]+a_col_idx] * grad[i*output_dim[1]+col_idx]);
	}
}

template <typename T>
struct SubMatmulBackMatBFunctor<GPUDevice, T> {
	void operator()(const GPUDevice& d, OpKernelContext* context,
			int mat_a_dim[2], int mat_b_dim[2], int output_dim[2],
			const Tensor& mat_a_tensor, const Tensor& mat_b_tensor,
			const Tensor& output_mask_tensor, Tensor* output_tensor,
			const Tensor& grad_tensor) {

		auto a_mat_flat = mat_a_tensor.flat<T>().data();
		auto b_mat_flat = mat_b_tensor.flat<T>().data();
		auto output_mask_flat = output_mask_tensor.flat<int>().data();
		auto grad_flat = grad_tensor.flat<T>().data();

		auto output_flat = output_tensor->flat<T>().data();
		cudaMemset(output_flat, 0, sizeof(T)*output_dim[0]*output_dim[1]);

		int *mat_a_dim_dev;
		int *mat_b_dim_dev;
		int *output_dim_dev;
		int *valid_output_mask_dev;
		int *valid_output_mask_len_dev;
		int valid_output_mask_len;

		cudaMalloc(&mat_a_dim_dev, sizeof(int)*2);
		cudaMemcpy(mat_a_dim_dev, mat_a_dim, sizeof(int)*2, cudaMemcpyHostToDevice);
		cudaMalloc(&mat_b_dim_dev, sizeof(int)*2);
		cudaMemcpy(mat_b_dim_dev, mat_b_dim, sizeof(int)*2, cudaMemcpyHostToDevice);
		cudaMalloc(&output_dim_dev, sizeof(int)*2);
		cudaMemcpy(output_dim_dev, output_dim, sizeof(int)*2,
			cudaMemcpyHostToDevice);

		cudaMalloc(&valid_output_mask_dev, sizeof(int)*output_dim[1]);
		cudaMalloc(&valid_output_mask_len_dev, sizeof(int));

		GetValidOutputMaskKernel<T><<<1, 1>>>
			(output_mask_flat, output_dim_dev,
			 valid_output_mask_dev, valid_output_mask_len_dev);

		cudaMemcpy(&valid_output_mask_len, valid_output_mask_len_dev, sizeof(int),
			cudaMemcpyDeviceToHost);

		if (valid_output_mask_len <= 0) {
			cudaFree(valid_output_mask_len_dev);
			cudaFree(valid_output_mask_dev);
			cudaFree(output_dim_dev);
			cudaFree(mat_b_dim_dev);
			cudaFree(mat_a_dim_dev);
			return;
		}

		//int dev = 0;
		//cudaDeviceProp deviceProp;
		//cudaGetDeviceProperties(&deviceProp, dev);
		//printf("%d\n", deviceProp.maxThreadsPerBlock);

		dim3 block_size(16,4,16);
		dim3 num_blocks(
			(valid_output_mask_len+block_size.x-1)/block_size.x,
			(output_dim[0]+block_size.y-1)/block_size.y,
			(mat_a_dim[0]+block_size.z-1)/block_size.z);

		SubMatmulBackMatBKernel<T><<<num_blocks, block_size>>>
			(a_mat_flat, output_flat, grad_flat,
			 mat_a_dim_dev, mat_b_dim_dev, output_dim_dev,
			 valid_output_mask_dev, valid_output_mask_len_dev);

		cudaFree(valid_output_mask_len_dev);
		cudaFree(valid_output_mask_dev);
		cudaFree(output_dim_dev);
		cudaFree(mat_b_dim_dev);
		cudaFree(mat_a_dim_dev);
	}
};

template struct SubMatmulBackMatBFunctor<GPUDevice, float>;
template struct SubMatmulBackMatBFunctor<GPUDevice, int32>;

}  // end namespace functor
}  // end namespace tensorflow

#endif  // GOOGLE_CUDA
