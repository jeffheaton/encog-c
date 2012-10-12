#include <encog_cuda.h>

// Functions
void RandomInit(float*, int);

// Device code
__global__ void VecAddKernel(const float* A, const float* B, float* C, int N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N)
        C[i] = A[i] + B[i];
}

// Host code
extern "C" int TestVectorAdd()
{
	float* h_A;
	float* h_B;
	float* h_C;
	float* d_A;
	float* d_B;
	float* d_C;

    printf("Vector Addition\n");
    int N = 50000;
    size_t size = N * sizeof(float);

    // Allocate input vectors h_A and h_B in host memory
    h_A = (float*)EncogUtilAlloc(N,sizeof(float));
    h_B = (float*)EncogUtilAlloc(N,sizeof(float));
    h_C = (float*)EncogUtilAlloc(N,sizeof(float));
    
    // Initialize input vectors
    RandomInit(h_A, N);
    RandomInit(h_B, N);

    // Allocate vectors in device memory
    checkCudaErrors( cudaMalloc((void**)&d_A, size) );
    checkCudaErrors( cudaMalloc((void**)&d_B, size) );
    checkCudaErrors( cudaMalloc((void**)&d_C, size) );

    // Copy vectors from host memory to device memory
    checkCudaErrors( cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice) );
    checkCudaErrors( cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice) );

    // Invoke kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    VecAddKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    
	getLastCudaError("kernel launch failure");
#ifdef _DEBUG
    checkCudaErrors( cudaDeviceSynchronize() );
#endif

    // Copy result from device memory to host memory
    // h_C contains the result in host memory
    checkCudaErrors( cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost) );
    
    // Verify result
    int i;
	int error = 0;
    for (i = 0; i < N; ++i) {
        float sum = h_A[i] + h_B[i];
        if (fabs(h_C[i] - sum) > 1e-5) {
			error = 1;
            break;
		}
    }

    // cleanup resources
	// Free device memory
    if (d_A)
        cudaFree(d_A);
    if (d_B)
        cudaFree(d_B);
    if (d_C)
        cudaFree(d_C);

    // Free host memory
    if (h_A)
        free(h_A);
    if (h_B)
        free(h_B);
    if (h_C)
        free(h_C);
#if (CUDA_VERSION > 4010 )        
    cudaDeviceReset();
#endif	

	// display error
	if( error ) {
		printf("CUDA Vector Add Test failed.\n");
	} else {
		printf("CUDA Vector Add Test was successful.\n");
	}
	return 0;   
}

// Allocates an array with random float entries.
void RandomInit(float* data, int n)
{
    for (int i = 0; i < n; ++i)
        data[i] = rand() / (float)RAND_MAX;
}

