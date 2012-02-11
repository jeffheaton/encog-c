#include "encog_cuda.h"


// Device code

__device__ void EncogGPUActivationLinear(REAL *d,int count)
{
}

__device__ void EncogGPUActivationSigmoid(REAL *d,int count)
{
    int i;

    for(i=0; i<count; i++)
    {
        *d = 1.0 / (1.0 + exp(-1.0 * *d));
        d++;
    }
}

__device__ void EncogGPUActivationTANH(REAL *d,int count)
{
    int i;
    for(i=0; i<count; i++)
    {
        *d = tanh(*d);
        d++;
    }
}


__device__ REAL *EncogGPUDataGetInput(ENCOG_DATA *data, unsigned int index)
{
    int i = index*(data->inputCount+data->idealCount);
    return &data->data[i];
}

__device__ REAL *EncogGPUDataGetIdeal(ENCOG_DATA *data, unsigned int index)
{
    int i = index*(data->inputCount+data->idealCount);
    return &data->data[i+data->inputCount];
}

__global__ void EncogGPULink(ENCOG_NEURAL_NETWORK *net, ENCOG_DATA *data) {

	unsigned char *ptr;
	INT i;

	if( data!=NULL ) {
		data->data = (REAL*)(((char*)data)+sizeof(ENCOG_DATA));
	}

	if( net!=NULL ) {
		ptr = ((unsigned char*)net)+sizeof(ENCOG_NEURAL_NETWORK);
		net->layerCounts = (INT*)ptr; ptr+=net->layerCount*sizeof(INT);
		net->biasActivation = (REAL*)ptr; ptr+=net->layerCount*sizeof(REAL);
		net->activationFunctions = (ACTIVATION_FUNCTION*)ptr; ptr+=net->layerCount*sizeof(ACTIVATION_FUNCTION);
		net->layerContextCount = (INT*)ptr; ptr+=net->layerCount*sizeof(INT);
		net->weightIndex = (INT*)ptr; ptr+=net->layerCount*sizeof(INT);
		net->layerIndex = (INT*)ptr; ptr+=net->layerCount*sizeof(INT);
		net->layerFeedCounts = (INT*)ptr;ptr+=net->layerCount*sizeof(INT);
		net->biasActivation = (REAL*)ptr;ptr+=net->layerCount*sizeof(REAL);
		net->contextTargetOffset = (INT*)ptr;ptr+=net->layerCount*sizeof(INT);
		net->contextTargetSize = (INT*)ptr;ptr+=net->layerCount*sizeof(INT);
		net->weights = (REAL*)ptr; ptr+=net->weightCount*sizeof(REAL);
		net->layerOutput = (REAL*)ptr; ptr+=net->neuronCount*sizeof(REAL);
		net->layerSums = (REAL*)ptr; ptr+=net->neuronCount*sizeof(REAL);	
	}	
}

__device__ void _ComputeLayer(ENCOG_NEURAL_NETWORK *net, int currentLayer)
{
    int x;
    int y;
    int inputIndex = net->layerIndex[currentLayer];
    int outputIndex = net->layerIndex[currentLayer - 1];
    int inputSize = net->layerCounts[currentLayer];
    int outputSize = net->layerFeedCounts[currentLayer - 1];

    int index = net->weightIndex[currentLayer - 1];

    int limitX = outputIndex + outputSize;
    int limitY = inputIndex + inputSize;

    // weight values
    for (x = outputIndex; x < limitX; x++)
    {
        REAL sum = 0;
        for (y = inputIndex; y < limitY; y++)
        {
            sum += net->weights[index++] * net->layerOutput[y];
        }
        net->layerSums[x] = sum;
        net->layerOutput[x] = sum;
    }

	EncogGPUActivationSigmoid(
        net->layerOutput+outputIndex, outputSize);

    //(*net->activationFunctions[currentLayer - 1])(
    //    net->layerOutput+outputIndex, outputSize);
}

__device__ void EncogGPUNetworkCompute(ENCOG_NEURAL_NETWORK *net,REAL *input, REAL *output)
{
    int i;
    int sourceIndex;
	
	sourceIndex = net->neuronCount - net->layerCounts[net->layerCount - 1];

    memcpy(net->layerOutput+sourceIndex,input,net->inputCount*sizeof(REAL));

    for (i = net->layerCount - 1; i > 0; i--)
    {
        _ComputeLayer(net,i);
    }

	if( output!=NULL ) {
		memcpy(output,net->layerOutput,net->outputCount*sizeof(REAL));
	}
}


__global__ void EncogGPUEval(ENCOG_NEURAL_NETWORK *net, ENCOG_DATA *data, float *output)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
	REAL tmp[10];

    if (i < data->recordCount)
	{
        REAL *input = EncogGPUDataGetInput(data,i);
		REAL *ideal = EncogGPUDataGetIdeal(data,i);
		EncogGPUNetworkCompute(net,input,tmp);
		REAL delta = *tmp - *ideal;
		output[i] = delta*delta;
	}
}

// Host code
extern "C" float EncogCUDAErrorSSE(ENCOG_NEURAL_NETWORK *net, ENCOG_DATA *data)
{    
	ENCOG_NEURAL_NETWORK *deviceNet; 
	ENCOG_DATA *deviceData;
	float *deviceResults;
	float *results = (float*)EncogUtilAlloc(data->recordCount,sizeof(float));

    // Allocate vectors in device memory
    checkCudaErrors( cudaMalloc((void**)&deviceNet, net->memorySize) );
    checkCudaErrors( cudaMalloc((void**)&deviceData, data->memorySize) );
	checkCudaErrors( cudaMalloc((void**)&deviceResults, data->recordCount * sizeof(float)) );

    // Copy vectors from host memory to device memory
    checkCudaErrors( cudaMemcpy(deviceNet, net, net->memorySize, cudaMemcpyHostToDevice) );   
    checkCudaErrors( cudaMemcpy(deviceData, data, data->memorySize, cudaMemcpyHostToDevice) );

	EncogGPULink<<<1,1>>>(deviceNet, deviceData);

    // Invoke kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (data->recordCount + threadsPerBlock - 1) / threadsPerBlock;
   
	EncogGPUEval<<<blocksPerGrid, threadsPerBlock>>>(deviceNet,deviceData,deviceResults);
    
	getLastCudaError("kernel launch failure");
#ifdef _DEBUG
    checkCudaErrors( cudaDeviceSynchronize() );
#endif

    // Copy result from device memory to host memory
    // h_C contains the result in host memory
    checkCudaErrors( cudaMemcpy(results, deviceResults, data->recordCount * sizeof(float), cudaMemcpyDeviceToHost) );

    cudaFree(deviceData);
	cudaFree(deviceNet);
	cudaFree(deviceResults);
#if (CUDA_VERSION > 4010 )        
    cudaDeviceReset();
#endif	

	float sum = 0;
	for(int i=0;i<data->recordCount;i++) {
	printf("%f\n",results[i]);
		sum+=results[i];
	}

	return sum/data->recordCount;   
}
