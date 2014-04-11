#include "encog_cuda.h"
#include "encog.h"
__device__ __constant__ GPU_CONST_NETWORK cnet;

#define THREADS_PER_BLOCK 256

// Device code

__device__ REAL EncogGPUActivationLinear(REAL d)
{
	return d;
}

__device__ REAL EncogGPUActivationSigmoid(REAL d)
{
    return 1.0 / (1.0 + exp(-1.0 * d));
}

__device__ REAL EncogGPUActivationTANH(REAL d)
{
    return tanh(d);
       
}


__device__ REAL *EncogGPUDataGetInput(REAL *data, unsigned int index)
{
    int i = index*(cnet.inputCount+cnet.outputCount);
    return &data[i];
}

__device__ REAL *EncogGPUDataGetIdeal(REAL *data, unsigned int index)
{
    int i = index*(cnet.inputCount+cnet.outputCount);
    return &data[i+cnet.inputCount];
}

__device__ void _ComputeLayer(GPU_DYNAMIC_NETWORK *dnet, int currentLayer, REAL *input, REAL *output)
{
    int y;
    int inputSize = cnet.layerFeedCounts[currentLayer];
    int outputSize = cnet.layerFeedCounts[currentLayer - 1];
	REAL *iptr;

    int index = cnet.weightIndex[currentLayer - 1];
	int hasBias = (cnet.layerContextCount[currentLayer] + cnet.layerFeedCounts[currentLayer]) != cnet.layerCounts[currentLayer];

    // weight values
    while(outputSize--)
    {
        REAL sum = 0;
		iptr = input;
        for (y = 0; y < inputSize; y++)
        {
            sum += dnet->weights[index++] * *(iptr++);
        }

		if( hasBias ) {
			sum += dnet->weights[index++];
		}

		switch(cnet.activationFunctionIDs[currentLayer - 1]) 
		{
			case AF_LINEAR:
				*(output++) = EncogGPUActivationLinear(sum);
				break;
			case AF_SIGMOID:
				*(output++) = EncogGPUActivationSigmoid(sum);
				break;
			case AF_TANH:
				*(output++) = EncogGPUActivationTANH(sum);
				break;
		}
        //dnet->layerSums[x] = sum;
    }
}

__device__ float _ComputeOutputError(GPU_DYNAMIC_NETWORK *dnet, int currentLayer, REAL *input, REAL *ideal)
{
    int y;
    int inputSize = cnet.layerFeedCounts[currentLayer];
    int outputSize = cnet.layerFeedCounts[currentLayer - 1];
	REAL *iptr;
	REAL delta;
	float result;

    int index = cnet.weightIndex[currentLayer - 1];
	int hasBias = (cnet.layerContextCount[currentLayer] + cnet.layerFeedCounts[currentLayer]) != cnet.layerCounts[currentLayer];

	result = 0;

    // weight values
    while(outputSize--)
    {
        REAL sum = 0;
		iptr = input;
        for (y = 0; y < inputSize; y++)
        {
            sum += dnet->weights[index++] * *(iptr++);
        }

		if( hasBias ) {
			sum += dnet->weights[index++];
		}

		switch(cnet.activationFunctionIDs[currentLayer - 1]) 
		{
			case AF_LINEAR:
				delta = EncogGPUActivationLinear(sum);
				break;
			case AF_SIGMOID:
				delta = EncogGPUActivationSigmoid(sum);
				break;
			case AF_TANH:
				delta = EncogGPUActivationTANH(sum);
				break;
		}

		delta-=*(ideal++);
		result+=delta*delta;
    }

	return result;
}

__device__ float EncogGPUNetworkCompute(GPU_DYNAMIC_NETWORK *dnet,REAL *input, REAL *ideal)
{
    int i;
	REAL l1[1024];
	REAL l2[1024];
	REAL *inputPtr = l1;
	REAL *outputPtr = l2;
	REAL *temp;

    // compute the input layer to first hidden layer (h1)
	i = cnet.layerCount-1;
	_ComputeLayer(dnet,i,input,outputPtr);
	i--;

	// compute h2 to hx (if they even exist)
    while(i>1)
    {
		// swap the input ptr and output ptr
		temp = inputPtr;
		inputPtr = outputPtr;
		outputPtr = temp;
		// compute the layer
        _ComputeLayer(dnet,i,inputPtr,outputPtr);
		i--;
    }

	// compute hx to output
	// use outputPtr even though we want inputPtr, we are just being efficient and not performing a final "swap"
	return _ComputeOutputError(dnet,i,outputPtr, ideal);

}


__global__ void EncogGPUEval(REAL *data, REAL *weights, float *errors)
{
	__shared__ float cache[THREADS_PER_BLOCK];
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
	int cacheIndex = threadIdx.x;
	float temp = 0;

	GPU_DYNAMIC_NETWORK dnet;

    while (tid < cnet.recordCount)
	{	
		dnet.weights = weights;
        REAL *input = EncogGPUDataGetInput(data,tid);
		REAL *ideal = EncogGPUDataGetIdeal(data,tid);
		//errors = cnet.dynamicSize;
		temp += EncogGPUNetworkCompute(&dnet,input,ideal);		
		tid+=blockDim.x*gridDim.x;	
	}

	cache[cacheIndex] = temp;

	__syncthreads();

	int i = blockDim.x/2;
	while(i!=0) {
		if( cacheIndex<i) {
			cache[cacheIndex] += cache[cacheIndex+i];
		}
		__syncthreads();
		i/=2;
	}

	if(cacheIndex==0) {
		errors[blockIdx.x] = cache[0];
	}
}

extern "C" GPU_DEVICE *EncogGPUDeviceNew(INT deviceNumber, ENCOG_NEURAL_NETWORK *net, ENCOG_DATA *data)
{	
	GPU_DEVICE *result;
	GPU_CONST_NETWORK tempConstNet;
	
		// construct the temp const network
	tempConstNet.layerCount = net->layerCount;
    tempConstNet.neuronCount = net->neuronCount;
    tempConstNet.weightCount = net->weightCount;
    tempConstNet.inputCount = net->inputCount;
	tempConstNet.beginTraining = net->beginTraining;
	tempConstNet.connectionLimit = net->connectionLimit;
	tempConstNet.endTraining = net->endTraining;
	tempConstNet.hasContext = net->hasContext;
	tempConstNet.recordCount = data->recordCount;
	tempConstNet.outputCount = net->outputCount;
	tempConstNet.dynamicSize = (net->neuronCount*2); 

	for(int i=0;i<MAX_CUDA_LAYERS;i++) {
		if( i<net->layerCount ) {
		tempConstNet.contextTargetOffset[i] = net->contextTargetOffset[i];
		tempConstNet.contextTargetSize[i] = net->contextTargetSize[i];
	    tempConstNet.layerCounts[i] = net->layerCounts[i];
		tempConstNet.layerContextCount[i] = net->layerContextCount[i];
		tempConstNet.layerFeedCounts[i] = net->layerFeedCounts[i];
		tempConstNet.layerIndex[i] = net->layerIndex[i];
		tempConstNet.weightIndex[i] = net->weightIndex[i];
		tempConstNet.activationFunctionIDs[i] = net->activationFunctionIDs[i];
		tempConstNet.biasActivation[i] = net->biasActivation[i];
		} else {
				tempConstNet.contextTargetOffset[i] = 0;
		tempConstNet.contextTargetSize[i] = 0;
	    tempConstNet.layerCounts[i] = 0;
		tempConstNet.layerContextCount[i] = 0;
		tempConstNet.layerFeedCounts[i] = 0;
		tempConstNet.layerIndex[i] = 0;
		tempConstNet.weightIndex[i] = 0;
		tempConstNet.activationFunctionIDs[i] = 0;
		tempConstNet.biasActivation[i] = 0;
		}
	}

	cudaMemcpyToSymbol(cnet, &tempConstNet, sizeof(GPU_CONST_NETWORK));

	result = (GPU_DEVICE*)EncogUtilAlloc(1,sizeof(GPU_DEVICE));

	result->blocksPerGrid = MIN(32,(data->recordCount + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);

	int dataSize = (data->inputCount + data->idealCount + 1) * data->recordCount;

    // Allocate vectors in device memory
    checkCudaErrors( cudaMalloc((void**)&result->deviceData, dataSize*sizeof(REAL)) );
	checkCudaErrors( cudaMalloc((void**)&result->deviceErrors, result->blocksPerGrid * sizeof(float)) );
	checkCudaErrors( cudaMalloc((void**)&result->deviceWeights, net->weightCount*sizeof(REAL)) );
	result->errors = (float*)EncogUtilAlloc(data->recordCount,sizeof(float));
	result->recordCount = data->recordCount;

    // Copy vectors from host memory to device memory
//    
    checkCudaErrors( cudaMemcpy(result->deviceData, data->data, dataSize*sizeof(REAL), cudaMemcpyHostToDevice) );

	return result;
}

extern "C" void EncogGPUDeviceDelete(GPU_DEVICE *device) {
    cudaFree(device->deviceData);
	cudaFree(device->deviceErrors);
	cudaFree(device->deviceWeights);
	EncogUtilFree(device->errors);
	EncogUtilFree(device);
	#if (CUDA_VERSION > 4010 )        
    cudaDeviceReset();
#endif	

}

// Host code
extern "C" float EncogCUDAErrorSSE(GPU_DEVICE *device, ENCOG_NEURAL_NETWORK *net)
{   
	cudaEvent_t start,stop;
	float elapsed;
	checkCudaErrors( cudaMemcpy(device->deviceWeights, net->weights, net->weightCount * sizeof(REAL), cudaMemcpyHostToDevice) );   
   
   cudaEventCreate(&start);
   cudaEventCreate(&stop);

   checkCudaErrors( cudaEventRecord(start,0) );
	EncogGPUEval<<<device->blocksPerGrid, THREADS_PER_BLOCK>>>(device->deviceData, device->deviceWeights, device->deviceErrors);
   checkCudaErrors( cudaEventRecord(stop,0) );
    
	getLastCudaError("kernel launch failure");
    checkCudaErrors( cudaEventSynchronize(stop) );

    // Copy result from device memory to host memory
    // h_C contains the result in host memory
    checkCudaErrors( cudaMemcpy(device->errors, device->deviceErrors, device->blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost) );

	checkCudaErrors( cudaEventElapsedTime( &elapsed, start, stop ) );
	
	device->perfCount++;
	device->perfKernelTime+=elapsed;
	
	float sum = 0;
	for(int i=0;i<device->blocksPerGrid;i++) {	
		sum+=device->errors[i];
	}

	checkCudaErrors( cudaEventDestroy( start ) );
	checkCudaErrors( cudaEventDestroy( stop ) );

	return sum/(device->recordCount*net->outputCount);   
}


extern "C" float EncogCUDAPSOIterate(ENCOG_TRAIN_PSO *pso) {
	return 0;
}
