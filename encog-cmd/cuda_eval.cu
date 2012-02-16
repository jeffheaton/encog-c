#include "encog_cuda.h"



__device__ __constant__ GPU_CONST_NETWORK cnet;




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

__device__ void EncogGPUNetworkClearContext(GPU_DYNAMIC_NETWORK *dnet)
{
    INT index = 0;
    INT hasBias;
    INT i;
    INT j;

    for (i = 0; i < cnet.layerCount; i++)
    {
        hasBias = (cnet.layerContextCount[i] + cnet.layerFeedCounts[i]) != cnet.layerCounts[i];

        // fill in regular neurons
        for (j = 0; j < cnet.layerFeedCounts[i]; j++)
        {
            dnet->layerOutput[index++] = 0;
        }

        // fill in the bias
        if (hasBias)
        {
            dnet->layerOutput[index++] = cnet.biasActivation[i];
        }

        // fill in context
        for (j = 0; j < cnet.layerContextCount[i]; j++)
        {
            dnet->layerOutput[index++] = 0;
        }
    }
}

__device__ void _ComputeLayer(GPU_DYNAMIC_NETWORK *dnet, int currentLayer)
{
    int x;
    int y;
    int inputIndex = cnet.layerIndex[currentLayer];
    int outputIndex = cnet.layerIndex[currentLayer - 1];
    int inputSize = cnet.layerCounts[currentLayer];
    int outputSize = cnet.layerFeedCounts[currentLayer - 1];

    int index = cnet.weightIndex[currentLayer - 1];

    int limitX = outputIndex + outputSize;
    int limitY = inputIndex + inputSize;

    // weight values
    for (x = outputIndex; x < limitX; x++)
    {
        REAL sum = 0;
        for (y = inputIndex; y < limitY; y++)
        {
            sum += dnet->weights[index++] * dnet->layerOutput[y];
        }
        dnet->layerSums[x] = sum;
        dnet->layerOutput[x] = sum;
    }

	switch(cnet.activationFunctionIDs[currentLayer - 1]) {
		case AF_LINEAR:
			EncogGPUActivationLinear(dnet->layerOutput+outputIndex, outputSize);
			break;
		case AF_SIGMOID:
			EncogGPUActivationSigmoid(dnet->layerOutput+outputIndex, outputSize);
			break;
		case AF_TANH:
			EncogGPUActivationTANH(dnet->layerOutput+outputIndex, outputSize);
			break;
	}

}

__device__ void EncogGPUNetworkCompute(GPU_DYNAMIC_NETWORK *dnet,REAL *input)
{
    int i;
    int sourceIndex;
	
	sourceIndex = cnet.neuronCount - cnet.layerCounts[cnet.layerCount - 1];

    memcpy(dnet->layerOutput+sourceIndex,input,cnet.inputCount*sizeof(REAL));

    for (i = cnet.layerCount - 1; i > 0; i--)
    {
        _ComputeLayer(dnet,i);
    }
}


__global__ void EncogGPUEval(REAL *data, REAL *dynamic, REAL *weights, float *errors)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
	GPU_DYNAMIC_NETWORK dnet;

    if (i < cnet.recordCount)
	{
		dnet.layerOutput = dynamic + (cnet.dynamicSize*i);
		dnet.layerSums = dnet.layerOutput + cnet.neuronCount;
		dnet.weights = weights;
        REAL *input = EncogGPUDataGetInput(data,i);
		REAL *ideal = EncogGPUDataGetIdeal(data,i);
		//errors = cnet.dynamicSize;
		EncogGPUNetworkClearContext(&dnet);
		EncogGPUNetworkCompute(&dnet,input);
		REAL delta = *dnet.layerOutput - *ideal;
		errors[i] = delta*delta;
	}
}

// Host code
extern "C" float EncogCUDAErrorSSE(ENCOG_NEURAL_NETWORK *net, ENCOG_DATA *data)
{    
	GPU_CONST_NETWORK tempConstNet; 
	REAL *deviceData;
	REAL *deviceDynamic;
	float *deviceErrors;
	REAL *deviceWeights;
	float *errors = (float*)EncogUtilAlloc(data->recordCount,sizeof(float));

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

	cudaMemcpyToSymbol("cnet", &tempConstNet, sizeof(GPU_CONST_NETWORK));
	

	int dataSize = (data->inputCount + data->idealCount + 1) * data->recordCount;
	int totalDynamicSize = tempConstNet.dynamicSize * dataSize; 

    // Allocate vectors in device memory
    checkCudaErrors( cudaMalloc((void**)&deviceData, dataSize*sizeof(REAL)) );
    checkCudaErrors( cudaMalloc((void**)&deviceDynamic, totalDynamicSize*sizeof(REAL)) );
	checkCudaErrors( cudaMalloc((void**)&deviceErrors, data->recordCount * sizeof(float)) );
	checkCudaErrors( cudaMalloc((void**)&deviceWeights, net->weightCount*sizeof(REAL)) );

    // Copy vectors from host memory to device memory
    checkCudaErrors( cudaMemcpy(deviceWeights, net->weights, net->weightCount * sizeof(REAL), cudaMemcpyHostToDevice) );   
    checkCudaErrors( cudaMemcpy(deviceData, data->data, dataSize*sizeof(REAL), cudaMemcpyHostToDevice) );

    // Invoke kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (data->recordCount + threadsPerBlock - 1) / threadsPerBlock;
   
	EncogGPUEval<<<blocksPerGrid, threadsPerBlock>>>(deviceData, deviceDynamic, deviceWeights, deviceErrors);
    
	getLastCudaError("kernel launch failure");
#ifdef _DEBUG
    checkCudaErrors( cudaDeviceSynchronize() );
#endif

    // Copy result from device memory to host memory
    // h_C contains the result in host memory
    checkCudaErrors( cudaMemcpy(errors, deviceErrors, data->recordCount * sizeof(float), cudaMemcpyDeviceToHost) );

    cudaFree(deviceData);
	cudaFree(deviceDynamic);
	cudaFree(deviceErrors);
	cudaFree(deviceWeights);

#if (CUDA_VERSION > 4010 )        
    cudaDeviceReset();
#endif	

	float sum = 0;
	for(int i=0;i<data->recordCount;i++) {	
		sum+=errors[i];
	}

	return sum/data->recordCount;   
}
