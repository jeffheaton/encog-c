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

__device__ void EncogGPUNetworkClearContext(ENCOG_NEURAL_NETWORK *net,REAL *layerSums, REAL *layerOutput)
{
    INT index = 0;
    INT hasBias;
    INT i;
    INT j;

    for (i = 0; i < net->layerCount; i++)
    {
        hasBias = (net->layerContextCount[i] + net->layerFeedCounts[i]) != net->layerCounts[i];

        // fill in regular neurons
        for (j = 0; j < net->layerFeedCounts[i]; j++)
        {
            layerOutput[index++] = 0;
        }

        // fill in the bias
        if (hasBias)
        {
            layerOutput[index++] = net->biasActivation[i];
        }

        // fill in context
        for (j = 0; j < net->layerContextCount[i]; j++)
        {
            layerOutput[index++] = 0;
        }
    }
}

__device__ void _ComputeLayer(ENCOG_NEURAL_NETWORK *net, int currentLayer,REAL *layerSums, REAL *layerOutput)
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
            sum += net->weights[index++] * layerOutput[y];
        }
        layerSums[x] = sum;
        layerOutput[x] = sum;
    }

	EncogGPUActivationSigmoid(
        layerOutput+outputIndex, outputSize);

    //(*net->activationFunctions[currentLayer - 1])(
    //    net->layerOutput+outputIndex, outputSize);
}

__device__ void EncogGPUNetworkCompute(ENCOG_NEURAL_NETWORK *net,REAL *input, REAL *layerSums, REAL *layerOutput)
{
    int i;
    int sourceIndex;
	
	sourceIndex = net->neuronCount - net->layerCounts[net->layerCount - 1];

    memcpy(layerOutput+sourceIndex,input,net->inputCount*sizeof(REAL));

    for (i = net->layerCount - 1; i > 0; i--)
    {
        _ComputeLayer(net,i, layerSums, layerOutput);
    }
}


__global__ void EncogGPUEval(ENCOG_NEURAL_NETWORK *net, ENCOG_DATA *data, REAL *globalSums, REAL *globalOutput, float *errors)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < data->recordCount)
	{
		REAL *sums = globalSums + (i*net->neuronCount);
		REAL *output = globalOutput + (i*net->neuronCount);
        REAL *input = EncogGPUDataGetInput(data,i);
		REAL *ideal = EncogGPUDataGetIdeal(data,i);
		EncogGPUNetworkClearContext(net, sums, output);
		EncogGPUNetworkCompute(net,input,sums, output);
		REAL delta = *output - *ideal;
		errors[i] = delta*delta;
	}
}

// Host code
extern "C" float EncogCUDAErrorSSE(ENCOG_NEURAL_NETWORK *net, ENCOG_DATA *data)
{    
	ENCOG_NEURAL_NETWORK *deviceNet; 
	ENCOG_DATA *deviceData;
	float *deviceErrors;
	REAL *deviceSums;
	REAL *deviceOutput;
	float *errors = (float*)EncogUtilAlloc(data->recordCount,sizeof(float));

    // Allocate vectors in device memory
    checkCudaErrors( cudaMalloc((void**)&deviceNet, net->memorySize) );
    checkCudaErrors( cudaMalloc((void**)&deviceData, data->memorySize) );
	checkCudaErrors( cudaMalloc((void**)&deviceErrors, data->recordCount * sizeof(float)) );

	checkCudaErrors( cudaMalloc((void**)&deviceSums, data->recordCount * net->neuronCount * sizeof(REAL)) );
	checkCudaErrors( cudaMalloc((void**)&deviceOutput, data->recordCount * net->neuronCount * sizeof(REAL)) );

    // Copy vectors from host memory to device memory
    checkCudaErrors( cudaMemcpy(deviceNet, net, net->memorySize, cudaMemcpyHostToDevice) );   
    checkCudaErrors( cudaMemcpy(deviceData, data, data->memorySize, cudaMemcpyHostToDevice) );

	EncogGPULink<<<1,1>>>(deviceNet, deviceData);

    // Invoke kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (data->recordCount + threadsPerBlock - 1) / threadsPerBlock;
   
	EncogGPUEval<<<blocksPerGrid, threadsPerBlock>>>(deviceNet,deviceData,deviceSums,deviceOutput,deviceErrors);
    
	getLastCudaError("kernel launch failure");
#ifdef _DEBUG
    checkCudaErrors( cudaDeviceSynchronize() );
#endif

    // Copy result from device memory to host memory
    // h_C contains the result in host memory
    checkCudaErrors( cudaMemcpy(errors, deviceErrors, data->recordCount * sizeof(float), cudaMemcpyDeviceToHost) );

	REAL *temp = (REAL*)EncogUtilAlloc(data->recordCount * net->neuronCount ,sizeof(REAL));
	checkCudaErrors( cudaMemcpy(temp, deviceOutput, data->recordCount * net->neuronCount * sizeof(REAL), cudaMemcpyDeviceToHost) );

    cudaFree(deviceData);
	cudaFree(deviceNet);
	cudaFree(deviceErrors);
	cudaFree(deviceSums);
	cudaFree(deviceOutput);
#if (CUDA_VERSION > 4010 )        
    cudaDeviceReset();
#endif	

	float sum = 0;
	for(int i=0;i<data->recordCount;i++) {
		
		REAL *ptr = temp + (i*net->neuronCount);
		for(int j=0;j<net->neuronCount;j++) {
			printf("%f ",(float)*ptr);
			ptr++; 
		}
		printf("\n");
		//printf("\n%f\n",errors[i]);
	
		sum+=errors[i];
	}

	return sum/data->recordCount;   
}
