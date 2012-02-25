#ifndef __ENCOG_CUDA_H
#define __ENCOG_CUDA_H

#include "encog.h"
#include <cuda_runtime.h>

#define MAX_CUDA_LAYERS 10

typedef struct GPU_CONST_NETWORK
{
    INT layerCount;
    INT neuronCount;
    INT weightCount;
    INT inputCount;
    INT layerCounts[MAX_CUDA_LAYERS];
    INT layerContextCount[MAX_CUDA_LAYERS];
    INT layerFeedCounts[MAX_CUDA_LAYERS];
    INT layerIndex[MAX_CUDA_LAYERS];
    INT outputCount;
    INT weightIndex[MAX_CUDA_LAYERS];
	INT activationFunctionIDs[MAX_CUDA_LAYERS];
    REAL biasActivation[MAX_CUDA_LAYERS];
	INT beginTraining;
	REAL connectionLimit;
	INT contextTargetOffset[MAX_CUDA_LAYERS];
	INT contextTargetSize[MAX_CUDA_LAYERS];
	INT endTraining;
	INT hasContext;
	INT recordCount;
	INT dynamicSize;
} GPU_CONST_NETWORK;

typedef struct GPU_DYNAMIC_NETWORK
{
    REAL *weights;
} GPU_DYNAMIC_NETWORK;


// This will output the proper CUDA error strings in the event that a CUDA host call returns an error
#define checkCudaErrors(err)  __checkCudaErrors (err, __FILE__, __LINE__)

// This will output the proper error string when calling cudaGetLastError
#define getLastCudaError(msg)      __getLastCudaError (msg, __FILE__, __LINE__)

inline void __checkCudaErrors(cudaError err, const char *file, const int line )
{
    if(cudaSuccess != err)
    {
        fprintf(stderr, "%s(%i) : CUDA Runtime API error %d: %s.\n",file, line, (int)err, cudaGetErrorString( err ) );
        exit(-1);        
    }
}

inline void __getLastCudaError(const char *errorMessage, const char *file, const int line )
{
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err)
    {
        fprintf(stderr, "%s(%i) : getLastCudaError() CUDA error : %s : (%d) %s.\n",
        file, line, errorMessage, (int)err, cudaGetErrorString( err ) );
        exit(-1);
    }
}




#endif