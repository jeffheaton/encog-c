/*
 * Encog(tm) Core v1.0 - ANSI C Version
 * http://www.heatonresearch.com/encog/
 * http://code.google.com/p/encog-java/

 * Copyright 2008-2012 Heaton Research, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * For more information on Heaton Research copyrights, licenses
 * and trademarks visit:
 * http://www.heatonresearch.com/copyright
 */
#include "encog-cmd.h"
#ifdef ENCOG_CUDA
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <string.h>
#include <stdio.h>

		const char *sComputeMode[] = {
            "Default (multiple host threads can use ::cudaSetDevice() with device simultaneously)",
            "Exclusive (only one host thread in one process is able to use ::cudaSetDevice() with this device)",
            "Prohibited (no host thread can use ::cudaSetDevice() with this device)",
            "Exclusive Process (many threads in one process is able to use ::cudaSetDevice() with this device)",
            "Unknown",
            NULL
        };

void BridgeVectorAdd();

int ConvertSMVer2Cores(int major, int minor)
{
        // Defines for GPU Architecture types (using the SM version to determine the # of cores per SM
        typedef struct {
                int SM; // 0xMm (hexidecimal notation), M = SM Major version, and m = SM minor version
                int Cores;
        } sSMtoCores;

        sSMtoCores nGpuArchCoresPerSM[] =
        { { 0x10,  8 }, // Tesla Generation (SM 1.0) G80 class
          { 0x11,  8 }, // Tesla Generation (SM 1.1) G8x class
          { 0x12,  8 }, // Tesla Generation (SM 1.2) G9x class
          { 0x13,  8 }, // Tesla Generation (SM 1.3) GT200 class
          { 0x20, 32 }, // Fermi Generation (SM 2.0) GF100 class
          { 0x21, 48 }, // Fermi Generation (SM 2.1) GF10x class
          { 0x30, 192}, // Fermi Generation (SM 3.0) GK10x class
          {   -1, -1 }
        };

        int index = 0;
        while (nGpuArchCoresPerSM[index].SM != -1) {
                if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor) ) {
                        return nGpuArchCoresPerSM[index].Cores;
                }
                index++;
        }
        printf("MapSMtoCores SM %d.%d is undefined (please update to the latest SDK)!\n", major, minor);
        return -1;
}



void TestCUDA()
{
	struct cudaDeviceProp deviceProp;
	int count;
	int dev, driverVersion = 0, runtimeVersion = 0;     
	char msg[256];
	int error;


	memset(&deviceProp,0,sizeof(struct cudaDeviceProp));
	deviceProp.major = 1;
	deviceProp.minor = 3;

	error = (int)cudaGetDeviceCount(&count);
	if( error!= CUDA_SUCCESS )
	{
		printf("CUDA Error: cudaGetDeviceCount, returned %i.\n",error);
		exit(1);
	}
	else if( count==0 ) 
	{
		printf("No CUDA devices detectd.\n");
		exit(1);
	}
	
	for(dev=0;dev<count;dev++) {
		cudaGetDeviceProperties(&deviceProp,dev);
		printf("Device %i: %s\n ",dev, deviceProp.name);

        // Console log
#if CUDART_VERSION >= 2020
        cudaDriverGetVersion(&driverVersion);
        cudaRuntimeGetVersion(&runtimeVersion);
        printf("  CUDA Driver Version / Runtime Version          %d.%d / %d.%d\n", driverVersion/1000, (driverVersion%100)/10, runtimeVersion/1000, (runtimeVersion%100)/10);
   
        printf("  CUDA Capability Major/Minor version number:    %d.%d\n", deviceProp.major, deviceProp.minor);
#endif
        sprintf(msg, "  Total amount of global memory:                 %.0f MBytes (%llu bytes)\n", 
                      (float)deviceProp.totalGlobalMem/1048576.0f, (unsigned long long) deviceProp.totalGlobalMem);
        puts(msg);
#if CUDART_VERSION >= 2000
        printf("  (%2d) Multiprocessors x (%2d) CUDA Cores/MP:     %d CUDA Cores\n", 
			deviceProp.multiProcessorCount,
			ConvertSMVer2Cores(deviceProp.major, deviceProp.minor),
			ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount);
#endif   
        printf("  GPU Clock Speed:                               %.2f GHz\n", deviceProp.clockRate * 1e-6f);
  
    
        printf("  Total amount of constant memory:               %u bytes\n", (unsigned int)deviceProp.totalConstMem); 
        printf("  Total amount of shared memory per block:       %u bytes\n", (unsigned int)deviceProp.sharedMemPerBlock);
        printf("  Total number of registers available per block: %d\n", deviceProp.regsPerBlock);
        printf("  Warp size:                                     %d\n", deviceProp.warpSize);
        printf("  Maximum number of threads per block:           %d\n", deviceProp.maxThreadsPerBlock);
        printf("  Maximum sizes of each dimension of a block:    %d x %d x %d\n",
               deviceProp.maxThreadsDim[0],
               deviceProp.maxThreadsDim[1],
               deviceProp.maxThreadsDim[2]);
        printf("  Maximum sizes of each dimension of a grid:     %d x %d x %d\n",
               deviceProp.maxGridSize[0],
               deviceProp.maxGridSize[1],
               deviceProp.maxGridSize[2]);
        printf("  Maximum memory pitch:                          %u bytes\n", (unsigned int)deviceProp.memPitch);
		printf("  Texture alignment:                             %u bytes\n", (unsigned int)deviceProp.textureAlignment);
}
	printf("Performing CUDA test.\n");
	TestVectorAdd();
}

#else
void TestCUDA()
{
	puts("CUDA is not supported in this build of Encog.");
}
#endif
