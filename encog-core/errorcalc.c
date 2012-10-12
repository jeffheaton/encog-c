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
#include "encog.h"


float EncogErrorSSE(ENCOG_NEURAL_NETWORK *net, ENCOG_DATA *data)
{
#ifndef ENCOG_CUDA    
    return EncogCPUErrorSSE(net,data);
#else
	GPU_DEVICE *device;
	float result;
	
	if( encogContext.gpuEnabled ) {
	device = EncogGPUDeviceNew(0, net, data);
	result = EncogCUDAErrorSSE(device, net);
	EncogGPUDeviceDelete(device);
	return result;
	}
	else {
		return EncogCPUErrorSSE(net,data);
	}
#endif
}

float EncogCPUErrorSSE(ENCOG_NEURAL_NETWORK *net, ENCOG_DATA *data)
{
    INT i,j;
    REAL *input,*ideal,delta,sum;

	/* Clear out any previous errors */
	EncogErrorClear();

    sum = 0;
    for(i=0; i<data->recordCount; i++)
    {
        input = EncogDataGetInput(data,i);
        ideal = EncogDataGetIdeal(data,i);

        EncogNetworkCompute(net,input,NULL);
        for(j=0; j<net->outputCount; j++)
        {
			delta = net->layerOutput[j] - ideal[j];
            sum+=delta*delta;
        }
    }

	return (float)(sum/(data->recordCount*data->idealCount));
}
