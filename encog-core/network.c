/*
 * Encog(tm) Core v0.5 - ANSI C Version
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

/* Local functions */

static void _AddBlockToEnd(ENCOG_NEURAL_NETWORK *net,NETWORK_BLOCK *block)
{
    block->next = (struct NETWORK_BLOCK*)net->firstBlock;
    net->firstBlock = block;
}

static void _FreeChain(ENCOG_NEURAL_NETWORK *net)
{
    NETWORK_BLOCK *current;

    current = net->firstBlock;
    while(current!=NULL)
    {
        NETWORK_BLOCK *next = (NETWORK_BLOCK *)current->next;
        EncogUtilFree(current);
        current = next;
    }
    net->firstBlock = NULL;
}

static void _ComputeLayer(ENCOG_NEURAL_NETWORK *net, int currentLayer)
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

    (*net->activationFunctions[currentLayer - 1])(
        net->layerOutput+outputIndex, outputSize);
}

/* API Functions */
ENCOG_NEURAL_NETWORK *EncogNetworkNew()
{
    ENCOG_NEURAL_NETWORK *network = (ENCOG_NEURAL_NETWORK*)EncogUtilAlloc(1,sizeof(ENCOG_NEURAL_NETWORK));
    return network;
}

void EncogNetworkAddLayer(ENCOG_NEURAL_NETWORK *net, int count, ACTIVATION_FUNCTION af, unsigned char bias)
{
    NETWORK_BLOCK *block = (NETWORK_BLOCK *)EncogUtilAlloc(1,sizeof(NETWORK_BLOCK));
    memset(block,0,sizeof(NETWORK_BLOCK));
    block->feedCount = count;
    block->totalCount = count + ((bias==1)?1:0);
    block->af = af;
    block->bias = bias;
    net->layerCount++;
    if( net->firstBlock==NULL )
    {
        net->inputCount = count;
    }
    else
    {
        net->outputCount = count;
    }
    _AddBlockToEnd(net,block);


}

void EncogNetworkDelete(ENCOG_NEURAL_NETWORK *net)
{
    EncogUtilFree(net->activationFunctions);
    EncogUtilFree(net->biasActivation);
    EncogUtilFree(net->layerCounts);
    EncogUtilFree(net->layerFeedCounts);
    EncogUtilFree(net->layerSums);
    EncogUtilFree(net->weightIndex);
    EncogUtilFree(net->weights);
    _FreeChain(net);
    EncogUtilFree(net);
}

void EncogNetworkFinalizeStructure(ENCOG_NEURAL_NETWORK *net)
{
    NETWORK_BLOCK *current;
    int index, neuronCount, weightCount;

    /* Create the network */
    index = 0;
    neuronCount = 0;
    weightCount = 0;
    net->layerCounts = (INT*)EncogUtilAlloc(net->layerCount,sizeof(INT));
    net->biasActivation = (REAL*)EncogUtilAlloc(net->layerCount,sizeof(REAL));
    net->activationFunctions = (ACTIVATION_FUNCTION*)EncogUtilAlloc(net->layerCount,sizeof(ACTIVATION_FUNCTION));
    net->layerContextCount = (INT*)EncogUtilAlloc(net->layerCount,sizeof(INT));
    net->weightIndex = (INT*)EncogUtilAlloc(net->layerCount,sizeof(INT));
    net->layerIndex = (INT*)EncogUtilAlloc(net->layerCount,sizeof(INT));
    net->layerFeedCounts = (INT*)EncogUtilAlloc(net->layerCount,sizeof(INT));
    net->biasActivation = (REAL*)EncogUtilAlloc(net->layerCount,sizeof(REAL));

    current = net->firstBlock;
    while(current!=NULL)
    {
        NETWORK_BLOCK *next = (NETWORK_BLOCK *)current->next;
        net->layerCounts[index]=current->totalCount;
        net->layerFeedCounts[index]=current->feedCount;
        net->biasActivation[index]=current->bias;
        net->activationFunctions[index]=current->af;

        neuronCount += current->totalCount;

        if (next != NULL)
        {
            weightCount += current->feedCount * next->totalCount;
        }

        if (index == 0)
        {
            net->weightIndex[index] = 0;
            net->layerIndex[index] = 0;
        }
        else
        {
            net->weightIndex[index] = net->weightIndex[index - 1]
                                      + (net->layerCounts[index] * net->layerFeedCounts[index - 1]);
            net->layerIndex[index] = net->layerIndex[index - 1]
                                     + net->layerCounts[index - 1];
        }


        index++;
        current=(NETWORK_BLOCK*)current->next;
    }

    net->weights = (REAL*)EncogUtilAlloc(weightCount,sizeof(REAL));
    net->layerOutput = (REAL*)EncogUtilAlloc(neuronCount,sizeof(REAL));
    net->layerSums = (REAL*)EncogUtilAlloc(neuronCount,sizeof(REAL));
    net->neuronCount = neuronCount;
    net->weightCount = weightCount;

    _FreeChain(net);
    EncogNetworkClearContext(net);
}

void EncogNetworkCompute(ENCOG_NEURAL_NETWORK *net,REAL *input, REAL *output)
{
    int i;
    int sourceIndex = net->neuronCount - net->layerCounts[net->layerCount - 1];

    memcpy(net->layerOutput+sourceIndex,input,net->inputCount*sizeof(REAL));

    for (i = net->layerCount - 1; i > 0; i--)
    {
        _ComputeLayer(net,i);
    }

	if( output!=NULL ) {
		memcpy(output,net->layerOutput,net->outputCount*sizeof(REAL));
	}
}

void EncogNetworkRandomizeRange(ENCOG_NEURAL_NETWORK *net,REAL low, REAL high)
{
    INT i;
    REAL d;

    for(i=0; i<net->weightCount; i++)
    {
        d = (REAL)rand()/(REAL)RAND_MAX;
        d = (d*(high-low))+low;
        net->weights[i] = d;
    }
}


void EncogNetworkImportWeights(ENCOG_NEURAL_NETWORK *net, REAL *w)
{
    memcpy(net->weights,w,net->weightCount*sizeof(REAL));
}

void EncogNetworkExportWeights(ENCOG_NEURAL_NETWORK *net, REAL *w)
{
    memcpy(w,net->weights,net->weightCount*sizeof(REAL));
}

void EncogNetworkClearContext(ENCOG_NEURAL_NETWORK *net)
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
            net->layerOutput[index++] = 0;
        }

        // fill in the bias
        if (hasBias)
        {
            net->layerOutput[index++] = net->biasActivation[i];
        }

        // fill in context
        for (j = 0; j < net->layerContextCount[i]; j++)
        {
            net->layerOutput[index++] = 0;
        }
    }
}

void EncogNetworkDump(ENCOG_NEURAL_NETWORK *net)
{
    printf("* * Encog Neural Network * *\n");
    printf("Layer Count: %i\n", net->layerCount);
    printf("Weight Count: %i\n", net->weightCount);
    printf("Input Count: %i\n", net->inputCount);
    printf("Output Count: %i\n", net->outputCount);
    EncogUtilOutputRealArray("Weights:",net->weights, net->weightCount);
    EncogUtilOutputRealArray("Bias Activation:",net->biasActivation, net->layerCount);
    EncogUtilOutputIntArray("Layer Counts:",net->layerCounts, net->layerCount);
    EncogUtilOutputIntArray("Layer Feed Counts:",net->layerFeedCounts, net->layerCount);
    EncogUtilOutputIntArray("Layer Index:",net->layerIndex, net->layerCount);
    EncogUtilOutputIntArray("Layer Weight Index:",net->weightIndex, net->layerCount);
}

ENCOG_NEURAL_NETWORK *EncogNetworkClone(ENCOG_NEURAL_NETWORK *net)
{
    ENCOG_NEURAL_NETWORK *result = (ENCOG_NEURAL_NETWORK *)EncogUtilAlloc(1,sizeof(ENCOG_NEURAL_NETWORK));
    if( net->firstBlock!=NULL )
    {
        abort();
    }

    memcpy(result,net,sizeof(ENCOG_NEURAL_NETWORK));
    result->activationFunctions = (ACTIVATION_FUNCTION*)EncogUtilDuplicateMemory(net->activationFunctions,net->layerCount,sizeof(ACTIVATION_FUNCTION));
    result->layerContextCount = (INT*)EncogUtilDuplicateMemory(net->layerContextCount,net->layerCount,sizeof(INT));
    result->biasActivation = (REAL*)EncogUtilDuplicateMemory(net->biasActivation,net->layerCount,sizeof(REAL));
    result->layerCounts = (INT*)EncogUtilDuplicateMemory(net->layerCounts,net->layerCount,sizeof(INT));
    result->layerFeedCounts = (INT*)EncogUtilDuplicateMemory(net->layerFeedCounts,net->layerCount,sizeof(INT));
    result->layerIndex = (INT*)EncogUtilDuplicateMemory(net->layerIndex,net->layerCount,sizeof(INT));
    result->layerOutput = (REAL*)EncogUtilDuplicateMemory(net->layerOutput,net->neuronCount,sizeof(REAL));
    result->layerSums = (REAL*)EncogUtilDuplicateMemory(net->layerSums,net->neuronCount,sizeof(REAL));
    result->weights = (REAL*)EncogUtilDuplicateMemory(net->weights,net->weightCount,sizeof(REAL));
    result->weightIndex = (INT*)EncogUtilDuplicateMemory(net->weightIndex,net->layerCount,sizeof(INT));

    return result;
}

