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
	ENCOG_NEURAL_NETWORK *network;

	/* Clear out any previous errors */
	EncogErrorClear();

    network = (ENCOG_NEURAL_NETWORK*)EncogUtilAlloc(1,sizeof(ENCOG_NEURAL_NETWORK));
    return network;
}

NETWORK_LAYER *EncogNetworkCreateLayer(NETWORK_LAYER *prevLayer, int count, ACTIVATION_FUNCTION af, unsigned char bias)
{
    NETWORK_LAYER *result;
	
	/* Clear out any previous errors */
	EncogErrorClear();
	
	result = (NETWORK_LAYER *)EncogUtilAlloc(1,sizeof(NETWORK_LAYER));
    
    result->feedCount = count;
    result->totalCount = count + ((bias==1)?1:0);
    result->af = af;
    result->bias = bias;
	result->next = prevLayer;

	return result;

}

void EncogNetworkDelete(ENCOG_NEURAL_NETWORK *net)
{
	/* Clear out any previous errors */
	EncogErrorClear();
    EncogUtilFree(net);
}

ENCOG_NEURAL_NETWORK *EncogNetworkFinalizeStructure(NETWORK_LAYER *firstLayer, int freeLayers)
{
	ENCOG_NEURAL_NETWORK *result;
    NETWORK_LAYER *current;
    int index;
	int sizeofNetwork;
	int layerCount, neuronCount, weightCount;

	/* Clear out any previous errors */
	EncogErrorClear();

	if( firstLayer->next==NULL || firstLayer->next->next==NULL ) {		
		EncogErrorSet(ENCOG_ERROR_MIN_2LAYER);
		return NULL;
	}

	/* loop over the layers and calculate the network counts */
	layerCount = 0;
	neuronCount = 0;
	weightCount = 0;
	current = firstLayer;

	while(current!=NULL) {
		layerCount++;
		neuronCount+=current->totalCount;
		if (current->next != NULL)
        {
            weightCount += current->feedCount * current->next->totalCount;
        }

		current = current->next;
	}

	/* calculate how big the network is */
	sizeofNetwork = EncogNetworkDetermineSize(layerCount,neuronCount,weightCount);

    /* Create the network and lineup internal pointers */
    index = 0;
	result = (ENCOG_NEURAL_NETWORK*)EncogUtilAlloc(1,sizeofNetwork);

	/* Set initial values */
	result->beginTraining = 0;
	result->connectionLimit = 0;

	result->neuronCount = neuronCount;
    result->weightCount = weightCount;
	result->layerCount = layerCount;

	result->endTraining = result->layerCount-1;
	result->hasContext = 0;
	result->memorySize = sizeofNetwork;

	EncogNetworkLink(result);

	/* now set all of the values from the layers */
    current = firstLayer;
    while(current!=NULL)
    {
        NETWORK_LAYER *next = (NETWORK_LAYER *)current->next;
        result->layerCounts[index]=current->totalCount;
        result->layerFeedCounts[index]=current->feedCount;
        result->biasActivation[index]=current->bias;
        result->activationFunctions[index]=current->af;

        if (index == 0)
        {
            result->weightIndex[index] = 0;
            result->layerIndex[index] = 0;
        }
        else
        {
            result->weightIndex[index] = result->weightIndex[index - 1]
                                      + (result->layerCounts[index] * result->layerFeedCounts[index - 1]);
            result->layerIndex[index] = result->layerIndex[index - 1]
                                     + result->layerCounts[index - 1];
        }


        index++;
        current=(NETWORK_LAYER*)current->next;
    }

	result->inputCount = result->layerFeedCounts[result->layerCount-1];
	result->outputCount = result->layerFeedCounts[0];

    EncogNetworkClearContext(result);
	return result;
}

INT EncogNetworkDetermineSize(INT layerCount, INT neuronCount, INT weightCount) {
	INT sizeofNetwork;

	sizeofNetwork = sizeof(ENCOG_NEURAL_NETWORK);

	sizeofNetwork+=layerCount*sizeof(INT); // net->layerCounts
	sizeofNetwork+=layerCount*sizeof(REAL); // net->biasActivation
	sizeofNetwork+=layerCount*sizeof(ACTIVATION_FUNCTION); // net->activationFunctions
	sizeofNetwork+=layerCount*sizeof(INT); // net->layerContextCount 
	sizeofNetwork+=layerCount*sizeof(INT); // net->weightIndex
	sizeofNetwork+=layerCount*sizeof(INT); // net->layerIndex
	sizeofNetwork+=layerCount*sizeof(INT); // net->layerFeedCounts
	sizeofNetwork+=layerCount*sizeof(REAL); // net->biasActivation
	sizeofNetwork+=layerCount*sizeof(INT); // net->contextTargetOffset
	sizeofNetwork+=layerCount*sizeof(INT); // net->contextTargetSize
	sizeofNetwork+=weightCount*sizeof(REAL); // net->weights
    sizeofNetwork+=neuronCount*sizeof(REAL); // net->layerOutput
    sizeofNetwork+=neuronCount*sizeof(REAL); // net->layerSums
	return sizeofNetwork;
}

void EncogNetworkLink(ENCOG_NEURAL_NETWORK *net)
{
	unsigned char *ptr;

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
	assert( (ptr-((unsigned char*)net)) == net->memorySize);
}

void EncogNetworkCompute(ENCOG_NEURAL_NETWORK *net,REAL *input, REAL *output)
{
    int i;
    int sourceIndex;

	/* Clear out any previous errors */
	EncogErrorClear();

	if( net->weights == NULL ) {
		EncogErrorSet(ENCOG_ERROR_NETWORK_NOT_FINALIZED);
		return;
	}
	
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

void EncogNetworkRandomizeRange(ENCOG_NEURAL_NETWORK *net,REAL low, REAL high)
{
    INT i;
    REAL d;

	/* Clear out any previous errors */
	EncogErrorClear();

    for(i=0; i<net->weightCount; i++)
    {
        d = (REAL)rand()/(REAL)RAND_MAX;
        d = (d*(high-low))+low;
        net->weights[i] = d;
    }
}


void EncogNetworkImportWeights(ENCOG_NEURAL_NETWORK *net, REAL *w)
{
	/* Clear out any previous errors */
	EncogErrorClear();

	if( net->weights == NULL ) {
		EncogErrorSet(ENCOG_ERROR_NETWORK_NOT_FINALIZED);
		return;
	}

    memcpy(net->weights,w,net->weightCount*sizeof(REAL));
}

void EncogNetworkExportWeights(ENCOG_NEURAL_NETWORK *net, REAL *w)
{
	/* Clear out any previous errors */
	EncogErrorClear();

	if( net->weights == NULL ) {
		EncogErrorSet(ENCOG_ERROR_NETWORK_NOT_FINALIZED);
		return;
	}

    memcpy(w,net->weights,net->weightCount*sizeof(REAL));
}

void EncogNetworkClearContext(ENCOG_NEURAL_NETWORK *net)
{
    INT index = 0;
    INT hasBias;
    INT i;
    INT j;

	/* Clear out any previous errors */
	EncogErrorClear();

	if( net->weights == NULL ) {
		EncogErrorSet(ENCOG_ERROR_NETWORK_NOT_FINALIZED);
		return;
	}

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
	/* Clear out any previous errors */
	EncogErrorClear();

	if( net->weights == NULL ) {
		EncogErrorSet(ENCOG_ERROR_NETWORK_NOT_FINALIZED);
		return;
	}

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
    ENCOG_NEURAL_NETWORK *result;

	/* Clear out any previous errors */
	EncogErrorClear();

	result = (ENCOG_NEURAL_NETWORK *)EncogUtilDuplicateMemory(net,1,net->memorySize);
	EncogNetworkLink(result);

    return result;
}

ENCOG_NEURAL_NETWORK *EncogNetworkFactory(char *method, char *architecture, int defaultInputCount, int defaultOutputCount)
{
	char line[MAX_STR];
	ENCOG_NEURAL_NETWORK *network;
	int bias, phase, neuronCount;
	char *ptrBegin, *ptrEnd, *ptrMid;
	ACTIVATION_FUNCTION activation;
	NETWORK_LAYER *currentLayer;


	/* Clear out any previous errors */
	EncogErrorClear();

    network = EncogNetworkNew();

	strncpy(line,architecture,MAX_STR);
	EncogUtilStrupr(line);

	activation = EncogActivationLinear;
	ptrBegin = line;
	phase = 0;
	currentLayer = NULL;

	do {		
		ptrEnd = strstr(ptrBegin,"->");

		if( ptrEnd!=NULL ) {
			*ptrEnd = 0;						
		}

		bias = 0;
		ptrMid = strstr(ptrBegin,":");

		if( ptrMid!=NULL ) {
			*ptrMid = 0;
			ptrMid++;
			while( *ptrMid==' ' || *ptrMid=='\t' ) {
				ptrMid++;
			}
		
			if( *ptrMid=='B' ) {
				bias = 1;
			} else {
				EncogNetworkDelete(network);				
				EncogErrorSet(ENCOG_ERROR_FACTORY_INVALID_BIAS);				
				return NULL;
			}
		}

		if( !strcmp(ptrBegin,"SIGMOID") ) {
			activation = EncogActivationSigmoid;
		} else if(!strcmp(ptrBegin,"TANH") ) {
			activation = EncogActivationTANH;
		} else if(!strcmp(ptrBegin,"LINEAR") ) {
			activation = EncogActivationLinear;
		} else {
			if(!strcmp(ptrBegin,"?") ) {
				if( phase==0 ) {
					neuronCount = defaultInputCount;
				} else if(phase==1 ) {
					neuronCount = defaultOutputCount;
				} else {					
					EncogNetworkDelete(network);				
					EncogErrorSet(ENCOG_ERROR_FACTORY_INVALID_COND);
					return NULL;
				}
				phase++;
			} else {
				neuronCount = atoi(ptrBegin);
			}

			if( neuronCount==0 ) {				
				EncogNetworkDelete(network);
				EncogErrorSet(ENCOG_ERROR_FACTORY_INVALID_ACTIVATION);
				return NULL;
			}

			currentLayer = EncogNetworkCreateLayer(currentLayer,neuronCount,activation,bias);
			if( EncogErrorGet()!=ENCOG_ERROR_OK ) {
				EncogNetworkDelete(network);
				return NULL;
			}
		}
		if( ptrEnd!=NULL ) {
			ptrBegin = ptrEnd+2;
		}
	} while(ptrEnd!=NULL);

	network = EncogNetworkFinalizeStructure(currentLayer,1);
	if( network==NULL ) {
		return NULL;
	}

/* Randomize the neural network weights */
    EncogNetworkRandomizeRange(network,-1,1);
	if( EncogErrorGet()!=ENCOG_ERROR_OK ) {
		EncogNetworkDelete(network);
		return NULL;
	}

    return network;
}
