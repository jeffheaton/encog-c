#include "encog.h"

ENCOG_TRAIN_RPROP *EncogTrainRPROPNew(ENCOG_NEURAL_NETWORK *network, ENCOG_DATA *data)
{
	ENCOG_TRAIN_RPROP *result;

	/* Clear out any previous errors */
	EncogErrorClear();


	result = (ENCOG_TRAIN_RPROP *)EncogUtilAlloc(1,sizeof(ENCOG_TRAIN_RPROP));

	result->data = data;
	result->network = network;	
	result->reportTarget = &EncogTrainStandardCallback;
	result->deltas = (REAL*)EncogUtilAlloc(network->weightCount,sizeof(REAL));
	result->gradients = (REAL*)EncogUtilAlloc(network->weightCount,sizeof(REAL));

	memset(&result->currentReport,0,sizeof(ENCOG_TRAINING_REPORT));

	return result;
}

static void _ProcessLevel(INT currentLevel,ENCOG_TRAIN_RPROP *rprop )
{
	int fromLayerIndex;
	int toLayerIndex;
	int fromLayerSize;
	int toLayerSize;
	ENCOG_NEURAL_NETWORK *net;
	int index;
	DERIVATIVE_FUNCTION df;
	REAL output,sum;
	int yi;
	int xi;
	int wi;
	int x,y;

	net = rprop->network;
	fromLayerIndex = net->layerIndex[currentLevel + 1];
	toLayerIndex = net->layerIndex[currentLevel];
	fromLayerSize = net->layerCounts[currentLevel + 1];
	toLayerSize = net->layerFeedCounts[currentLevel];

	index = net->weightIndex[currentLevel];

	df = net->derivativeFunctions[currentLevel+1];

		// handle weights
		yi = fromLayerIndex;
		for (y = 0; y < fromLayerSize; y++) {
			output = net->layerOutput[yi];
			sum = 0;
			xi = toLayerIndex;
			wi = index + y;
			for (x = 0; x < toLayerSize; x++) {
				rprop->gradients[wi] += output * rprop->layerDelta[xi];
				sum += net->weights[wi] * rprop->layerDelta[xi];
				wi += fromLayerSize;
				xi++;
			}

			rprop->layerDelta[yi] = sum
					* ((*df)(net->layerSums[yi],net->layerOutput[yi]));
			yi++;
		}
}

void _Process(ENCOG_TRAIN_RPROP *rprop, REAL *input, REAL *ideal, double s) 
{
	ENCOG_NEURAL_NETWORK *net;
	REAL delta;
	INT i;

	net = rprop->network;

	EncogNetworkCompute(net,input,NULL);

	for(i=0; i<net->outputCount; i++)
	{
		delta = net->layerOutput[i] - ideal[i];
		rprop->layerDelta[i] = (*net->derivativeFunctions)(net->layerSums[i],net->layerOutput[i])*delta;
		rprop->errorSum+=delta*delta;
	}

	for(i=0;i<net->layerCount;i++)
	{
		_ProcessLevel(i,rprop);
	}
}

float EncogTrainRPROPRun(ENCOG_TRAIN_RPROP *rprop)
{
	INT i,j;
    REAL *input,*ideal,delta,sum;
	float error = 0.0;
	ENCOG_DATA *data;
	ENCOG_NEURAL_NETWORK *net;

	/* Clear out any previous errors */
	EncogErrorClear();

	rprop->currentReport.iterations = 0;
	rprop->currentReport.lastUpdate = 0;
	rprop->currentReport.stopRequested = 0;
	rprop->currentReport.trainingStarted = time(NULL);

	net = rprop->network;
	data = rprop->data;

	while( !rprop->currentReport.stopRequested )
	{
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

		error = (float)(sum/data->recordCount);
	}

    return error;
}
