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

/**
	 * The default zero tolerance.
	 */
const double DEFAULT_ZERO_TOLERANCE = 0.00000000000000001;

	/**
	 * The POSITIVE ETA value. This is specified by the resilient propagation
	 * algorithm. This is the percentage by which the deltas are increased by if
	 * the partial derivative is greater than zero.
	 */
const double POSITIVE_ETA = 1.2;

	/**
	 * The NEGATIVE ETA value. This is specified by the resilient propagation
	 * algorithm. This is the percentage by which the deltas are increased by if
	 * the partial derivative is less than zero.
	 */
const double NEGATIVE_ETA = 0.5;

	/**
	 * The minimum delta value for a weight matrix value.
	 */
const double DELTA_MIN = 1e-6;

	/**
	 * The starting update for a delta.
	 */
const double DEFAULT_INITIAL_UPDATE = 0.1;

	/**
	 * The maximum amount a delta can reach.
	 */
const double DEFAULT_MAX_STEP = 50;

ENCOG_TRAIN_RPROP *EncogTrainRPROPNew(ENCOG_NEURAL_NETWORK *network, ENCOG_DATA *data)
{
	ENCOG_TRAIN_RPROP *result;
	INT i,maxThread;

	/* Clear out any previous errors */
	EncogErrorClear();

	maxThread = omp_get_max_threads();
	
	result = (ENCOG_TRAIN_RPROP *)EncogUtilAlloc(1,sizeof(ENCOG_TRAIN_RPROP));
	result->threadCount = maxThread;
	result->data = data;
	result->targetNetwork = network;	
	result->reportTarget = &EncogTrainStandardCallback;
	result->lastWeightChange = (REAL*)EncogUtilAlloc(network->weightCount,sizeof(REAL));
	result->updateValues = (REAL*)EncogUtilAlloc(network->weightCount,sizeof(REAL));
	result->gradients = (REAL*)EncogUtilAlloc(network->weightCount,sizeof(REAL));
	result->lastGradient = (REAL*)EncogUtilAlloc(network->weightCount,sizeof(REAL));
	result->layerDelta = (REAL**)EncogUtilAlloc(maxThread,sizeof(REAL*));
	result->network = (ENCOG_NEURAL_NETWORK**)EncogUtilAlloc(maxThread,sizeof(ENCOG_NEURAL_NETWORK*));
	memset(&result->currentReport,0,sizeof(ENCOG_TRAINING_REPORT));

	for(i=0;i<network->weightCount;i++)
	{
		result->updateValues[i] = DEFAULT_INITIAL_UPDATE;
		result->lastWeightChange[i] = 0;
		result->lastGradient[i] = 0;
	}

	for(i=0;i<maxThread;i++) 
	{
		result->layerDelta[i] = (REAL*)EncogUtilAlloc(network->neuronCount,sizeof(REAL));
		result->network[i] = (ENCOG_NEURAL_NETWORK*)EncogNetworkTransactionClone(network);
	}

	EncogObjectRegister(result, ENCOG_TYPE_RPROP);
	result->currentReport.trainer = (ENCOG_OBJECT*)result;

	return result;
}

static void _ProcessLevel(INT currentLevel,ENCOG_NEURAL_NETWORK *net, ENCOG_TRAIN_RPROP *rprop, REAL *layerDelta )
{
	int fromLayerIndex;
	int toLayerIndex;
	int fromLayerSize;
	int toLayerSize;
	int index;
	DERIVATIVE_FUNCTION df;
	REAL output,sum;
	int yi;
	int xi;
	int wi;
	int x,y;

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

#pragma omp critical
				{
				rprop->gradients[wi] += output * layerDelta[xi];
				}
				sum += net->weights[wi] * layerDelta[xi];
				wi += fromLayerSize;
				xi++;
			}

			layerDelta[yi] = sum
					* ((*df)(net->layerSums[yi],net->layerOutput[yi]));
			yi++;
		}
}

float _Process(ENCOG_TRAIN_RPROP *rprop, 
	ENCOG_NEURAL_NETWORK *net,
	REAL *layerDelta, REAL *input, REAL *ideal, double s) 
{
	REAL delta;
	INT i;
	float errorSum;

	EncogNetworkCompute(net,input,NULL);

	errorSum = 0;
	for(i=0; i<net->outputCount; i++)
	{
		delta = ideal[i] - net->layerOutput[i];
		errorSum += (float)(delta*delta);
		layerDelta[i] = (*net->derivativeFunctions)(net->layerSums[i],net->layerOutput[i])*delta;
	}

	for(i=0;i<net->layerCount;i++)
	{
		_ProcessLevel(i,net,rprop,layerDelta);
	}

	return errorSum;
}

	/**
	 * Determine the sign of the value.
	 * 
	 * @param value
	 *            The value to check.
	 * @return -1 if less than zero, 1 if greater, or 0 if zero.
	 */
	 int sign(double value) {
		if (fabs(value) < 0.000000001) {
			return 0;
		} else if (value > 0) {
			return 1;
		} else {
			return -1;
		}
	}

void _UpdateRPROPWeight(int index, ENCOG_TRAIN_RPROP *rprop)
{
	int change;
	REAL delta;
	REAL *gradients;
	REAL *lastGradient;
	REAL *updateValues;
	REAL weightChange;
	REAL *lastWeightChange;
	ENCOG_NEURAL_NETWORK *net;

	net = rprop->targetNetwork;
	gradients = rprop->gradients;
	lastGradient = rprop->lastGradient;
	updateValues = rprop->updateValues;
	lastWeightChange = rprop->lastWeightChange;

	// multiply the current and previous gradient, and take the
	// sign. We want to see if the gradient has changed its sign.
	change = sign(gradients[index] * lastGradient[index]);
	weightChange = 0;

		// if the gradient has retained its sign, then we increase the
		// delta so that it will converge faster
		if (change > 0) {
			double delta = updateValues[index]
					* POSITIVE_ETA;
			delta = MIN(delta, DEFAULT_MAX_STEP);
			weightChange = sign(gradients[index]) * delta;
			updateValues[index] = delta;
			lastGradient[index] = gradients[index];
		} else if (change < 0) {
			// if change<0, then the sign has changed, and the last
			// delta was too big
			double delta = updateValues[index]
					* NEGATIVE_ETA;
			delta = MAX(delta, DELTA_MIN);
			updateValues[index] = delta;
			weightChange = -lastWeightChange[index];
			// set the previous gradent to zero so that there will be no
			// adjustment the next iteration
			lastGradient[index] = 0;
		} else if (change == 0) {
			// if change==0 then there is no change to the delta
			delta = updateValues[index];
			weightChange = sign(gradients[index]) * delta;
			lastGradient[index] = gradients[index];
		}

		// apply the weight change, if any
		net->weights[index]+=weightChange;
}

float EncogTrainRPROPRun(ENCOG_TRAIN_RPROP *rprop)
{
	int i, tid;
    REAL *input,*ideal;
	float errorSum;
	ENCOG_DATA *data;

	/* Clear out any previous errors */
	EncogErrorClear();

	rprop->currentReport.iterations = 0;
	rprop->currentReport.lastUpdate = 0;
	rprop->currentReport.stopRequested = 0;
	rprop->currentReport.trainingStarted = time(NULL);

	data = rprop->data;

	while( !rprop->currentReport.stopRequested )
	{
		errorSum = 0;
		memset(rprop->gradients,0, sizeof(REAL)*rprop->targetNetwork->weightCount);

		#pragma omp parallel for private(i,input,ideal, tid) reduction(+:errorSum) default(shared)
		for(i=0; i<(int)data->recordCount; i++)
		{
			tid = omp_get_thread_num();
	        input = EncogDataGetInput(data,i);
		    ideal = EncogDataGetIdeal(data,i);

			errorSum = errorSum + _Process(rprop,
				rprop->network[tid],
				rprop->layerDelta[tid], input, ideal, 1.0);
		}

		// now learn!

		for(i=0;i<(int)rprop->targetNetwork->weightCount;i++) {
			_UpdateRPROPWeight(i,rprop);
			//net->weights[i]+=rprop->gradients[i]*0.7;
		}

		rprop->currentReport.error = (float)(errorSum/(data->recordCount*data->idealCount));
		rprop->currentReport.iterations++;
		rprop->reportTarget(&rprop->currentReport);
	}

    return rprop->currentReport.error;
}
