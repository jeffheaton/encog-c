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



static float _CalculatePSOError(ENCOG_TRAIN_PSO *pso, ENCOG_NEURAL_NETWORK *network) 
{
	float result;
	float start,stop;

#ifdef ENCOG_CUDA
	if( encogContext.gpuEnabled && omp_get_thread_num()==0 ) 
	{		
		result = EncogCUDAErrorSSE(pso->device, network);
	}
	else 
	{
		start = omp_get_wtime();
		result = EncogCPUErrorSSE( network, pso->data);
		stop = omp_get_wtime();

		#pragma omp critical 
		{
			pso->cpuWorkUnitTime+=(stop-start);
			pso->cpuWorkUnitCalls++;
		}
		return result;
	}	
	return result;
#else
	start = (float)omp_get_wtime();
	result = EncogErrorSSE( network, pso->data);
	stop = (float)omp_get_wtime();
	#pragma omp critical 
	{
		pso->cpuWorkUnitTime+=(stop-start);
		pso->cpuWorkUnitCalls++;
	}

	return result;
#endif
}


/* Internal functions */


static void _UpdateGlobalBestPosition(ENCOG_TRAIN_PSO *pso)
{
    int bestUpdated = 0, i;
    ENCOG_PARTICLE *best,*particle;

    if( pso->bestParticle!=-1 )
    {
        best = &pso->particles[pso->bestParticle];
    }
    else
    {
        best = NULL;
    }


    for (i = 0; i < pso->populationSize; i++)
    {
        particle = &pso->particles[i];
        if ((best==NULL) || particle->bestError<best->bestError )
        {
            pso->bestParticle = i;
            best = particle;
            bestUpdated = 1;
        }
    }
    if (bestUpdated)
    {
        EncogVectorCopy(pso->bestVector, best->network->weights,pso->dimensions);
        pso->bestError = best->bestError;
    }
}

/* API functions */

ENCOG_TRAIN_PSO *EncogTrainPSONew(int populationSize, ENCOG_NEURAL_NETWORK *model, ENCOG_DATA *data)
{
    int i;
    ENCOG_PARTICLE *particle;
    ENCOG_TRAIN_PSO *pso;
    ENCOG_NEURAL_NETWORK *clone;

	/* Clear out any previous errors */
	EncogErrorClear();

    pso = (ENCOG_TRAIN_PSO*)EncogUtilAlloc(1,sizeof(ENCOG_TRAIN_PSO));
	pso->inertiaWeight = EncogHashGetFloat(encogContext.config,PARAM_INERTIA,(float)0.4);
	pso->c1 = EncogHashGetFloat(encogContext.config,PARAM_C1,2.0);
	pso->c2 = EncogHashGetFloat(encogContext.config,PARAM_C2,2.0);
    pso->populationSize = populationSize;
    pso->maxPosition = EncogHashGetFloat(encogContext.config,PARAM_MAXPOS,(float)-1);
    pso->maxVelocity = EncogHashGetFloat(encogContext.config,PARAM_MAXVEL,(float)2);
    pso->bestParticle = -1;
    pso->dimensions = model->weightCount;
    pso->data = data;
    pso->bestVector = (REAL*)EncogUtilAlloc(model->weightCount,sizeof(REAL));
	pso->reportTarget = &EncogTrainStandardCallback;

	memset(&pso->currentReport,0,sizeof(ENCOG_TRAINING_REPORT));

    /* construct the arrays */

    pso->particles = (ENCOG_PARTICLE*)EncogUtilAlloc(populationSize,sizeof(ENCOG_PARTICLE));

#ifdef ENCOG_CUDA
	if( encogContext.gpuEnabled ) {
		pso->device = EncogGPUDeviceNew(0, model, data);
	}
#endif
	
    for(i=0; i<populationSize; i++)
    {
        particle = &pso->particles[i];
        clone  = EncogNetworkClone(model);
	particle->index = i;
		particle->pso = (struct ENCOG_TRAIN_PSO*)pso;
        particle->network = clone;
        particle->velocities = (REAL*)EncogUtilAlloc(clone->weightCount,sizeof(REAL));
        particle->vtemp = (REAL*)EncogUtilAlloc(clone->weightCount,sizeof(REAL));
        particle->bestVector = (REAL*)EncogUtilAlloc(clone->weightCount,sizeof(REAL));
        particle->bestError = 1.0;
		particle->particleState = PARTICLE_STATE_CALC;
		if( i>0 ) {
			EncogNetworkRandomizeRange(particle->network,-1,1);
		}
        EncogNetworkExportWeights(particle->network,particle->bestVector);
        EncogVectorRandomise(particle->velocities, pso->maxVelocity, clone->weightCount);
    }

	EncogObjectRegister(pso, ENCOG_TYPE_PSO);
	pso->currentReport.trainer = (ENCOG_OBJECT*)pso;

    _UpdateGlobalBestPosition(pso);
    return pso;
}

/**
 * Update the personal best position of a particle.
 *
 * @param particleIndex     index of the particle in the swarm
 * @param particlePosition  the particle current position vector
 */
static void _PSOPerformCalc(ENCOG_PARTICLE *particle)
{
    float score;

    // set the network weights and biases from the vector
    score = _CalculatePSOError( particle->pso, particle->network);

    // update the best vectors (g and i)
    if ( (particle->bestError == 0) || score<particle->bestError)
    {
        particle->bestError = score;
        EncogVectorCopy(particle->bestVector, particle->network->weights, particle->pso->dimensions);

		#pragma omp critical 
		{
			if( particle->pso->bestError == 0 || score<particle->pso->bestError ) {
				EncogVectorCopy(particle->pso->bestVector, particle->network->weights,particle->pso->dimensions);
				particle->pso->bestError = score;
				particle->pso->bestParticle = particle->index;
			}
		}
    }

	particle->particleState = PARTICLE_STATE_ITERATION;
}

static void _PSOPerformMove(ENCOG_PARTICLE *particle)
{	
	ENCOG_TRAIN_PSO *pso;
	INT i;

	pso = (ENCOG_TRAIN_PSO *)particle->pso;

	for(i=0;i<(INT)pso->dimensions;i++) {
		// update velocity
		particle->velocities[i] *= pso->inertiaWeight;

		// cognitive term
		particle->velocities[i]+=(particle->bestVector[i]-particle->network->weights[i]) * pso->c1 *  ((REAL)rand()/(REAL)RAND_MAX);

	    // social term
		if (particle->index != pso->bestParticle)
		{
			particle->velocities[i]+=(pso->bestVector[i]-particle->network->weights[i]) * pso->c2 *  ((REAL)rand()/(REAL)RAND_MAX);
		}

		// clamp
		if( pso->maxVelocity!=-1 ) {
			particle->velocities[i] = MAX(particle->velocities[i],-pso->maxVelocity);
			particle->velocities[i] = MIN(particle->velocities[i],pso->maxVelocity);
		}


		// update weights
		particle->network->weights[i]+=particle->velocities[i];
		if( pso->maxPosition!=-1 ) {
			particle->network->weights[i] = MAX(particle->network->weights[i],-pso->maxPosition);
			particle->network->weights[i] = MIN(particle->network->weights[i],pso->maxPosition);
		}
	}	

	particle->particleState = PARTICLE_STATE_CALC;
}

ENCOG_PARTICLE *_getNextParticle(ENCOG_TRAIN_PSO *pso)
{
	ENCOG_PARTICLE *result = NULL;
	int i;

	/* First preference is to move */

	for(i=0;i<pso->populationSize;i++) {
		result = &pso->particles[i];
		if( result->particleState == PARTICLE_STATE_MOVE ) {
			result->particleState = PARTICLE_STATE_MOVING;
			return result;
		}
	}

	/* Second preference is to recalculate */

	for(i=0;i<pso->populationSize;i++) {
		result = &pso->particles[i];
		if( result->particleState == PARTICLE_STATE_CALC ) {
			result->particleState = PARTICLE_STATE_CALCING;
			return result;
		}
	}

	/* Third, we set everyone back to move, and return the last particle */
	for(i=0;i<pso->populationSize;i++) {
		result = &pso->particles[i];
		result->particleState = PARTICLE_STATE_MOVE;
	}

	pso->currentReport.iterations++;
	pso->currentReport.error = pso->bestError;	
	pso->reportTarget(&pso->currentReport);

	return result;
}

float EncogTrainPSORun(ENCOG_TRAIN_PSO *pso)
{
    ENCOG_PARTICLE *particle;

	/* Clear out any previous errors */
	EncogErrorClear();

	pso->currentReport.iterations = 0;
	pso->currentReport.lastUpdate = 0;
	pso->currentReport.stopRequested = 0;
	pso->currentReport.trainingStarted = time(NULL);

	#pragma omp parallel private(particle) 
	{
		while( !pso->currentReport.stopRequested )
		{
			#pragma omp critical 
			{
				particle = _getNextParticle(pso);
			}

			if( particle->particleState == PARTICLE_STATE_CALCING ) 
			{
				//printf("##Calc: %i\n",particle->index);
				_PSOPerformCalc(particle);
			} else if( particle->particleState == PARTICLE_STATE_MOVING ) 
			{
				//printf("##Move: %i\n",particle->index);
				_PSOPerformMove(particle);				
			}
		}
	} 

    return pso->bestError;
}

void EncogTrainPSOImportBest(ENCOG_TRAIN_PSO *pso, ENCOG_NEURAL_NETWORK *net)
{
    ENCOG_PARTICLE *particle;

	/* Clear out any previous errors */
	EncogErrorClear();

    particle = &pso->particles[pso->bestParticle];
    EncogNetworkImportWeights(net,particle->bestVector);
}

void EncogTrainPSOFinish(ENCOG_TRAIN_PSO *pso) {
#ifdef ENCOG_CUDA
	if( encogContext.gpuEnabled ) {
		pso->cudaKernelTime=pso->device->perfKernelTime/pso->device->perfCount;
		pso->cudaKernelCalls=pso->device->perfCount;
	}
#endif
}
