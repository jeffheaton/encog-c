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



static float _CalculatePSOError(ENCOG_TRAIN_PSO *pso, ENCOG_NEURAL_NETWORK *network) 
{
	float result;
	double start,stop;

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
	start = omp_get_wtime();
	result = EncogErrorSSE( network, pso->data);
	stop = omp_get_wtime();
	#pragma omp critical 
	{
		pso->cpuWorkUnitTime+=(stop-start);
		pso->cpuWorkUnitCalls++;
	}

	return result;
#endif
}


/* Internal functions */
/**
 * Update the personal best position of a particle.
 *
 * @param particleIndex     index of the particle in the swarm
 * @param particlePosition  the particle current position vector
 */
static void _UpdatePersonalBestPosition(ENCOG_TRAIN_PSO *pso, int particleIndex)
{
    ENCOG_PARTICLE *particle;
    float score;

    // set the network weights and biases from the vector
    particle = &pso->particles[particleIndex];
    score = _CalculatePSOError( pso, particle->network);

    // update the best vectors (g and i)
    if ( (particle->bestError == 0) || score<particle->bestError)
    {
        particle->bestError = score;
        EncogVectorCopy(particle->bestVector, particle->network->weights, pso->dimensions);
    }
}

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
    int cpuCount;

	/* Clear out any previous errors */
	EncogErrorClear();

    pso = (ENCOG_TRAIN_PSO*)EncogUtilAlloc(1,sizeof(ENCOG_TRAIN_PSO));
    pso->c1 = 2.0;
    pso->c2 = 2.0;
    pso->populationSize = populationSize;
    pso->inertiaWeight = (REAL)0.4;
    pso->maxPosition = (REAL)-1;
    pso->maxVelocity = (REAL)2;
    pso->pseudoAsynchronousUpdate = 0;
    pso->bestParticle = -1;
    pso->dimensions = model->weightCount;
    pso->data = data;
    pso->bestVector = (REAL*)EncogUtilAlloc(model->weightCount,sizeof(REAL));

    /* construct the arrays */

    pso->particles = (ENCOG_PARTICLE*)EncogUtilAlloc(populationSize,sizeof(ENCOG_PARTICLE));

#ifdef ENCOG_CUDA
	if( encogContext.gpuEnabled ) {
		pso->device = EncogGPUDeviceNew(0, model, data);
	}
#endif
	

	#pragma omp parallel for private(particle,clone) 
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
        particle->bestError = 0;
		if( i>0 ) {
			EncogNetworkRandomizeRange(particle->network,-1,1);
		}
        EncogNetworkExportWeights(particle->network,particle->bestVector);
        EncogVectorRandomise(particle->velocities, pso->maxVelocity, clone->weightCount);
        _UpdatePersonalBestPosition(pso, i);
    }

    _UpdateGlobalBestPosition(pso);
    return pso;
}

void EncogTrainPSODelete(ENCOG_TRAIN_PSO *pso)
{
    int i;
    ENCOG_PARTICLE *particle;

	/* Clear out any previous errors */
	EncogErrorClear();

    /* first delete the particles */
    for(i=0;i<pso->populationSize;i++) {
        particle = &pso->particles[i];
        EncogNetworkDelete(particle->network);
        EncogUtilFree(particle->velocities);
        EncogUtilFree(particle->bestVector);
        EncogUtilFree(particle->vtemp);
    }

    /* delete anything on the PSO, including particle structure */
    EncogUtilFree(pso->particles);
    EncogUtilFree(pso->bestVector);

    /* finally, delete the PSO */
    EncogUtilFree(pso);
}

static void _PSOTask(void *v)
{
	ENCOG_PARTICLE *particle;
	ENCOG_TRAIN_PSO *pso;
	INT i;

	particle = (ENCOG_PARTICLE *)v;

	pso = (ENCOG_TRAIN_PSO *)particle->pso;

	for(i=0;i<pso->dimensions;i++) {
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

	_UpdatePersonalBestPosition(pso, particle->index);
}

float EncogTrainPSOIterate(ENCOG_TRAIN_PSO *pso)
{
    int i;
    ENCOG_PARTICLE *particle;

	/* Clear out any previous errors */
	EncogErrorClear();

	#pragma omp parallel for
    for(i=0; i<pso->populationSize; i++)
    {
		particle = &pso->particles[i];
		_PSOTask(particle);
    }

    _UpdateGlobalBestPosition(pso);
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
	pso->cudaKernelTime=pso->device->perfKernelTime/pso->device->perfCount;
	pso->cudaKernelCalls=pso->device->perfCount;
#endif
}
