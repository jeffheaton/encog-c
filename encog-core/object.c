#include "encog.h"

static void _freeNeuralNetwork(ENCOG_NEURAL_NETWORK *net)
{
	EncogUtilFree(net->layerCounts);
    EncogUtilFree(net->biasActivation);
    EncogUtilFree(net->activationFunctions);
	EncogUtilFree(net->activationFunctionIDs);
    EncogUtilFree(net->layerContextCount);
    EncogUtilFree(net->weightIndex);
    EncogUtilFree(net->layerIndex);
    EncogUtilFree(net->layerFeedCounts);
	EncogUtilFree(net->contextTargetOffset);
	EncogUtilFree(net->contextTargetSize);

    EncogUtilFree(net);
}

static void _freeData(ENCOG_DATA *data)
{
	EncogUtilFree(data->data);
    EncogUtilFree(data);
}

static void _freePSO(ENCOG_TRAIN_PSO *pso)
{
	int i;
    ENCOG_PARTICLE *particle;

	/* Clear out any previous errors */
	EncogErrorClear();

    /* first delete the particles */
    for(i=0;i<pso->populationSize;i++) {
        particle = &pso->particles[i];
        EncogObjectFree(particle->network);
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

static void _freeHash(ENCOG_HASH *hash)
{
}


static void _freeRPROP(ENCOG_TRAIN_RPROP *hash)
{
}

void EncogObjectRegister(void *obj, int type) 
{
	ENCOG_OBJECT *encogObject;

	encogObject = (ENCOG_OBJECT *)obj;
	encogObject->id[0] = 'E';
	encogObject->id[1] = 'G';
	encogObject->type = type;
}

void EncogObjectValidate(void *obj, int type) 
{
	ENCOG_OBJECT *encogObject;
	EncogErrorClear();

	encogObject = (ENCOG_OBJECT *)obj;

	if( encogObject->id[0]!='E' || encogObject->id[1]!='G' )
	{
		EncogErrorSet(ENCOG_ERROR_OBJECT);
		return;
	}

	if(  encogObject->type != type)
	{
		EncogErrorSet(ENCOG_ERROR_OBJECT_TYPE);
		return;
	}
}

void EncogObjectFree(void *obj)
{
	ENCOG_OBJECT *encogObject;
	encogObject = (ENCOG_OBJECT *)obj;

	EncogErrorClear();

	if( encogObject->id[0]!='E' || encogObject->id[1]!='G' )
	{
		EncogErrorSet(ENCOG_ERROR_OBJECT);
		return;
	}

	switch(encogObject->type)
	{
		case ENCOG_TYPE_NEURAL_NETWORK:
			_freeNeuralNetwork((ENCOG_NEURAL_NETWORK*)obj);
			break;
		case ENCOG_TYPE_DATA:
			_freeData((ENCOG_DATA*)obj);
			break;
		case ENCOG_TYPE_PSO:
			_freePSO((ENCOG_TRAIN_PSO*)obj);
			break;
		case ENCOG_TYPE_RPROP:
			_freeRPROP((ENCOG_TRAIN_RPROP*)obj);
			break;
		case ENCOG_TYPE_HASH:
			_freeHash((ENCOG_HASH*)obj);
			break;
	}
}

int EncogObjectGetType(ENCOG_OBJECT *encogObject)
{
	if( encogObject->id[0]!='E' || encogObject->id[1]!='G' )
	{
		EncogErrorSet(ENCOG_ERROR_OBJECT);
		return -1;
	}

	return encogObject->type;
}

char *EncogObjectType(ENCOG_OBJECT *encogObject)
{
	switch(encogObject->type)
	{
		case ENCOG_TYPE_NEURAL_NETWORK:
			return "NEURAL_NETWORK";
		case ENCOG_TYPE_DATA:
			return "ENCOG_DATA";
		case ENCOG_TYPE_PSO:
			return "ENCOG_TRAIN_PSO";
		case ENCOG_TYPE_RPROP:
			return "ENCOG_TRAIN_RPROP";
		case ENCOG_TYPE_HASH:
			return "ENCOG_HASH";
		case ENCOG_TYPE_NM:
			return "ENCOG_TRAIN_NM";
		default:
			return "unknown";
	}
}


