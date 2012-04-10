#include "encog.h"
#include "asa047.h"

ENCOG_TRAIN_NM *EncogTrainNMNew(ENCOG_NEURAL_NETWORK *network, ENCOG_DATA *data)
{
	ENCOG_TRAIN_NM *result;
	int i,maxThread;

	/* Clear out any previous errors */
	EncogErrorClear();

	maxThread = omp_get_max_threads();

	result = (ENCOG_TRAIN_NM *)EncogUtilAlloc(1,sizeof(ENCOG_TRAIN_RPROP));

	result->data = data;
	result->targetNetwork = network;	
	result->reportTarget = &EncogTrainStandardCallback;
	result->network = (ENCOG_NEURAL_NETWORK**)EncogUtilAlloc(maxThread,sizeof(ENCOG_NEURAL_NETWORK*));
	memset(&result->currentReport,0,sizeof(ENCOG_TRAINING_REPORT));

	for(i=0;i<maxThread;i++) 
	{
		//result->layerDelta[i] = (REAL*)EncogUtilAlloc(network->neuronCount,sizeof(REAL));
		result->network[i] = (ENCOG_NEURAL_NETWORK*)EncogNetworkTransactionClone(network);
	}

	EncogObjectRegister(result, ENCOG_TYPE_NM);
	result->currentReport.trainer = (ENCOG_OBJECT*)result;

	return result;
}

double _evaluate ( double x[] )
{
}

float EncogTrainNMRun(ENCOG_TRAIN_NM *nm)
{
	int n,i;
    REAL *input,*ideal,delta;
	float errorSum;
	ENCOG_DATA *data;
	double ynewlo;
	double reqmin;
	double *step;
	double *start;
	double *xmin;
	int konvge; 
	int kcount;
	int icount; 
	int numres; 
	int ifault;

	/* Clear out any previous errors */
	EncogErrorClear();

	nm->currentReport.iterations = 0;
	nm->currentReport.lastUpdate = 0;
	nm->currentReport.stopRequested = 0;
	nm->currentReport.trainingStarted = time(NULL);

	data = nm->data;
	n = nm->network[0]->weightCount;
	start = nm->network[0]->weights;
	reqmin = 0;
	step = (double*)EncogUtilAlloc(n,sizeof(double));
	xmin = (double*)EncogUtilAlloc(n,sizeof(double));
	konvge = 0;
	kcount = 0;

	for(i=0;i<n;i++) 
	{
		step[i] = 1;
	}

	nelmin ( _evaluate, n, start, xmin, &ynewlo, 0, step, konvge, kcount, &icount, &numres, &ifault );

	nm->currentReport.error = ynewlo;

	EncogUtilFree(step);

    return nm->currentReport.error;
}