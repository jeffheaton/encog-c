#include "encog.h"

ENCOG_OBJECT *EncogTrainNew(ENCOG_NEURAL_NETWORK *net, ENCOG_DATA *data)
{
	char *ttype;
	ENCOG_TRAIN_PSO *pso;
	ENCOG_TRAIN_RPROP *rprop;
	int particles;
	float c1,c2;

	ttype = (char*)EncogHashGet(encogContext.config,"TRAIN");

	if( !EncogUtilStrcmpi(ttype,"RPROP") )
	{
		rprop = EncogTrainRPROPNew(net,data);
		return &rprop->encog;
	}
	else if( !EncogUtilStrcmpi(ttype,"PSO") )
	{
		particles = EncogHashGetInteger(encogContext.config,"PARTICLES",30);
		pso = EncogTrainPSONew(particles, net, data);
		pso->inertiaWeight = EncogHashGetFloat(encogContext.config,"INERTIA",0.4);
		pso->c1 = EncogHashGetFloat(encogContext.config,"C1",2.0);
		pso->c2 = EncogHashGetFloat(encogContext.config,"C2",2.0);
		EncogErrorCheck();
		pso->reportTarget = EncogTrainStandardCallback;
		return &pso->encog;
	}
	else {
		EncogErrorSet(ENCOG_ERROR_UNKNOWN_TRAINING);
		return NULL;
	}
}

ENCOG_TRAINING_REPORT *EncogTrainReport(ENCOG_OBJECT *train)
{
	int t;

	if( (t = EncogObjectGetType(train)) == -1 )
		return NULL;

	if( t==ENCOG_TYPE_PSO )
	{
		return &((ENCOG_TRAIN_PSO*)train)->currentReport;
	}
	else if( t==ENCOG_TYPE_PSO )
	{
		return &((ENCOG_TRAIN_RPROP*)train)->currentReport;
	}
	else
	{
		EncogErrorSet(ENCOG_ERROR_OBJECT_TYPE);
		return NULL;
	}
}

void EncogTrainRun(ENCOG_OBJECT *train, ENCOG_NEURAL_NETWORK *net)
{
	int t;

	if( (t = EncogObjectGetType(train)) == -1 )
		return;

	if( t==ENCOG_TYPE_PSO )
	{
	    EncogTrainPSORun((ENCOG_TRAIN_PSO*)train);
		EncogTrainPSOFinish((ENCOG_TRAIN_PSO*)train);
		EncogTrainPSOImportBest((ENCOG_TRAIN_PSO*)train,net);
	}
	else if( t==ENCOG_TYPE_RPROP )
	{
		EncogTrainRPROPRun((ENCOG_TRAIN_RPROP*)train);
	}
	else {
		EncogErrorSet(ENCOG_ERROR_OBJECT_TYPE);
	}
}