#include "encog.h"

ENCOG_OBJECT *EncogTrainNew(ENCOG_NEURAL_NETWORK *net, ENCOG_DATA *data)
{
	char *ttype;
	ENCOG_TRAIN_PSO *pso;
	ENCOG_TRAIN_RPROP *rprop;
	ENCOG_TRAIN_NM *nm;
	int particles;
	float c1,c2;

	ttype = (char*)EncogHashGet(encogContext.config,PARAM_TRAIN, "PSO");

	if( !EncogUtilStrcmpi(ttype,TRAIN_TYPE_RPROP) )
	{
		rprop = EncogTrainRPROPNew(net,data);
		return &rprop->encog;
	}
	else if( !EncogUtilStrcmpi(ttype,TRAIN_TYPE_PSO) )
	{
		particles = EncogHashGetInteger(encogContext.config,PARAM_PARTICLES,30);
		pso = EncogTrainPSONew(particles, net, data);
		EncogErrorCheck();
		pso->reportTarget = EncogTrainStandardCallback;
		return &pso->encog;
	}
	else if( !EncogUtilStrcmpi(ttype,TRAIN_TYPE_NM) )
	{
		nm = EncogTrainNMNew(net,data);
		return &nm->encog;
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
	else if( t==ENCOG_TYPE_RPROP )
	{
		return &((ENCOG_TRAIN_RPROP*)train)->currentReport;
	}
	else if( t==ENCOG_TYPE_NM )
	{
		return &((ENCOG_TRAIN_NM*)train)->currentReport;
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
	else if( t==ENCOG_TYPE_NM )
	{
		EncogTrainNMRun((ENCOG_TRAIN_NM*)train);
	}
	else {
		EncogErrorSet(ENCOG_ERROR_OBJECT_TYPE);
	}
}

void EncogTrainSetCallback(ENCOG_OBJECT *train, ENCOG_REPORT_FUNCTION callback)
{
	int t;

	if( (t = EncogObjectGetType(train)) == -1 )
		return;

	if( t==ENCOG_TYPE_PSO )
	{
		((ENCOG_TRAIN_PSO*)train)->reportTarget = callback;
	}
	else if( t==ENCOG_TYPE_RPROP )
	{
		((ENCOG_TRAIN_RPROP*)train)->reportTarget = callback;
	}
	else if( t==ENCOG_TYPE_NM )
	{
		((ENCOG_TRAIN_NM*)train)->reportTarget = callback;
	}
	else {
		EncogErrorSet(ENCOG_ERROR_OBJECT_TYPE);
	}
}
