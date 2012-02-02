#include "encog.h"

static int _CheckNetwork(FILE *fp)
{
	int index, v;
	char line[MAX_STR];
	char arg[MAX_STR];

	fgets(line,sizeof(line),fp);

	index = 0;

	/* Make sure this is an Encog file */
	index = EncogStrPopLine(line,arg,index,sizeof(arg));
	if( strcmp(arg,"encog") )
	{
		return -1;
	}

	/* Make sure this is a BasicNetwork */
	index = EncogStrPopLine(line,arg,index,sizeof(arg));
	if( strcmp(arg,"BasicNetwork") )
	{
		return -1;
	}

	/* Encog platform */
	index = EncogStrPopLine(line,arg,index,sizeof(arg));

	/* Encog version */
	index = EncogStrPopLine(line,arg,index,sizeof(arg));

	/* File version */
	index = EncogStrPopLine(line,arg,index,sizeof(arg));
	v = atoi(arg);

	if( v>1 ) 
	{
		return -1;
	}

	return 0;
}


static void _LoadBasic(char *line, ENCOG_NEURAL_NETWORK *network)
{
	char name[MAX_STR],value[MAX_STR];

	EncogStrStripCRLF(line);
	EncogStrParseNV(line,name,value,MAX_STR);

	if(!strcmp(name,"beginTraining") )
	{
		network->beginTraining = atoi(value);
	}
	else if(!strcmp(name,"connectionLimit") )
	{
		network->connectionLimit = atof(value);
	}
	else if(!strcmp(name,"contextTargetOffset") )
	{
		network->contextTargetOffset = EncogStrParseIntList(value);
	}
	else if(!strcmp(name,"contextTargetSize") )
	{
		network->contextTargetSize = EncogStrParseIntList(value);
	}
	else if(!strcmp(name,"endTraining") )
	{
		network->endTraining = atoi(value);
	}
	else if(!strcmp(name,"hasContext") )
	{
		network->hasContext = EncogStrParseBoolean(value);
	}
	else if(!strcmp(name,"inputCount") )
	{
		network->inputCount = atoi(value);
	}
	else if(!strcmp(name,"layerCounts") )
	{
		network->layerCount = EncogStrCountValues(value);
		network->layerCounts = EncogStrParseIntList(value);
		network->activationFunctions = (ACTIVATION_FUNCTION*)EncogUtilAlloc(network->layerCount,sizeof(ACTIVATION_FUNCTION));
	}
	else if(!strcmp(name,"layerFeedCounts") )
	{
		network->layerFeedCounts = EncogStrParseIntList(value);
	}
	else if(!strcmp(name,"layerContextCount") )
	{
		network->layerContextCount = EncogStrParseIntList(value);
	}
	else if(!strcmp(name,"layerIndex") )
	{
		network->layerIndex = EncogStrParseIntList(value);
	}
	else if(!strcmp(name,"output") )
	{
		network->neuronCount = EncogStrCountValues(value);
		network->layerOutput = EncogStrParseDoubleList(value);
	}
	else if(!strcmp(name,"outputCount") )
	{
		network->outputCount = atoi(value);
	}
	else if(!strcmp(name,"weightIndex") )
	{
		network->weightIndex = EncogStrParseIntList(value);
	}
	else if(!strcmp(name,"weights") )
	{
		network->weightCount = EncogStrCountValues(value);
		network->weights = EncogStrParseDoubleList(value);
	}
	else if(!strcmp(name,"biasActivation") )
	{
		network->biasActivation = EncogStrParseDoubleList(value);
	}	
}

static void _LoadActivation(char *line, ENCOG_NEURAL_NETWORK *network, int currentActivation)
{
	EncogStrStripQuotes(line);
	if( !strcmp(line,"ActivationLinear") )
	{
		network->activationFunctions[currentActivation] = &EncogActivationLinear;
	}
	else if( !strcmp(line,"ActivationSigmoid") )
	{
		network->activationFunctions[currentActivation] = &EncogActivationSigmoid;
	}
	else if( !strcmp(line,"ActivationTANH") )
	{
		network->activationFunctions[currentActivation] = EncogActivationTANH;
	}
}

ENCOG_NEURAL_NETWORK *EncogNetworkLoad(char *name)
{
	char line[MAX_STR];
	ENCOG_NEURAL_NETWORK *result;
	int mode, currentActivation;

	FILE *fp = fopen(name,"r");

	if( _CheckNetwork(fp) == -1 )
	{
		return NULL;
	}

	result = (ENCOG_NEURAL_NETWORK *)malloc(sizeof(ENCOG_NEURAL_NETWORK));
	mode = 0;
	currentActivation = 0;

	while( fgets(line,sizeof(line),fp) ) 
	{
		EncogStrStripCRLF(line);

		if(!strcmp(line,"[BASIC:NETWORK]") )
		{
			mode = 1;
		}
		else if(!strcmp(line,"[BASIC:ACTIVATION]") )
		{
			mode = 2;
		}
		else 
		{
			switch(mode)
			{
				case 1:
					_LoadBasic(line,result);
					break;
				case 2:
					_LoadActivation(line,result,currentActivation++);
					break;
			}
		}		
	}

	fclose(fp);
	return result;
}

void EncogNetworkSave(char *name, ENCOG_NEURAL_NETWORK *network)
{
	char line[MAX_STR];
	INT i;
	FILE *fp;
	time_t t;

	/* Write the header line */
	fp = fopen(name,"w");
	*line=0;
	strcat(line,"encog");
	strcat(line,",");
	strcat(line,"BasicNetwork");
	strcat(line,",");
	strcat(line,"c++");
	strcat(line,",");
	strcat(line,"3.0");
	strcat(line,",");
	strcat(line,"1");
	strcat(line,",");
	
	time(&t);
	*line = 0;
	EncogStrCatLong(line,t,MAX_STR);
	fputs(line,fp);
	fputs("\n[BASIC]\n",fp);
	fputs("[BASIC:PARAMS]\n",fp);
	fputs("[BASIC:NETWORK]\n",fp);
	EncogFileWriteValueInt(fp,"beginTraining",network->beginTraining);
	EncogFileWriteValueDouble(fp,"connectionLimit",network->connectionLimit);
	EncogFileWriteValueIntArray(fp,"contextTargetOffset",network->contextTargetOffset,network->layerCount);
	EncogFileWriteValueIntArray(fp,"contextTargetSize",network->contextTargetSize,network->layerCount);
	EncogFileWriteValueInt(fp,"endTraining",network->endTraining);
	EncogFileWriteValueBoolean(fp,"hasContext",network->hasContext);
	EncogFileWriteValueInt(fp,"inputCount",network->inputCount);
	EncogFileWriteValueIntArray(fp,"layerCounts",network->layerCounts,network->layerCount);
	EncogFileWriteValueIntArray(fp,"layerFeedCounts",network->layerFeedCounts,network->layerCount);
	EncogFileWriteValueIntArray(fp,"layerContextCount",network->layerContextCount,network->layerCount);
	EncogFileWriteValueIntArray(fp,"layerIndex",network->layerIndex,network->layerCount);
	EncogFileWriteValueDoubleArray(fp,"output",network->layerOutput,network->neuronCount);
	EncogFileWriteValueInt(fp,"outputCount",network->outputCount);
	EncogFileWriteValueIntArray(fp,"weightIndex",network->weightIndex,network->layerCount);
	EncogFileWriteValueDoubleArray(fp,"weights",network->weights,network->weightCount);
	EncogFileWriteValueDoubleArray(fp,"biasActivation",network->biasActivation,network->layerCount);
	fputs("[BASIC:ACTIVATION]\n",fp);
	for(i=0;i<network->layerCount;i++)
	{
		fputc('\"',fp);
		if( network->activationFunctions[i]==&EncogActivationLinear ) {
			fputs("ActivationLinear",fp);
		} else if( network->activationFunctions[i]==&EncogActivationTANH ) {
			fputs("ActivationTANH",fp);
		} else if( network->activationFunctions[i]==&EncogActivationSigmoid ) {
			fputs("ActivationSigmoid",fp);
		}
		
		fputs("\"\n",fp);
	}

	/* Write the basic info */

	/* Write the activation functions */

	fclose(fp);
}

