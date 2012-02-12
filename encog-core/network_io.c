#include "encog.h"

typedef struct {
	char *line;
	INT layerCount;
    INT neuronCount;
    INT weightCount;
    INT inputCount;
    INT *layerCounts;
    INT *layerContextCount;
    INT *layerFeedCounts;
    INT *layerIndex;
    REAL *layerOutput;
    INT outputCount;
    INT *weightIndex;
    REAL *weights;
    ACTIVATION_FUNCTION *activationFunctions;
    REAL *biasActivation;
	INT beginTraining;
	REAL connectionLimit;
	INT *contextTargetOffset;
	INT *contextTargetSize;
	INT endTraining;
	INT hasContext;
	INT totalNetworkSize;

	FILE *fp;
} _PARSED_NETWORK;

typedef struct _ARRAY_SEGMENT {
	double array[2048];
	int length;
	struct _ARRAY_SEGMENT *next;
} _ARRAY_SEGMENT;

static REAL *_ParseLargeDoubleList(_PARSED_NETWORK *parse,char *firstline, INT *size)
{
	INT index, i, len;
	REAL *result;
	char arg[MAX_STR],*ptr;
	REAL *dptr;

	if( firstline[0]=='#' && firstline[1]=='#' ) {
		fgets(parse->line,SIZE_MEGABYTE,parse->fp);
		/* first, strip the length off the end */
		ptr = parse->line+strlen(parse->line)-1;
		while(*(ptr-1)!='#' && ptr>firstline ) {
			ptr--;
		}
		*size = atoi(ptr);

		/* allocate enough space */
		result = (REAL*)malloc((*size)*sizeof(double));
		dptr = result;
		len = 0;

		while( fgets(parse->line,SIZE_MEGABYTE,parse->fp) ) 
		{
			EncogStrStripCRLF(parse->line);

			index = 0;

			if( *parse->line=='#' ) {
				break;
			}
			
			do {
				index = EncogStrPopLine(parse->line, arg, index, sizeof(arg));
				if( *arg || parse->line[index] ) {
					*(dptr++) = (REAL)atof(arg);
					len++;
				}
			} while(parse->line[index] && len<*size);
		}
		
	} else {
	*size = EncogStrCountValues(firstline);		
	result = (REAL*)malloc((*size)*sizeof(double));

	index = 0;
	for(i = 0; i<(*size); i++ )
	{
		index = EncogStrPopLine(firstline, arg, index, sizeof(arg));
		result[i] = (REAL)atof(arg);
	}
	}
	return result;
}


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


static void _LoadBasic(_PARSED_NETWORK *parse)
{
	char name[MAX_STR],*value;

	EncogStrStripCRLF(parse->line);
	value = EncogStrParseNV(parse->line,name,MAX_STR);

	if(!strcmp(name,"beginTraining") )
	{
		parse->beginTraining = atoi(value);
	}
	else if(!strcmp(name,"connectionLimit") )
	{
		parse->connectionLimit = (REAL)atof(value);
	}
	else if(!strcmp(name,"contextTargetOffset") )
	{
		parse->contextTargetOffset = EncogStrParseIntList(value);
	}
	else if(!strcmp(name,"contextTargetSize") )
	{
		parse->contextTargetSize = EncogStrParseIntList(value);
	}
	else if(!strcmp(name,"endTraining") )
	{
		parse->endTraining = atoi(value);
	}
	else if(!strcmp(name,"hasContext") )
	{
		parse->hasContext = EncogStrParseBoolean(value);
	}
	else if(!strcmp(name,"inputCount") )
	{
		parse->inputCount = atoi(value);
	}
	else if(!strcmp(name,"layerCounts") )
	{
		parse->layerCount = EncogStrCountValues(value);
		parse->layerCounts = EncogStrParseIntList(value);
		parse->activationFunctions = (ACTIVATION_FUNCTION*)EncogUtilAlloc(parse->layerCount,sizeof(ACTIVATION_FUNCTION));
	}
	else if(!strcmp(name,"layerFeedCounts") )
	{
		parse->layerFeedCounts = EncogStrParseIntList(value);
	}
	else if(!strcmp(name,"layerContextCount") )
	{
		parse->layerContextCount = EncogStrParseIntList(value);
	}
	else if(!strcmp(name,"layerIndex") )
	{
		parse->layerIndex = EncogStrParseIntList(value);
	}
	else if(!strcmp(name,"output") )
	{
		parse->layerOutput = _ParseLargeDoubleList(parse,value,&parse->neuronCount);
	}
	else if(!strcmp(name,"outputCount") )
	{
		parse->outputCount = atoi(value);
	}
	else if(!strcmp(name,"weightIndex") )
	{
		parse->weightIndex = EncogStrParseIntList(value);
	}
	else if(!strcmp(name,"weights") )
	{
		parse->weights = _ParseLargeDoubleList(parse,value,&parse->weightCount);
	}
	else if(!strcmp(name,"biasActivation") )
	{
		parse->biasActivation = EncogStrParseDoubleList(value);
	}	
}

static void _LoadActivation(char *line, _PARSED_NETWORK *network, int currentActivation)
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
	int mode, currentActivation;
	_PARSED_NETWORK parse;
	ENCOG_NEURAL_NETWORK *result;

	parse.fp = fopen(name,"r");

	if( _CheckNetwork(parse.fp) == -1 )
	{
		return NULL;
	}

	parse.line = (char*)EncogUtilAlloc(SIZE_MEGABYTE,sizeof(char));
	mode = 0;
	currentActivation = 0;

	while( fgets(parse.line,SIZE_MEGABYTE,parse.fp) ) 
	{
		EncogStrStripCRLF(parse.line);

		if(!strcmp(parse.line,"[BASIC:NETWORK]") )
		{
			mode = 1;
		}
		else if(!strcmp(parse.line,"[BASIC:ACTIVATION]") )
		{
			mode = 2;
		}
		else 
		{
			switch(mode)
			{
				case 1:
					_LoadBasic(&parse);
					break;
				case 2:
					_LoadActivation(parse.line,&parse,currentActivation++);
					break;
			}
		}		
	}

	fclose(parse.fp);
	EncogUtilFree(parse.line);

	// calculate total network size
	parse.totalNetworkSize = EncogNetworkDetermineSize(parse.layerCount,parse.neuronCount,parse.weightCount);

	// now actually build the neural network from the parsed structure
	result = (ENCOG_NEURAL_NETWORK*)EncogUtilAlloc(1,parse.totalNetworkSize);
	result->beginTraining = parse.beginTraining;
	result->connectionLimit = parse.connectionLimit;
	result->endTraining = parse.endTraining;
	result->hasContext = parse.hasContext;
	result->inputCount = parse.inputCount;
	result->layerCount = parse.layerCount;
	result->neuronCount = parse.neuronCount;
	result->outputCount = parse.outputCount;
	result->memorySize = parse.totalNetworkSize;
	result->weightCount = parse.weightCount;
	EncogNetworkLink(result);
	
	// copy array values loaded
	memcpy(result->activationFunctions,parse.activationFunctions,parse.layerCount*sizeof(ACTIVATION_FUNCTION));
	memcpy(result->biasActivation,parse.biasActivation,parse.layerCount*sizeof(REAL));
	memcpy(result->contextTargetOffset,parse.contextTargetOffset,parse.layerCount*sizeof(INT));
	memcpy(result->contextTargetSize,parse.contextTargetSize,parse.layerCount*sizeof(INT));
	memcpy(result->layerContextCount,parse.layerContextCount,parse.layerCount*sizeof(INT));
	memcpy(result->layerCounts,parse.layerCounts,parse.layerCount*sizeof(INT));
	memcpy(result->layerFeedCounts,parse.layerFeedCounts,parse.layerCount*sizeof(INT));
	memcpy(result->layerIndex,parse.layerIndex,parse.layerCount*sizeof(INT));
	memcpy(result->layerOutput,parse.layerOutput,parse.neuronCount*sizeof(REAL));
	result->layerSums = (REAL*)EncogUtilAlloc(parse.neuronCount,sizeof(REAL));
	memcpy(result->weightIndex,parse.weightIndex,parse.layerCount*sizeof(INT));
	memcpy(result->weights,parse.weights,parse.weightCount*sizeof(REAL));

	// clean up the parsed structure and de-alloc
	EncogUtilFree(parse.activationFunctions);
	EncogUtilFree(parse.biasActivation);
	EncogUtilFree(parse.contextTargetOffset);
	EncogUtilFree(parse.contextTargetSize);
	EncogUtilFree(parse.layerContextCount);
	EncogUtilFree(parse.layerCounts);
	EncogUtilFree(parse.layerFeedCounts);
	EncogUtilFree(parse.layerIndex);
	EncogUtilFree(parse.layerOutput);
	EncogUtilFree(parse.weightIndex);
	EncogUtilFree(parse.weights);

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
	if( fp==NULL ) {
		EncogErrorSet(ENCOG_ERROR_FILE_NOT_FOUND);
		return;
	}

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
	EncogStrCatLong(line,(long)t,MAX_STR);
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

