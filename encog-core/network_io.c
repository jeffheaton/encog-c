#include "encog.h"

typedef struct {
	char *line;
	ENCOG_NEURAL_NETWORK *result;
	FILE *fp;
} _PARSED_NETWORK;

typedef struct _ARRAY_SEGMENT {
	double array[2048];
	int length;
	struct _ARRAY_SEGMENT *next;
} _ARRAY_SEGMENT;

static double *_ParseLargeDoubleList(_PARSED_NETWORK *parse,char *firstline, INT *size)
{
	int index, i, len;
	double *result;
	char arg[MAX_STR],*ptr;
	double *dptr;

	if( firstline[0]=='#' && firstline[1]=='#' ) {
		fgets(parse->line,SIZE_MEGABYTE,parse->fp);
		/* first, strip the length off the end */
		ptr = firstline+strlen(parse->line)-1;
		while(*(ptr-1)!='#' && ptr>firstline ) {
			ptr--;
		}
		*size = atoi(ptr);

		/* allocate enough space */
		result = (double*)malloc((*size)*sizeof(double));
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
				*(dptr++) = atof(arg);
				len++;
			} while(*arg && len<*size);
		}
	} else {
	*size = EncogStrCountValues(firstline);		
	result = (double*)malloc((*size)*sizeof(double));

	index = 0;
	for(i = 0; i<(*size); i++ )
	{
		index = EncogStrPopLine(firstline, arg, index, sizeof(arg));
		result[i] = atof(arg);
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
		parse->result->beginTraining = atoi(value);
	}
	else if(!strcmp(name,"connectionLimit") )
	{
		parse->result->connectionLimit = atof(value);
	}
	else if(!strcmp(name,"contextTargetOffset") )
	{
		parse->result->contextTargetOffset = EncogStrParseIntList(value);
	}
	else if(!strcmp(name,"contextTargetSize") )
	{
		parse->result->contextTargetSize = EncogStrParseIntList(value);
	}
	else if(!strcmp(name,"endTraining") )
	{
		parse->result->endTraining = atoi(value);
	}
	else if(!strcmp(name,"hasContext") )
	{
		parse->result->hasContext = EncogStrParseBoolean(value);
	}
	else if(!strcmp(name,"inputCount") )
	{
		parse->result->inputCount = atoi(value);
	}
	else if(!strcmp(name,"layerCounts") )
	{
		parse->result->layerCount = EncogStrCountValues(value);
		parse->result->layerCounts = EncogStrParseIntList(value);
		parse->result->activationFunctions = (ACTIVATION_FUNCTION*)EncogUtilAlloc(parse->result->layerCount,sizeof(ACTIVATION_FUNCTION));
	}
	else if(!strcmp(name,"layerFeedCounts") )
	{
		parse->result->layerFeedCounts = EncogStrParseIntList(value);
	}
	else if(!strcmp(name,"layerContextCount") )
	{
		parse->result->layerContextCount = EncogStrParseIntList(value);
	}
	else if(!strcmp(name,"layerIndex") )
	{
		parse->result->layerIndex = EncogStrParseIntList(value);
	}
	else if(!strcmp(name,"output") )
	{
		parse->result->layerOutput = _ParseLargeDoubleList(parse,value,&parse->result->neuronCount);
	}
	else if(!strcmp(name,"outputCount") )
	{
		parse->result->outputCount = atoi(value);
	}
	else if(!strcmp(name,"weightIndex") )
	{
		parse->result->weightIndex = EncogStrParseIntList(value);
	}
	else if(!strcmp(name,"weights") )
	{
		parse->result->weights = _ParseLargeDoubleList(parse,value,&parse->result->weightCount);
	}
	else if(!strcmp(name,"biasActivation") )
	{
		parse->result->biasActivation = EncogStrParseDoubleList(value);
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
	int mode, currentActivation;
	_PARSED_NETWORK parse;

	parse.fp = fopen(name,"r");

	if( _CheckNetwork(parse.fp) == -1 )
	{
		return NULL;
	}

	parse.line = (char*)EncogUtilAlloc(SIZE_MEGABYTE,sizeof(char));
	parse.result = (ENCOG_NEURAL_NETWORK *)EncogUtilAlloc(1,sizeof(ENCOG_NEURAL_NETWORK));
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
					_LoadActivation(parse.line,parse.result,currentActivation++);
					break;
			}
		}		
	}

	fclose(parse.fp);
	EncogUtilFree(parse.line);

	parse.result->layerSums = (REAL*)EncogUtilAlloc(parse.result->neuronCount,sizeof(REAL));
	return parse.result;
}

void EncogNetworkSave(char *name, ENCOG_NEURAL_NETWORK *network)
{
	char line[MAX_STR];
	INT i;
	FILE *fp;
	time_t t;

	if( network->weights == NULL || network->firstBlock!=NULL ) {
		EncogErrorSet(ENCOG_ERROR_NETWORK_NOT_FINALIZED);
		return;
	}

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

