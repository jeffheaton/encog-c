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

typedef struct {
	char *line;
	ENCOG_NEURAL_NETWORK *network;
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
		if( fgets(parse->line,SIZE_MEGABYTE,parse->fp) == NULL ) {
			*(parse->line) = 0;
		}

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

	if( fgets(line,sizeof(line),fp)==NULL ) {
		*line = 0;
	}

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
		parse->network->beginTraining = atoi(value);
	}
	else if(!strcmp(name,"connectionLimit") )
	{
		parse->network->connectionLimit = (REAL)atof(value);
	}
	else if(!strcmp(name,"contextTargetOffset") )
	{
		parse->network->contextTargetOffset = EncogStrParseIntList(value);
	}
	else if(!strcmp(name,"contextTargetSize") )
	{
		parse->network->contextTargetSize = EncogStrParseIntList(value);
	}
	else if(!strcmp(name,"endTraining") )
	{
		parse->network->endTraining = atoi(value);
	}
	else if(!strcmp(name,"hasContext") )
	{
		parse->network->hasContext = EncogStrParseBoolean(value);
	}
	else if(!strcmp(name,"inputCount") )
	{
		parse->network->inputCount = atoi(value);
	}
	else if(!strcmp(name,"layerCounts") )
	{
		parse->network->layerCount = EncogStrCountValues(value);
		parse->network->layerCounts = EncogStrParseIntList(value);
		parse->network->activationFunctions = (ACTIVATION_FUNCTION*)EncogUtilAlloc(parse->network->layerCount,sizeof(ACTIVATION_FUNCTION));
		parse->network->derivativeFunctions = (DERIVATIVE_FUNCTION*)EncogUtilAlloc(parse->network->layerCount,sizeof(DERIVATIVE_FUNCTION));
		parse->network->activationFunctionIDs = (INT*)EncogUtilAlloc(parse->network->layerCount,sizeof(INT));
	}
	else if(!strcmp(name,"layerFeedCounts") )
	{
		parse->network->layerFeedCounts = EncogStrParseIntList(value);
	}
	else if(!strcmp(name,"layerContextCount") )
	{
		parse->network->layerContextCount = EncogStrParseIntList(value);
	}
	else if(!strcmp(name,"layerIndex") )
	{
		parse->network->layerIndex = EncogStrParseIntList(value);
	}
	else if(!strcmp(name,"output") )
	{
		parse->network->layerOutput = _ParseLargeDoubleList(parse,value,&parse->network->neuronCount);
	}
	else if(!strcmp(name,"outputCount") )
	{
		parse->network->outputCount = atoi(value);
	}
	else if(!strcmp(name,"weightIndex") )
	{
		parse->network->weightIndex = EncogStrParseIntList(value);
	}
	else if(!strcmp(name,"weights") )
	{
		parse->network->weights = _ParseLargeDoubleList(parse,value,&parse->network->weightCount);
	}
	else if(!strcmp(name,"biasActivation") )
	{
		parse->network->biasActivation = EncogStrParseDoubleList(value);
	}	
}

static void _LoadActivation(char *line, _PARSED_NETWORK *parse, int currentActivation)
{
	EncogStrStripQuotes(line);
	if( !strcmp(line,"ActivationLinear") )
	{
		parse->network->activationFunctions[currentActivation] = &EncogActivationLinear;
		parse->network->derivativeFunctions[currentActivation] = &EncogDerivativeLinear;
		parse->network->activationFunctionIDs[currentActivation] = AF_LINEAR;
	}
	else if( !strcmp(line,"ActivationSigmoid") )
	{
		parse->network->activationFunctions[currentActivation] = &EncogActivationSigmoid;
		parse->network->derivativeFunctions[currentActivation] = &EncogDerivativeSigmoid;
		parse->network->activationFunctionIDs[currentActivation] = AF_SIGMOID;
	}
	else if( !strcmp(line,"ActivationTANH") )
	{
		parse->network->activationFunctions[currentActivation] = EncogActivationTANH;
		parse->network->derivativeFunctions[currentActivation] = &EncogDerivativeTANH;
		parse->network->activationFunctionIDs[currentActivation] = AF_TANH;
	}
}


ENCOG_NEURAL_NETWORK *EncogNetworkLoad(char *name)
{
	int mode, currentActivation;
	_PARSED_NETWORK parse;

	parse.fp = fopen(name,"r");

	if( parse.fp==NULL )
	{
		EncogErrorSet(ENCOG_ERROR_FILE_NOT_FOUND);
		EncogErrorSetArg(name);
		return NULL;
	}

	if( _CheckNetwork(parse.fp) == -1 )
	{
		return NULL;
	}

	parse.line = (char*)EncogUtilAlloc(SIZE_MEGABYTE,sizeof(char));
	parse.network = EncogNetworkNew();
	
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

	parse.network->layerSums = (REAL*)EncogUtilAlloc(parse.network->neuronCount,sizeof(REAL));

	return parse.network;
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

