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

static const char *HEADER = "ENCOG-00";

ENCOG_DATA *EncogDataCreate(unsigned int inputCount, unsigned int idealCount, unsigned long records)
{
	ENCOG_DATA *data;

	/* Clear out any previous errors */
	EncogErrorClear();

    data = (ENCOG_DATA*)EncogUtilAlloc(1,sizeof(ENCOG_DATA));
    data->inputCount = inputCount;
    data->idealCount = idealCount;
    data->recordCount = records;
    data->data = (REAL*)EncogUtilAlloc(records*(inputCount+idealCount+1),sizeof(REAL));
    data->cursor = data->data;  

	EncogObjectRegister(data, ENCOG_TYPE_DATA);

	return data;
}

void EncogDataAdd(ENCOG_DATA *data,char *str)
{
    char ch, *ptr;
    char temp[MAX_STR];
    REAL d;

	/* Clear out any previous errors */
	EncogErrorClear();

    *temp = 0;
    ptr = str;

    while(*ptr)
    {
        ch = *ptr;
        if( ch==',' )
        {
            d = (REAL)atof(temp);
            *(data->cursor++) = d;
            *temp = 0;
        }
        else if( isdigit((int)ch) || ch=='-' || ch=='.' )
        {
            EncogStrCatChar(temp,ch,MAX_STR);
        }
        ptr++;
    }

    if(*temp)
    {
        d = (REAL)atof(temp);
        *(data->cursor++) = d;
    }
}

void EncogDataAddVar(ENCOG_DATA *data, ...)
{
    int i,total;
    REAL d = 0.0;
	va_list arguments;

	/* Clear out any previous errors */
	EncogErrorClear();
    
    va_start ( arguments, data );
    total = data->inputCount + data->idealCount;

    for(i=0; i<total; i++)
    {
        d = (REAL)va_arg(arguments,double);
        *(data->cursor++) = d;
    }

    va_end( arguments );
}

REAL *EncogDataGetInput(ENCOG_DATA *data, unsigned int index)
{
    int i;

	/* Clear out any previous errors */
	EncogErrorClear();
	
	i = index*(data->inputCount+data->idealCount);
    return &data->data[i];
}

REAL *EncogDataGetIdeal(ENCOG_DATA *data, unsigned int index)
{
    int i;

	/* Clear out any previous errors */
	EncogErrorClear();
	
	i = index*(data->inputCount+data->idealCount);
    return &data->data[i+data->inputCount];
}

void EncogDataCSVSave(char *filename, ENCOG_DATA *data, int decimals)
{
    char temp[MAX_STR];
    INT i,j;
    REAL *input, *ideal;
    FILE *fp;

	/* Clear out any previous errors */
	EncogErrorClear();

    fp = fopen(filename,"w");

	if( fp==NULL ) {
		EncogErrorSet(ENCOG_ERROR_FILE_NOT_FOUND);
		EncogErrorSetArg(filename);
		return;
	}


    for(i=0; i<data->recordCount; i++)
    {
        input = EncogDataGetInput(data,i);
        ideal = EncogDataGetIdeal(data,i);

        for(j=0; j<data->inputCount; j++)
        {
            if(j>0)
            {
                fprintf(fp,",");
            }
            *temp=0;
            EncogStrCatDouble(temp,input[j],decimals,MAX_STR);
			fputs(temp,fp);
        }

        for(j=0; j<data->idealCount; j++)
        {
            fprintf(fp,",");
			*temp = 0;
            EncogStrCatDouble(temp,ideal[j],decimals,MAX_STR);
			fputs(temp,fp);
        }
        fputs("\n",fp);
    }
    fclose(fp);
}

ENCOG_DATA *EncogDataGenerateRandom(INT inputCount, INT idealCount, INT records, REAL low, REAL high)
{
	ENCOG_DATA *data;
	REAL *ptr;
	int i,size;

	/* Clear out any previous errors */
	EncogErrorClear();

	data = EncogDataCreate(inputCount,idealCount,records);
	size = (inputCount+idealCount)*records;
	
	ptr = data->data;
	for(i=0;i<size;i++) {
		*(ptr++) = EncogUtilRandomRange(low,high);		
	}

	return data;
}

ENCOG_DATA *EncogDataEGBLoad(char *f)
{
	REAL *ptr;
	double *record,d;
	INT records,i,j,k,recordSize;
	EGB_HEADER header;
	ENCOG_DATA *result;
	FILE *fp;
	long s;

	/* Clear out any previous errors */
	EncogErrorClear();

	fp = fopen(f,"rb");

	if( fp==NULL ) {
		EncogErrorSet(ENCOG_ERROR_FILE_NOT_FOUND);
		EncogErrorSetArg(f);
		return NULL;
	}

	fseek(fp,0,SEEK_END);
	s = ftell(fp);
	fseek(fp,0,SEEK_SET);

	if( (fread(&header,sizeof(EGB_HEADER),1,fp) != 1) || memcmp(HEADER,header.ident,8) ) {
		EncogErrorSet( ENCOG_ERROR_INVALID_EGB_FILE );
		EncogErrorSetArg(f);
		fclose(fp);
		return NULL;
	}

	recordSize = (INT)(header.input+header.ideal+1);
	records = (s-sizeof(EGB_HEADER))/(recordSize*sizeof(double));
	result = EncogDataCreate((INT)header.input,(INT)header.ideal,records);
	record = (double*)EncogUtilAlloc(recordSize,sizeof(double));

	/* read in data, the size of REAL may not match the doubles written to the file */
	ptr = result->data;
	for(i=0;i<records;i++) {
		k=0;

		if( fread(record,sizeof(double),recordSize,fp) != recordSize ) {
			EncogErrorSet( ENCOG_ERROR_INVALID_EGB_FILE );
			EncogErrorSetArg(f);
			fclose(fp);
			return NULL;		
		}

		for(j=0;j<result->inputCount;j++) {
			d = record[k++];
			*(ptr++) = (REAL)d;
		}
		for(j=0;j<result->idealCount;j++) {
			d = record[k++];
			*(ptr++) = (REAL)d;
		}
	}

	EncogUtilFree(record);

	fclose(fp);
	return result;
}

ENCOG_DATA *EncogDataCSVLoad(char *csvFile, INT inputCount, INT idealCount)
{
	FILE *fp;
	char numBuffer[ MAX_STR ], ch;
	int records,lineCount,lineSize;
	ENCOG_DATA *result;
	REAL *optr;
	int lastCR;
	int lastError;
	int currentLine;

	lineSize = inputCount+idealCount;

	/* Clear out any previous errors */
	EncogErrorClear();

	/* Open the file */
	fp = fopen ( csvFile, "rb" );

	if( fp==NULL ) {
		EncogErrorSet(ENCOG_ERROR_FILE_NOT_FOUND);
		EncogErrorSetArg(csvFile);
		return NULL;
	}

	lastError = 0;
	
	/* initially count the lines (pass 1) */
	records = 0;
	lastCR = 0;
	while ( (ch=fgetc(fp)) !=EOF ) 
	{
		if( ch==13 || (ch==10&&!lastCR) ) {
			records++;
		}  

		lastCR = (ch==13);
	}

	/* allocate space to hold data */
	result = EncogDataCreate(inputCount,idealCount,records);
	if( EncogErrorGet() ) {
		return NULL;
	}

	/* return to beginning and parse the file (pass 2) */
	fseek(fp,0,SEEK_SET);
	optr = result->data;


	lastCR = 0;
	*numBuffer = 0;
	lineCount = 0;

	currentLine = 1;
	while ( (ch=fgetc(fp)) !=EOF ) 
	{
		if( ch==13 || (ch==10&&!lastCR) ) {
			if( lineCount ) {
				*(optr++) = (REAL)atof(numBuffer);
				*numBuffer = 0;
				lineCount++;

				/* too much data for the line? */
				if( ( lineCount>lineSize ) || ( lineCount<lineSize ) ) {
					lastError = ENCOG_ERROR_SIZE_MISMATCH;
					break;
				}
		
				lineCount = 0;
				currentLine++;
			}
		} if( ch==',' ) {
			/* too much data for the line? */
			if( lineCount>=lineSize ) {
				lastError = ENCOG_ERROR_SIZE_MISMATCH;
				break;
			}

			*(optr++) = (REAL)atof(numBuffer);
			*numBuffer = 0;
			lineCount++;
		}
		else {
			EncogStrCatChar(numBuffer, ch, sizeof(numBuffer));
		}

		lastCR = (ch==13);
	}

	fclose(fp);

	if( lastError ) {
		*numBuffer = 0;
		EncogStrCatStr(numBuffer,"While processling line #",sizeof(numBuffer));
		EncogStrCatInt(numBuffer,currentLine,sizeof(numBuffer));
		EncogStrCatStr(numBuffer,", was expecting ",sizeof(numBuffer));
		EncogStrCatInt(numBuffer,lineSize,sizeof(numBuffer));
		EncogStrCatStr(numBuffer," columns.",sizeof(numBuffer));

		EncogObjectFree(result);
		EncogErrorSet(ENCOG_ERROR_SIZE_MISMATCH);
		EncogErrorSetArg(numBuffer);
		return NULL;
	}


	return result;
}

void EncogDataEGBSave(char *egbFile,ENCOG_DATA *data)
{
	FILE *fp;
	double *line;
	EGB_HEADER hdr;
	INT i,j,lineSize;
	REAL *ptr;

	/* Clear out any previous errors */
	EncogErrorClear();

	/* Open the file */
	if( (fp=fopen(egbFile,"wb"))==NULL ) {
		EncogErrorSet(ENCOG_ERROR_FILE_NOT_FOUND);
		EncogErrorSetArg(egbFile);
		return;
	}

	/* Write the header */
	memcpy(hdr.ident,HEADER,8);
	hdr.ideal = data->idealCount;
	hdr.input = data->inputCount;
	fwrite(&hdr,sizeof(EGB_HEADER),1,fp);

	/* write the data */
	lineSize = data->idealCount + data->inputCount + 1;
	line = (double*)EncogUtilAlloc(lineSize,sizeof(double));
	ptr = data->data;

	for(i=0;i<data->recordCount;i++) {
		for(j=0;j<lineSize;j++) {
			line[j] = ptr[j];
		}
		fwrite(line,sizeof(double),lineSize,fp);
		ptr+=lineSize;
	}
	EncogUtilFree(line);

	fclose(fp);
}

