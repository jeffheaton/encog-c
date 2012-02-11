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

static const char *HEADER = "ENCOG-00";

ENCOG_DATA *EncogDataCreate(unsigned int inputCount, unsigned int idealCount, unsigned long records)
{
	ENCOG_DATA *data;
	int totalSize;

	/* Clear out any previous errors */
	EncogErrorClear();

	totalSize = sizeof(ENCOG_DATA) + ((records*(inputCount+idealCount+1)) * sizeof(REAL));
    data = (ENCOG_DATA*)EncogUtilAlloc(1,totalSize);
	data->memorySize = totalSize;
    data->inputCount = inputCount;
    data->idealCount = idealCount;
    data->recordCount = records;
    data->data = (REAL*)(((char*)data)+sizeof(ENCOG_DATA));
    data->cursor = data->data;
	assert( (((char*)data->data)-((char*)data))==sizeof(ENCOG_DATA));
	return data;
}

void EncogDataDelete(ENCOG_DATA *data)
{
	/* Clear out any previous errors */
	EncogErrorClear();

    EncogUtilFree(data);
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
            d = atof(temp);
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
        d = atof(temp);
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
        d = va_arg(arguments,double);
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
		return NULL;
	}

	fseek(fp,0,SEEK_END);
	s = ftell(fp);
	fseek(fp,0,SEEK_SET);
	fread(&header,sizeof(EGB_HEADER),1,fp);
	if( memcmp(HEADER,header.ident,8) )
	{
		EncogErrorSet( ENCOG_ERROR_INVALID_EGB_FILE );
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
		fread(record,sizeof(double),recordSize,fp);
		for(j=0;j<result->inputCount;j++) {
			d = record[k++];
			*(ptr++) = d;
		}
		for(j=0;j<result->idealCount;j++) {
			d = record[k++];
			*(ptr++) = d;
		}
	}

	EncogUtilFree(record);

	fclose(fp);
	return result;
}

ENCOG_DATA *EncogDataCSVLoad(char *csvFile, INT inputCount, INT idealCount)
{
	FILE *fp;
	char line [ MAX_STR ], lastNumber[MAX_STR];
	int records,lineCount,lineSize;
	ENCOG_DATA *result;
	char *ptr, *nptr;
	double *optr;

	lineSize = inputCount+idealCount;

	/* Clear out any previous errors */
	EncogErrorClear();

	/* Open the file */
	fp = fopen ( csvFile, "r" );

	if( fp==NULL ) {
		EncogErrorSet(ENCOG_ERROR_FILE_NOT_FOUND);
		return NULL;
	}
	
	/* initially count the lines (pass 1) */
	records = 0;
	while ( fgets ( line, sizeof line, fp ) != NULL ) 
	{
		records++;
	}

	/* allocate space to hold data */
	result = EncogDataCreate(inputCount,idealCount,records);
	if( EncogErrorGet() ) {
		return NULL;
	}

	/* return to beginning and parse the file (pass 2) */
	fseek(fp,0,SEEK_SET);
	optr = result->data;

	while ( fgets ( line, sizeof line, fp ) != NULL ) 
	{
		records++;
		nptr  = lastNumber;
		ptr = line;
		*nptr = 0;
		lineCount = 0;
		
		while(*ptr) {
			if( *ptr==',' ) {
				*(optr++) = atof(lastNumber);
				/* too much data for the line? */
				if( lineCount++>lineSize ) {
					EncogDataDelete(result);
					EncogErrorSet(ENCOG_ERROR_SIZE_MISMATCH);					
					return NULL;
				}
				nptr = lastNumber;
				*nptr = 0;
			} else {
				*(nptr++)=*ptr;
				*nptr = 0;
			}

			ptr++;
		}

		/* too much or to little data for the line? */
		if( ++lineCount!=lineSize ) {
			EncogDataDelete(result);
			EncogErrorSet(ENCOG_ERROR_SIZE_MISMATCH);			
			return NULL;
		}

		*(optr++) = atof(lastNumber);

		/* Skip significance */
		*(optr++) = 1.0;
	}

	fclose(fp);

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

