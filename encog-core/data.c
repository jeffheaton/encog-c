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

void EncogDataCreate(ENCOG_DATA **data, unsigned int inputCount, unsigned int idealCount, unsigned long records)
{
    *data = (ENCOG_DATA*)EncogUtilAlloc(1,sizeof(ENCOG_DATA));
    (*data)->inputCount = inputCount;
    (*data)->idealCount = idealCount;
    (*data)->recordCount = records;
    (*data)->data = (REAL*)EncogUtilAlloc(records*(inputCount+idealCount),sizeof(REAL));
    (*data)->cursor = (*data)->data;
}

void EncogDataDelete(ENCOG_DATA *data)
{
    EncogUtilFree(data->data);
    EncogUtilFree(data);
}

void EncogDataAdd(ENCOG_DATA *data,char *str)
{
    char ch, *ptr;
    char temp[MAX_STR];
    REAL d;

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
        else if( isdigit(ch) || ch=='-' || ch=='.' )
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
    int i = index*(data->inputCount+data->idealCount);
    return &data->data[i];
}

REAL *EncogDataGetIdeal(ENCOG_DATA *data, unsigned int index)
{
    int i = index*(data->inputCount+data->idealCount);
    return &data->data[i+data->inputCount];
}

void EncogDataSaveCSV(char *filename, ENCOG_DATA *data, int decimals)
{
    char temp[MAX_STR];
    INT i,j;
    REAL *input, *ideal;
    FILE *fp;

    fp = fopen(filename,"w");
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
        }

        for(j=0; j<data->idealCount; j++)
        {
            fprintf(fp,",");
            EncogStrCatDouble(temp,ideal[j],decimals,MAX_STR);
        }
        fputs("\n",fp);
    }
    fclose(fp);
}
