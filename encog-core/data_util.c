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

REAL EncogDataVariancePerIndex(const ENCOG_DATA *data,const int index,const REAL mean)
{	
	unsigned long i;
    REAL variance = 0.0;
	for (i = 0; i < data->recordCount; i++)
    {
        const REAL delta = (data->data[i*(data->inputCount+data->idealCount)+index] - mean);
        variance += (delta * delta - variance) / (i + 1);
    }
	return variance;
}
REAL EncogDataStdevPerIndex(const ENCOG_DATA *data,const int index,const REAL mean)
{	
    return sqrt (EncogDataVariancePerIndex(data,index,mean) * ((REAL)data->recordCount / (REAL)(data->recordCount - 1)));
}
REAL EncogDataMeanPerIndex(const ENCOG_DATA *data,const int index)
{
	unsigned long i;
    REAL mean = 0.0;
    for (i = 0;i<data->recordCount;i++)
    {
        mean += (data->data[i*(data->inputCount+data->idealCount)+index] - mean) / (i + 1);
    }
    return mean;
}
ENCOG_DATA_NORM *EncogDataNormalise(ENCOG_DATA *data){
	int i;
	unsigned long j;
	int data_count=data->inputCount+data->idealCount;

	ENCOG_DATA_NORM *normalise_params=(ENCOG_DATA_NORM*)EncogUtilAlloc(data_count,sizeof(ENCOG_DATA_NORM));

	for(i=0; i<data_count; i++)
    {
		const REAL mean = EncogDataMeanPerIndex(data,i);
		const REAL stdev = EncogDataStdevPerIndex(data,i,mean);

		normalise_params[i].mean=mean;
		normalise_params[i].stdev=stdev;

        for(j=0; j<data->recordCount; j++)
        {
			unsigned long index=j*data_count+i;
			data->data[index]= (data->data[index]-mean)/stdev;
		}
	}

	return normalise_params;
}
void EncogDataDenormalise(ENCOG_DATA *data,ENCOG_DATA_NORM *normalise_params)
{
	int i;
	unsigned long j;
    int data_count=data->inputCount+data->idealCount;

	for(i=0; i<data_count; i++)
    {
		const REAL mean = normalise_params[i].mean;
		const REAL stdev = normalise_params[i].stdev;

        for(j=0; j<data->recordCount; j++)
        {
			unsigned long index=j*data_count+i;
			data->data[index]= (data->data[index]*stdev)+mean;
		}
	}
}