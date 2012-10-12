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

ENCOG_CONTEXT encogContext;

void EncogInit() {
	memset(&encogContext,0,sizeof(ENCOG_CONTEXT));
	encogContext.config = EncogHashNew(5,1);

#ifdef ENCOG_CUDA
	encogContext.gpuEnabled = 1;
#endif	
	
	encogContext.versionMajor = 0;
	encogContext.versionMajor = 1;
	strncpy(encogContext.version,"1.0",sizeof(encogContext.version));

	EncogUtilInitRandom();
}

void EncogShutdown() {
}

void EncogTrainMinimalCallback(ENCOG_TRAINING_REPORT *report) {
	if( report->error < report->maxError ) {
		report->stopRequested = 1;
	}

	if( report->maxIterations!=0 && (report->iterations>=report->maxIterations) ) {
		report->stopRequested = 1;
	}

	//printf("%i\n",report->iterations);

}

void EncogTrainStandardCallback(ENCOG_TRAINING_REPORT *report) {
	char line[MAX_STR];
	time_t currentTime;
	time_t sinceLastUpdate;

	EncogTrainMinimalCallback(report);

	currentTime = time(NULL);
	sinceLastUpdate = currentTime-report->lastUpdate;

	if( report->iterations==1 ) {
		printf("Beginning training (%s).\n", EncogObjectType(report->trainer));
	}
	
	/* display every updateSeconds seconds, plus first and last iterations */
	if( report->stopRequested || sinceLastUpdate>=report->updateSeconds || report->iterations==1 )
	{
		report->lastUpdate = time(NULL);
		*line = 0;
		EncogStrCatStr(line,"Iteration #",MAX_STR);
		EncogStrCatInt(line,report->iterations,MAX_STR);
		EncogStrCatStr(line,", Error: ",MAX_STR);
		EncogStrCatDouble(line,report->error*100.0,4,MAX_STR);
		EncogStrCatChar(line,'%',MAX_STR);
		EncogStrCatStr(line,", press [Enter] to quit and save.", MAX_STR);
		puts(line);

		if( kbhit() )
		{
			puts("Stopping at user request.");				
			report->stopRequested = 1;	
		}
	}

	if( report->stopRequested ) {
		printf("Training complete.\n");
	}

}

