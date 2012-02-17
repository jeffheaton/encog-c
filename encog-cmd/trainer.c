#include "encog-cmd.h"

void TrainNetwork(ENCOG_TRAIN_PSO *pso, float maxError,int updateSeconds) 
{
	char line[MAX_STR];
	int iteration;
	float error;
	int done;
	time_t currentTime;
	time_t lastUpdate;
	time_t sinceLastUpdate;

	printf("Beginning training.\n");
	/* Begin training, report progress. */
	done = 0;
    iteration = 1;
	lastUpdate = time(NULL);

    do
    {
        error = EncogTrainPSOIterate(pso);
		EncogErrorCheck();

		currentTime = time(NULL);
		
		sinceLastUpdate = currentTime-lastUpdate;
		//printf(" c:%i\nlu:%i\nsl:%i\n",currentTime,lastUpdate,sinceLastUpdate);

		if( error<maxError ) {
			done = 1;
		}

		/* display every updateSeconds seconds, plus first and last iterations */
		if( done || sinceLastUpdate>=updateSeconds || iteration==1 )
		{
			lastUpdate = time(NULL);
			*line = 0;
			EncogStrCatStr(line,"Iteration #",MAX_STR);
			EncogStrCatInt(line,iteration,MAX_STR);
			EncogStrCatStr(line,", Error: ",MAX_STR);
			EncogStrCatDouble(line,error,4,MAX_STR);
			puts(line);
		}
        iteration++;
    } while(!done);
}
