#include "encog.h"

int main(int argc, char* argv[])
{
    /* local variables */
    char line[MAX_STR];
    int i, iteration;
    REAL *input,*ideal;
    REAL output[1];
    float error;
    ENCOG_DATA *data;
    ENCOG_NEURAL_NETWORK *net;
    ENCOG_TRAIN_PSO *pso;

/* Seed the random number generator */
    EncogUtilInitRandom();

/* Load the data for XOR */
    EncogDataCreate(&data, 2, 1, 4);
    EncogDataAdd(data,"0,0,  0");
    EncogDataAdd(data,"1,0,  1");
    EncogDataAdd(data,"0,1,  1");
    EncogDataAdd(data,"1,1,  0");

/* Create a 3 layer neural network, with sigmoid transfer functions and bias */

    net = EncogNetworkNew();
    EncogNetworkAddLayer(net,2,&EncogActivationSigmoid,1);
    EncogNetworkAddLayer(net,3,&EncogActivationSigmoid,1);
    EncogNetworkAddLayer(net,1,&EncogActivationSigmoid,0);
    EncogNetworkFinalizeStructure(net);

/* Randomize the neural network weights */
    EncogNetworkRandomizeRange(net,-1,1);

/* Create a PSO trainer */
    pso = EncogTrainPSONew(30, net, data);

/* Begin training, report progress. */
    iteration = 1;
    do
    {
        error = EncogTrainPSOIterate(pso);
        *line = 0;
        EncogStrCatStr(line,"Iteration #",MAX_STR);
        EncogStrCatInt(line,iteration,MAX_STR);
        EncogStrCatStr(line,", Error: ",MAX_STR);
        EncogStrCatDouble(line,error,4,MAX_STR);
        puts(line);
        iteration++;
    }
    while(error>0.01);

/* Pull the best neural network that the PSO found */
    EncogTrainPSOImportBest(pso,net);
    EncogTrainPSODelete(pso);

/* Display the results from the neural network, see if it learned anything */
    printf("\nResults:\n");
    for(i=0; i<4; i++)
    {
        input = EncogDataGetInput(data,i);
        ideal = EncogDataGetIdeal(data,i);
        EncogNetworkCompute(net,input,output);
        *line = 0;
        EncogStrCatStr(line,"[",MAX_STR);
        EncogStrCatDouble(line,input[0],8,MAX_STR);
        EncogStrCatStr(line," ",MAX_STR);
        EncogStrCatDouble(line,input[1],8,MAX_STR);
        EncogStrCatStr(line,"] = ",MAX_STR);
        EncogStrCatDouble(line,output[0],8,MAX_STR);
        puts(line);
    }

/* Obtain the SSE error, display it */
    error = EncogErrorSSE(net, data);
    *line = 0;
    EncogStrCatStr(line,"Error: ",MAX_STR);
    EncogStrCatDouble(line,(double)error,4,MAX_STR);
    puts(line);

/* Delete the neural network */
    EncogNetworkDelete(net);
    return 0;
}
