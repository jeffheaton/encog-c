#include "encog-cmd.h"
#include <string.h>

char parsedOption[MAX_STR];
char parsedArgument[MAX_STR];

void PerformTrain(ENCOG_NEURAL_NETWORK *net, ENCOG_DATA *data);

void Usage() {
	puts("\nUsage:\n");
	puts("encog xor");
	puts("encog benchmark");
	puts("encog train [eg file] [egb file]");
	puts("encog egb2csv [egb file] [csv file]");
	puts("encog csv2egb [csv file] [egb file]");
	puts("");
	puts("Options:");
	puts("");
	puts("/input:## The number of inputs.");
	puts("/ideal:## The number of ideals.");
	puts("/records:## The number of ideals.");
	puts("/iterations:## The number of ideals.");
	puts("/threads:## The number of threads.");
	puts("");
}

void cudaNotCompiled() {
	printf("CUDA is not available in this distribution od Encog\n");
		printf("If you wish to use CUDA, please download a CUDA enabled version of Encog.\n");
		exit(0);
}

void ParseOption(char *str)
{
	char *ptr;
	int l;

	if( *str=='/' || *str=='-' ) {
		str++;
	}

	ptr = strchr(str,':');

	if( ptr!=NULL ) {
		l=ptr-str;
		strncpy(parsedOption,str,MIN(MAX_STR,l));
		*(parsedOption+l)=0;
		strncpy(parsedArgument,ptr+1,MAX_STR);
	} else {
		strncpy(parsedOption,str,MAX_STR);
		*parsedArgument = 0;
	}
}

void displayStats(ENCOG_TRAIN_PSO *pso) {
#ifdef ENCOG_CUDA
	if( encogContext.gpuEnabled ) {
		printf("CUDA Stats: avg work unit time = %f ms, work unit calls = %i\n", pso->cudaKernelTime, pso->cudaKernelCalls);
	} else {
		puts("CUDA Stats: GPU disabled\n");
	}
#endif
	printf("CPU Stats: avg work unit time = %f ms, work unit calls = %i\n", pso->cpuWorkUnitTime, pso->cpuWorkUnitCalls);

}

void RunBenchmark() {
	ENCOG_DATA *data;
	ENCOG_NEURAL_NETWORK *net;
	ENCOG_OBJECT *train;
	NETWORK_LAYER *layer;
	INT i;
	double startTime, endTime, elapsed;
	ENCOG_TRAINING_REPORT *report;
	INT idealCount, inputCount, records, iterations;

	inputCount = EncogHashGetInteger(encogContext.config, PARAM_INPUT, 10);
	idealCount = EncogHashGetInteger(encogContext.config, PARAM_IDEAL, 1);
	records = EncogHashGetInteger(encogContext.config, PARAM_RECORDS, 10000);
	iterations = EncogHashGetInteger(encogContext.config, PARAM_INPUT, 100);

	printf("\nPerforming benchmark\n");
	printf("Input Count: %i\n",inputCount);
	printf("Ideal Count: %i\n",idealCount);
	//printf("Particle Count: %i\n", _particles);
	printf("Records: %i\n",records);
	printf("Iterations: %i\n",iterations);

	data = EncogDataGenerateRandom(inputCount,idealCount,records,-1,1);

	net = EncogNetworkNew();
	EncogErrorCheck();
    layer = EncogNetworkCreateLayer(NULL,inputCount,AF_TANH,1);
    layer = EncogNetworkCreateLayer(layer,50,AF_TANH,1);
    layer = EncogNetworkCreateLayer(layer,idealCount,AF_TANH,1);
    net = EncogNetworkFinalizeStructure(layer,1);
	EncogErrorCheck();

	EncogNetworkRandomizeRange(net,-1,1);

	train = EncogTrainNew(net, data);
	EncogErrorCheck();

	report = EncogTrainReport(train);

	startTime = omp_get_wtime();

	report->maxError = 0.00f;
	report->maxIterations = iterations;
	report->updateSeconds = 0;
	
	EncogTrainSetCallback(train, EncogTrainMinimalCallback);
	EncogErrorCheck();

    EncogTrainRun(train, net);
	EncogErrorCheck();

	endTime = omp_get_wtime();
	
	elapsed = endTime - startTime;

	printf("Benchmark time(seconds): %.4f\nBenchmark time includes only training time.\n\n",(float)elapsed);
	//displayStats(train);
}

void XORTest() {
	    /* local variables */
    char line[MAX_STR];
    int i;
    REAL *input,*ideal;
    REAL output[1];
    float error;
    ENCOG_DATA *data;
    ENCOG_NEURAL_NETWORK *net;
    ENCOG_TRAIN_PSO *pso;

/* Load the data for XOR */
    data = EncogDataCreate(2, 1, 4);
	EncogErrorCheck();
    EncogDataAdd(data,"0,0,  0");
    EncogDataAdd(data,"1,0,  1");
    EncogDataAdd(data,"0,1,  1");
    EncogDataAdd(data,"1,1,  0");

/* Create a 3 layer neural network, with sigmoid transfer functions and bias */

    net = EncogNetworkFactory("basic", "2:B->SIGMOID->2:B->SIGMOID->1", 0,0);
	EncogErrorCheck();

	PerformTrain(net, data, 0);

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
    EncogObjectFree(net);
	EncogErrorCheck();

}

void train(char *egFile, char *egbFile ) {
	ENCOG_DATA *data;
	ENCOG_NEURAL_NETWORK *net;
	INT iterations;

	if( *egFile==0 || *egbFile==0 ) {
		printf("Usage: train [egFile] [egbFile]\n");
		return;
	}

	iterations = EncogHashGetInteger(encogContext.config, PARAM_ITERATIONS, 0);

	data = EncogDataEGBLoad(egbFile);
	EncogErrorCheck();

	printf("Training\n");
	printf("Input Count: %i\n", data->inputCount);
	printf("Ideal Count: %i\n", data->idealCount);
	printf("Record Count: %ld\n", data->recordCount);	    
	//printf("Particles: %i\n", _particles);	    

	net = EncogNetworkLoad(egFile);
	EncogErrorCheck();

	if( data->inputCount != net->inputCount ) {
		EncogObjectFree(net);
		EncogErrorCheck();
		printf("Error: The network has a different input count than the training data.\n");
		return;
	}

	if( data->idealCount != net->outputCount ) {
		EncogObjectFree(net);
		EncogErrorCheck();
		printf("Error: The network has a different output count than the training data.\n");
		return;
	}

	PerformTrain(net, data, iterations);

	EncogNetworkSave(egFile,net);
	EncogErrorCheck();

	EncogObjectFree(net);
	EncogErrorCheck();

}

void PerformTrain(ENCOG_NEURAL_NETWORK *net, ENCOG_DATA *data, int iterations) 
{
	ENCOG_OBJECT *trainer;
	ENCOG_TRAINING_REPORT *report;

	trainer = EncogTrainNew(net,data);
	EncogErrorCheck();

	report = EncogTrainReport(trainer);
	EncogErrorCheck();

/* Begin training, report progress. */	
	report->maxError = 0.00f;
	report->maxIterations = iterations;
	report->updateSeconds = 1;
	report->maxError = 0.01;

	EncogTrainRun(trainer,net);

}

void EGB2CSV(char *egbFile, char *csvFile) 
{
	ENCOG_DATA *data;

	data = EncogDataEGBLoad(egbFile);
	EncogErrorCheck();

	printf("Converting EGB to CSV\n");
	printf("Input Count: %i\n", data->inputCount);
	printf("Ideal Count: %i\n", data->idealCount);
	printf("Record Count: %ld\n", data->recordCount);
	printf("Source File: %s\n", egbFile );
	printf("Target File: %s\n", csvFile );

	EncogDataCSVSave(csvFile,data,10);
	EncogErrorCheck();
	EncogObjectFree(data);
	EncogErrorCheck();

	printf("Conversion done.\n");
}

void CSV2EGB(char *csvFile, char *egbFile) 
{
	ENCOG_DATA *data;
	int inputCount, idealCount;

	if( EncogHashGet(encogContext.config,PARAM_INPUT)==NULL 
		|| EncogHashGet(encogContext.config,PARAM_IDEAL)==NULL ) {
		printf("You must specify both input and ideal counts.\n");
		exit(1);
	}

	inputCount = EncogHashGetInteger(encogContext.config,PARAM_INPUT,0);
	idealCount = EncogHashGetInteger(encogContext.config,PARAM_IDEAL,0);

	data = EncogDataCSVLoad(csvFile, inputCount, idealCount);
	EncogErrorCheck();

	printf("Converting CSV to EGB\n");
	printf("Input Count: %i\n", data->inputCount);
	printf("Ideal Count: %i\n", data->idealCount);
	printf("Record Count: %ld\n", data->recordCount);
	printf("Source File: %s\n", csvFile );
	printf("Target File: %s\n", egbFile );

	EncogDataEGBSave(egbFile,data);
	EncogErrorCheck();
	EncogObjectFree(data);
	EncogErrorCheck();

	printf("Conversion done.\n");
}

void CreateNetwork(char *egFile, char *method, char *architecture) {
	ENCOG_NEURAL_NETWORK *network;
	int inputCount, idealCount;

	if( *egFile==0 || *method==0 || *architecture==0 ) {
		printf("Must call with: create [egFile] [method] [architecture]\nNote: Because architecture includes < and > it must be includes in qutoes on most OS's.\n");
		exit(1);
	}

	inputCount = EncogHashGetInteger(encogContext.config,PARAM_INPUT,0);
	idealCount = EncogHashGetInteger(encogContext.config,PARAM_IDEAL,0);

	printf("Creating neural network\n");
	printf("Method: %s\n", method);
	printf("Architecture: %s\n", architecture);

	network = EncogNetworkFactory(method, architecture, inputCount,idealCount);
	EncogErrorCheck();
	printf("Input Count: %i\n",network->inputCount);
	printf("Output Count: %i\n",network->outputCount);

	EncogNetworkSave(egFile,network);
	EncogErrorCheck();
	printf("Network Saved\n");
}

void EvaluateError(char *egFile, char *egbFile) {
	ENCOG_DATA *data;
	ENCOG_NEURAL_NETWORK *net;
	float error;
	char line[MAX_STR];

	if( *egFile==0 || *egbFile==0 ) {
		printf("Usage: error [egFile] [egbFile]\n");
		return;
	}

	data = EncogDataEGBLoad(egbFile);
	EncogErrorCheck();

	printf("Evaluate Error\n");
	printf("Input Count: %i\n", data->inputCount);
	printf("Ideal Count: %i\n", data->idealCount);
	printf("Record Count: %ld\n", data->recordCount);	    

	net = EncogNetworkLoad(egFile);
	EncogErrorCheck();

	if( data->inputCount != net->inputCount ) {
		EncogObjectFree(net);
		EncogErrorCheck();
		printf("Error: The network has a different input count than the training data.\n");
		return;
	}

	if( data->idealCount != net->outputCount ) {
		EncogObjectFree(net);
		EncogErrorCheck();
		printf("Error: The network has a different output count than the training data.\n");
		return;
	}

	error = EncogErrorSSE(net,data);

	EncogObjectFree(net);
	EncogObjectFree(data);

	*line = 0;
	EncogStrCatStr(line,"SSE Error: ",MAX_STR);
	EncogStrCatDouble(line,error*100.0,2,MAX_STR);
	EncogStrCatStr(line,"%\n",MAX_STR);
	puts(line);
}

void RandomizeNetwork(char *egFile) {
	ENCOG_NEURAL_NETWORK *net;

	if( *egFile==0  ) {
		printf("Usage: randomize [egFile]\n");
		return;
	}

	net = EncogNetworkLoad(egFile);
	EncogErrorCheck();

	EncogNetworkRandomizeRange(net,-1,1);
	EncogErrorCheck();

	EncogNetworkSave(egFile,net);
	EncogErrorCheck();


	printf("Network randomized and saved.\n");
}

void enableGPU(char *str) {
#ifdef ENCOG_CUDA
	 if( !EncogUtilStrcmpi(str,"enable") ) {
		 encogContext.gpuEnabled = 1;
	 } else if( !EncogUtilStrcmpi(str,"disable") ) {
		 encogContext.gpuEnabled = 0;
	 } else {
	 }
#else
	cudaNotCompiled();
#endif
}

int main(int argc, char* argv[])
{
	double started, ended;
	INT i;
	int phase = 0;
	char command[MAX_STR];
	char arg1[MAX_STR];
	char arg2[MAX_STR];
	char arg3[MAX_STR];
	char *cudastr;
	
	started = omp_get_wtime( );
#ifdef ENCOG_CUDA
	cudastr = ", CUDA";
#else 
	cudastr = "";
#endif

	EncogInit();
	printf("\n* * Encog C/C++(%i bit%s) Command Line v%s * *\n",(int)(sizeof(void*)*8),cudastr,encogContext.version);
	printf("Processor/Core Count: %i\n", (int)omp_get_num_procs());
	printf("Basic Data Type: %s (%i bits)\n", (sizeof(REAL)==8)?"double":"float", (int)sizeof(REAL)*8);

	*arg1=*arg2=*arg3=0;

	for(i=1;i<(INT)argc;i++) {
		if( *argv[i]=='/' || *argv[i]=='-' )
		{
			ParseOption(argv[i]);
			EncogHashPut(encogContext.config,parsedOption,strdup(parsedArgument)); 			
		}
		else 
		{
			if( phase==0 ) {
				strncpy(command,argv[i],MAX_STR);
				EncogUtilStrlwr(command);
			} else if( phase==1 ) {
				strncpy(arg1,argv[i],MAX_STR);
			} else if( phase==2 ) {
				strncpy(arg2,argv[i],MAX_STR);
			} else if( phase==3 ) {
				strncpy(arg3,argv[i],MAX_STR);
			}

			phase++;
		}
	}

#ifdef ENCOG_CUDA
	printf("GPU: %s\n",encogContext.gpuEnabled?"enabled":"disabled");
#endif

	if(!EncogUtilStrcmpi(command,"xor") ) {
		XORTest();
	} else if (!EncogUtilStrcmpi(command,"benchmark") ) {
		RunBenchmark();
	} else if (!EncogUtilStrcmpi(command,"train") ) {
		train(arg1,arg2);
	} else if (!EncogUtilStrcmpi(command,"egb2csv") ) {
		EGB2CSV(arg1,arg2);
	} else if (!EncogUtilStrcmpi(command,"csv2egb") ) {
		CSV2EGB(arg1,arg2);
	} else if (!EncogUtilStrcmpi(command,"create") ) {
		CreateNetwork(arg1,arg2,arg3);
	} else if (!EncogUtilStrcmpi(command,"randomize") ) {
		RandomizeNetwork(arg1);
	} else if (!EncogUtilStrcmpi(command,"error") ) {
		EvaluateError(arg1,arg2);
	} else if (!EncogUtilStrcmpi(command,"node") ) {
		EncogNodeMain(8080);
	}else if (!EncogUtilStrcmpi(command,"cuda") ) {
#ifdef ENCOG_CUDA
		if( encogContext.gpuEnabled ) {
			TestCUDA();
		} else {
			printf("CUDA has been disable, can't test it\n");
		}
#else
		cudaNotCompiled();
#endif
	} else {
		Usage();
	}

	ended = omp_get_wtime( );

	EncogShutdown();

	*command = 0;
	EncogStrCatStr(command,"Encog Finished.  Run time ",sizeof(command));
	EncogStrCatRuntime(command, ended-started, sizeof(command));
	puts(command);

    return 0;
}
