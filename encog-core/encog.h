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
#ifndef __ENCOG_H
#define __ENCOG_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <ctype.h>
#include <omp.h>
#include <assert.h>
#ifdef _MSC_VER
#include <conio.h>
#endif


#define AF_LINEAR	0
#define AF_SIGMOID	1
#define AF_TANH		2

#define TRAIN_TYPE_PSO "PSO"
#define TRAIN_TYPE_RPROP "RPROP"
#define TRAIN_TYPE_NM "NM"

#define ENCOG_ERROR_OK				0
#define ENCOG_ERROR_FILE_NOT_FOUND	1
#define ENCOG_ERROR_IO				2
#define ENCOG_ERROR_SIZE_MISMATCH	3
#define ENCOG_ERROR_INVALID_EG_FILE	4
#define ENCOG_ERROR_INVALID_EGB_FILE	5
#define ENCOG_ERROR_INVALID_EGA_FILE	6
#define ENCOG_ERROR_NETWORK_NOT_FINALIZED 7
#define ENCOG_ERROR_NETWORK_FINALIZED 8
#define ENCOG_ERROR_MIN_2LAYER 9
#define ENCOG_ERROR_FACTORY_INVALID_ACTIVATION 10
#define ENCOG_ERROR_FACTORY_INVALID_BIAS 11
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
#define ENCOG_ERROR_FACTORY_INVALID_COND 12
#define ENCOG_ERROR_OBJECT	13
#define ENCOG_ERROR_OBJECT_TYPE	14
#define	ENCOG_ERROR_UNKNOWN_TRAINING 15

#define PARAM_INPUT			"INPUT"
#define PARAM_IDEAL			"IDEAL"
#define PARAM_RECORDS		"RECORDS"
#define PARAM_ITERATIONS	"ITERATIONS"
#define PARAM_THREADS		"THREADS"
#define PARAM_PARTICLES		"PARTICLES"
#define PARAM_INERTIA		"INERTIA"
#define PARAM_C1			"C1"
#define PARAM_C2			"C2"
#define PARAM_GPU			"GPU"
#define PARAM_TRAIN			"TRAIN"
#define PARAM_MAXPOS		"MAXPOS"
#define PARAM_MAXVEL		"MAXVEL"

/* Nelder Mead */
#define PARAM_STEP			"STEP"
#define PARAM_KONVERGE		"KONVERGE"
#define PARAM_REQMIN		"REQMIN"

#define ENCOG_TYPE_NEURAL_NETWORK	1
#define ENCOG_TYPE_DATA				2
#define ENCOG_TYPE_PSO				3
#define ENCOG_TYPE_RPROP			4
#define ENCOG_TYPE_HASH				5
#define ENCOG_TYPE_NM				6

#define SIZE_BYTE 1
#define SIZE_KILOBYTE (SIZE_BYTE*1024)
#define SIZE_MEGABYTE (SIZE_KILOBYTE*1024)

#define PARTICLE_STATE_MOVE			0
#define PARTICLE_STATE_MOVING		1
#define PARTICLE_STATE_CALC			2
#define PARTICLE_STATE_CALCING		3
#define PARTICLE_STATE_ITERATION	4

/* Deal with Microsoft Visual C++ */
#ifdef _MSC_VER
#pragma warning( disable : 4996 )
#define snprintf _snprintf
#endif

#define MAX_STR 200

#ifndef MAX
#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))
#endif

typedef double REAL;
typedef unsigned int INT;

struct ENCOG_TRAIN_PSO;
struct ENCOG_TRAINING_REPORT;
struct ENCOG_HASH;

typedef void(*ACTIVATION_FUNCTION)(REAL *,int);
typedef REAL(*DERIVATIVE_FUNCTION)(REAL b, REAL a);
typedef void(*ENCOG_TASK)(void*);

typedef struct ENCOG_OBJECT {
	char id[2];
	char type;
} ENCOG_OBJECT;

typedef struct ENCOG_CONTEXT {
#ifdef ENCOG_CUDA
	INT gpuEnabled;
#endif
	char version[10];
	INT versionMajor;
	INT versionMinor;
	struct ENCOG_HASH *config;
} ENCOG_CONTEXT;

typedef struct GPU_DEVICE {
	INT deviceID;
	REAL *deviceData;
//	REAL *deviceDynamic;
	float *deviceErrors;
	REAL *deviceWeights;
	float *errors;
	INT recordCount;
	INT perfCount;
	float perfKernelTime;
	INT blocksPerGrid;
} GPU_DEVICE;

typedef struct NETWORK_LAYER
{
    struct NETWORK_LAYER *next;
    INT feedCount;
    INT totalCount;
    INT af;
    unsigned char bias;
} NETWORK_LAYER;

typedef struct ENCOG_NEURAL_NETWORK
{
	ENCOG_OBJECT encog;
    INT layerCount;
    INT neuronCount;
    INT weightCount;
    INT inputCount; /* Input neuron count */
    INT *layerCounts; /* neuron count per layer */
    INT *layerContextCount; /* context neurons per layer */
    INT *layerFeedCounts; /* number of neurons, per layer, fed by prev layer */
    INT *layerIndex; /* index to begin of each layer */
    REAL *layerOutput; /* The outputs from each of the neurons. */
    REAL *layerSums; /* sum of each layer (before activ) */
    INT outputCount; /* output neuron count */
    INT *weightIndex; /* index to layer weights */
    REAL *weights; /* neural network weights */
    ACTIVATION_FUNCTION *activationFunctions; /* The activation types. */
	DERIVATIVE_FUNCTION *derivativeFunctions;
	INT *activationFunctionIDs;
    REAL *biasActivation; /* bias activation, per layer, typically 1 */
	INT beginTraining;
	REAL connectionLimit;
	INT *contextTargetOffset;
	INT *contextTargetSize;
	INT endTraining;
	INT hasContext;	

} ENCOG_NEURAL_NETWORK;

typedef struct ENCOG_DATA
{
	ENCOG_OBJECT encog;
    INT inputCount;
    INT idealCount;
    unsigned long recordCount;
    REAL *cursor;
    REAL *data;
} ENCOG_DATA;

typedef struct ENCOG_TRAINING_REPORT {
	float error;
	INT iterations;
	int stopRequested;
	time_t lastUpdate;
	time_t trainingStarted;
	float maxError;
	INT maxIterations;
	INT updateSeconds;
	ENCOG_OBJECT *trainer;
} ENCOG_TRAINING_REPORT;

typedef void(*ENCOG_REPORT_FUNCTION)(ENCOG_TRAINING_REPORT *);

typedef struct
{
    ENCOG_NEURAL_NETWORK *network;
    REAL *velocities;
    REAL *bestVector;
    REAL *vtemp;
    float bestError;
	INT index;
	INT particleState;
	struct ENCOG_TRAIN_PSO *pso;
} ENCOG_PARTICLE;

typedef struct ENCOG_TRAIN_PSO
{
	ENCOG_OBJECT encog;
    ENCOG_PARTICLE *particles;
    int bestParticle;
    int populationSize;
    REAL maxPosition;
    REAL maxVelocity;
    REAL c1;
    REAL c2;
    REAL inertiaWeight;

    int dimensions;

    REAL *bestVector;
    float bestError;
	GPU_DEVICE *device;

    ENCOG_DATA *data;
	float cudaKernelTime;
	INT cudaKernelCalls;
	float cpuWorkUnitTime;
	INT cpuWorkUnitCalls;
	ENCOG_TRAINING_REPORT currentReport;
	ENCOG_REPORT_FUNCTION reportTarget;

} ENCOG_TRAIN_PSO;

typedef struct ENCOG_TRAIN_RPROP
{
	ENCOG_OBJECT encog;
	ENCOG_DATA *data;
	ENCOG_TRAINING_REPORT currentReport;
	ENCOG_REPORT_FUNCTION reportTarget;
	ENCOG_NEURAL_NETWORK **network;
	ENCOG_NEURAL_NETWORK *targetNetwork;

	REAL *lastGradient;
	REAL *updateValues;
	REAL *lastWeightChange;

	REAL *gradients;
	REAL **layerDelta;
	float errorSum;
	int threadCount;

} ENCOG_TRAIN_RPROP;

typedef struct ENCOG_NM
{
	ENCOG_OBJECT encog;
	ENCOG_DATA *data;
	ENCOG_TRAINING_REPORT currentReport;
	ENCOG_REPORT_FUNCTION reportTarget;
	ENCOG_NEURAL_NETWORK **network;
	ENCOG_NEURAL_NETWORK *targetNetwork;
	REAL step;
	REAL reqmin;
	float error;
	int n;
	int konvge; 
	int numres; 
	int ifault;
	int threadCount;
} ENCOG_TRAIN_NM;

typedef struct {
char ident[8];
double input;
double ideal;
} EGB_HEADER;

typedef struct ENCOG_HASH_NODE {
	char *key;
	void *value;
	struct ENCOG_HASH_NODE *next;
	int hashCode;
} ENCOG_HASH_NODE;

typedef struct ENCOG_HASH {
	ENCOG_OBJECT encog;
	INT tableSize;
	INT ignoreCase;
	ENCOG_HASH_NODE **table;
} ENCOG_HASH;


void EncogActivationLinear(REAL *d,int count);
void EncogActivationSigmoid(REAL *d,int count);
void EncogActivationTANH(REAL *d,int count);

REAL EncogDerivativeLinear(REAL b, REAL a);
REAL EncogDerivativeSigmoid(REAL b, REAL a);
REAL EncogDerivativeTANH(REAL b, REAL a);

ENCOG_NEURAL_NETWORK *EncogNetworkNew();
ENCOG_NEURAL_NETWORK *EncogNetworkFinalizeStructure(NETWORK_LAYER *firstLayer, int freeLayers);
NETWORK_LAYER *EncogNetworkCreateLayer(NETWORK_LAYER *prevLayer, int count, INT af, unsigned char bias);
void EncogNetworkCompute(ENCOG_NEURAL_NETWORK *net,REAL *input, REAL *output);
void EncogNetworkRandomizeRange(ENCOG_NEURAL_NETWORK *net,REAL low, REAL high);
void EncogNetworkImportWeights(ENCOG_NEURAL_NETWORK *net, REAL *weights);
void EncogNetworkExportWeights(ENCOG_NEURAL_NETWORK *net, REAL *weights);
void EncogNetworkDump(ENCOG_NEURAL_NETWORK *net);
void EncogNetworkClearContext(ENCOG_NEURAL_NETWORK *net);
ENCOG_NEURAL_NETWORK *EncogNetworkClone(ENCOG_NEURAL_NETWORK *net);
ENCOG_NEURAL_NETWORK *EncogNetworkTransactionClone(ENCOG_NEURAL_NETWORK *net);
ENCOG_NEURAL_NETWORK *EncogNetworkLoad(char *name);
void EncogNetworkSave(char *name, ENCOG_NEURAL_NETWORK *network);
ENCOG_NEURAL_NETWORK *EncogNetworkFactory(char *method, char *architecture, int defaultInputCount, int defaultOutputCount);
ACTIVATION_FUNCTION EncogNetworkResolveAF(INT af);
DERIVATIVE_FUNCTION EncogNetworkResolveDR(INT af);

void EncogUtilInitRandom();
REAL EncogUtilRandomRange(REAL low, REAL high);
void EncogUtilOutputRealArray(char *tag, REAL *array, INT len);
void EncogUtilOutputIntArray(char *tag, INT *array, INT len);
void *EncogUtilAlloc(size_t nelem, size_t elsize);
void EncogUtilFree(void *m);
void *EncogUtilDuplicateMemory(void *source,size_t elementCount,size_t elementSize);
void EncogStrCatChar(char *base, char ch, size_t len);
void EncogStrCatStr(char *base, char *str, size_t len );
void EncogStrCatDouble(char *base, double d, int decimals,size_t len);
void EncogStrCatInt(char *base, INT i,size_t len);
void EncogStrCatLong(char *base, long i,size_t len);
void EncogStrCatNL(char *base, size_t len);
void EncogStrCatRuntime(char *base, double t,size_t len);
char *EncogUtilStrlwr(char *string);
char *EncogUtilStrupr(char *strint);
int EncogUtilStrcmpi(char *s1, char *s2);
unsigned long EncogUtilHash(unsigned char *str);
#ifndef _MSC_VER
int kbhit(void);
#endif

void EncogDataCSVSave(char *filename, ENCOG_DATA *data, int decimals);
ENCOG_DATA *EncogDataCSVLoad(char *csvFile, INT inputCount, INT idealCount);
ENCOG_DATA *EncogDataCreate(unsigned int inputCount, unsigned int idealCount, unsigned long records);
void EncogDataAddVar(ENCOG_DATA *data, ...);
void EncogDataAdd(ENCOG_DATA *data,char *str);
REAL *EncogDataGetInput(ENCOG_DATA *data, INT index);
REAL *EncogDataGetIdeal(ENCOG_DATA *data, INT index);
ENCOG_DATA *EncogDataGenerateRandom(INT inputCount, INT idealCount, INT records, REAL low, REAL high);
ENCOG_DATA *EncogDataEGBLoad(char *f);
void EncogDataEGBSave(char *egbFile,ENCOG_DATA *data);

void EncogVectorAdd(REAL *v1, REAL *v2, int length);
void EncogVectorSub(REAL* v1, REAL* v2, int length);
void EncogVectorNeg(REAL* v, int length);
void EncogVectorMulRand(REAL* v, REAL k, int length);
void EncogVectorMul(REAL* v, REAL k, int length);
void EncogVectorCopy(REAL* dst, REAL *src, int length);
void EncogVectorRandomise(REAL* v, REAL maxValue, int length);
void EncogVectorRandomiseDefault(REAL* v, int length);
void EncogVectorClampComponents(REAL* v, REAL maxValue,int length);

float EncogErrorSSE(ENCOG_NEURAL_NETWORK *net, ENCOG_DATA *data);
float EncogCPUErrorSSE(ENCOG_NEURAL_NETWORK *net, ENCOG_DATA *data);

ENCOG_TRAIN_PSO *EncogTrainPSONew(int populationSize, ENCOG_NEURAL_NETWORK *model, ENCOG_DATA *data);
void EncogTrainPSODelete(ENCOG_TRAIN_PSO *pso);
float EncogTrainPSORun(ENCOG_TRAIN_PSO *pso);
void EncogTrainPSOImportBest(ENCOG_TRAIN_PSO *pso, ENCOG_NEURAL_NETWORK *net);
void EncogTrainPSOFinish(ENCOG_TRAIN_PSO *pso);

ENCOG_TRAIN_RPROP *EncogTrainRPROPNew(ENCOG_NEURAL_NETWORK *network, ENCOG_DATA *data);
float EncogTrainRPROPRun(ENCOG_TRAIN_RPROP *rprop);

void EncogErrorClear();
int EncogErrorGet();
void EncogErrorCheck();
char *EncogErrorMessage();
char *EncogErrorArgument();
void EncogErrorSet(int e);
void EncogErrorSetArg(char *arg);

int EncogStrPopLine(char *line, char *arg, int start, int len);
int EncogStrIsWhiteSpace(char ch);
void EncogStrTrim(char *line);
void EncogStrStripQuotes(char *line);
char *EncogStrParseNV(char *line, char *name, size_t len);
void EncogStrStripCRLF(char *str);
int EncogStrCountValues(char *line);
INT *EncogStrParseIntList(char *line);
int EncogStrParseBoolean(char *line);
REAL *EncogStrParseDoubleList(char *line);

void EncogFileWriteValueInt(FILE *fp, char *name, INT value);
void EncogFileWriteValueBoolean(FILE *fp, char *name, INT value);
void EncogFileWriteValueIntArray(FILE *fp, char *name, INT *a, INT count);
void EncogFileWriteValueDouble(FILE *fp, char *name, REAL value);
void EncogFileWriteValueDoubleArray(FILE *fp, char *name, REAL *a, INT count);

ENCOG_HASH *EncogHashNew(INT tableSize, INT ignoreCase);
void EncogHashPut(ENCOG_HASH *hashTable, char *key, void *obj);
void *EncogHashGet(ENCOG_HASH *hashTable, char *key, void *defaultValue);
void EncogHashDump(ENCOG_HASH *hashTable);
int EncogHashGetInteger(ENCOG_HASH *hashTable, char *key, int defaultValue);
float EncogHashGetFloat(ENCOG_HASH *hashTable, char *key, float defaultValue);
int EncogHashContains(ENCOG_HASH *hashTable, char *key);

void EncogInit();
void EncogShutdown();
void EncogTrainMinimalCallback(ENCOG_TRAINING_REPORT *report);
void EncogTrainStandardCallback(ENCOG_TRAINING_REPORT *report);

void EncogObjectRegister(void *obj, int type);
void EncogObjectValidate(void *obj, int type);
void EncogObjectFree(void *obj);
int EncogObjectGetType(ENCOG_OBJECT *encogObject);
char *EncogObjectType(ENCOG_OBJECT *encogObject);

ENCOG_OBJECT *EncogTrainNew(ENCOG_NEURAL_NETWORK *net, ENCOG_DATA *data);
void EncogTrainRun(ENCOG_OBJECT *train, ENCOG_NEURAL_NETWORK *net);
ENCOG_TRAINING_REPORT *EncogTrainReport(ENCOG_OBJECT *train);
void EncogTrainSetCallback(ENCOG_OBJECT *train, ENCOG_REPORT_FUNCTION callback);

ENCOG_TRAIN_NM *EncogTrainNMNew(ENCOG_NEURAL_NETWORK *network, ENCOG_DATA *data);
float EncogTrainNMRun(ENCOG_TRAIN_NM *nm);


#ifdef ENCOG_CUDA

GPU_DEVICE *EncogGPUDeviceNew(INT deviceNumber, ENCOG_NEURAL_NETWORK *net, ENCOG_DATA *data);
void EncogGPUDeviceDelete(GPU_DEVICE *device);
float EncogCUDAErrorSSE(GPU_DEVICE *device, ENCOG_NEURAL_NETWORK *net);
float EncogCUDAPSOIterate(ENCOG_TRAIN_PSO *pso);
#endif

extern ENCOG_CONTEXT encogContext;

#ifdef __cplusplus
}
#endif 

#endif
