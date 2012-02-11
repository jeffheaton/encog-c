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
#define ENCOG_ERROR_FACTORY_INVALID_COND 12

#define SIZE_BYTE 1
#define SIZE_KILOBYTE (SIZE_BYTE*1024)
#define SIZE_MEGABYTE (SIZE_KILOBYTE*1024)

/* Deal with Microsoft Visual C++ */
#ifdef _MSC_VER
#pragma warning( disable : 4996 )
#endif

#define MAX_STR 200

#ifndef MAX
#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))
#endif

typedef double REAL;
typedef unsigned int INT;

typedef void(*ACTIVATION_FUNCTION)(REAL *,int);
typedef void(*ENCOG_TASK)(void*);

struct ENCOG_TRAIN_PSO;

typedef struct NETWORK_LAYER
{
    struct NETWORK_LAYER *next;
    INT feedCount;
    INT totalCount;
    ACTIVATION_FUNCTION af;
    unsigned char bias;
} NETWORK_LAYER;

typedef struct
{
    INT layerCount;
    INT neuronCount;
    INT weightCount;

    /**
    	 * The number of input neurons in this network.
    	 */
    INT inputCount;

    /**
     * The number of neurons in each of the layers.
     */
    INT *layerCounts;

    /**
     * The number of context neurons in each layer. These context neurons will
     * feed the next layer.
     */
    INT *layerContextCount;

    /**
     * The number of neurons in each layer that are actually fed by neurons in
     * the previous layer. Bias neurons, as well as context neurons, are not fed
     * from the previous layer.
     */
    INT *layerFeedCounts;

    /**
     * An index to where each layer begins (based on the number of neurons in
     * each layer).
     */
    INT *layerIndex;

    /**
     * The outputs from each of the neurons.
     */
    REAL *layerOutput;

    /**
     * The sum of the layer, before the activation function is applied, producing the layerOutput.
     */
    REAL *layerSums;

    /**
     * The number of output neurons in this network.
     */
    INT outputCount;

    /**
     * The index to where the weights that are stored at for a given layer.
     */
    INT *weightIndex;

    /**
     * The weights for a neural network.
     */
    REAL *weights;

    /**
     * The activation types.
     */
    ACTIVATION_FUNCTION *activationFunctions;

    /**
     * The bias activation for each layer. This is usually either 1, for a bias,
     * or zero for no bias.
     */
    REAL *biasActivation;

		INT beginTraining;
	REAL connectionLimit;
	INT *contextTargetOffset;
	INT *contextTargetSize;
	INT endTraining;
	INT hasContext;
	INT totalNetworkSize;

} ENCOG_NEURAL_NETWORK;

typedef struct
{
    INT inputCount;
    INT idealCount;
    unsigned long recordCount;
    REAL *cursor;
    REAL *data;
} ENCOG_DATA;

typedef struct
{
    ENCOG_NEURAL_NETWORK *network;
    REAL *velocities;
    REAL *bestVector;
    REAL *vtemp;
    float bestError;
	INT index;
	struct ENCOG_TRAIN_PSO *pso;
} ENCOG_PARTICLE;

typedef struct ENCOG_TRAIN_PSO
{
    ENCOG_PARTICLE *particles;
    int bestParticle;

    // Typical range is 20 - 40 for many problems.
    // More difficult problems may need much higher value.
    // Must be low enough to keep the training process
    // computationally efficient.
    int populationSize;

    // Determines the size of the search space.
    // The position components of particle will be bounded to
    // [-maxPos, maxPos]
    // A well chosen range can improve the performance.
    // -1 is a special value that represents boundless search space.
    double maxPosition;

    // Maximum change one particle can take during one iteration.
    // Imposes a limit on the maximum absolute value of the velocity
    // components of a particle.
    // Affects the granularity of the search.
    // If too high, particle can fly past optimum solution.
    // If too low, particle can get stuck in local minima.
    // Usually set to a fraction of the dynamic range of the search
    // space (10% was shown to be good for high dimensional problems).
    // -1 is a special value that represents boundless velocities.
    double maxVelocity;

    // c1, cognitive learning rate >= 0
    // tendency to return to personal best position
    double c1;

    // c2, social learning rate >= 0
    // tendency to move towards the swarm best position
    double c2;

    // w, inertia weight.
    // Controls global (higher value) vs local exploration
    // of the search space.
    // Analogous to temperature in simulated annealing.
    // Must be chosen carefully or gradually decreased over time.
    // Value usually between 0 and 1.
    double inertiaWeight;

    // If true, the position of the previous global best position
    // can be updated *before* the other particles have been modified.
    int pseudoAsynchronousUpdate;

    int dimensions;

    double *bestVector;
    float bestError;

    ENCOG_DATA *data;

} ENCOG_TRAIN_PSO;

typedef struct {
char ident[8];
double input;
double ideal;
} EGB_HEADER;

void EncogActivationLinear(REAL *d,int count);
void EncogActivationSigmoid(REAL *d,int count);
void EncogActivationTANH(REAL *d,int count);

ENCOG_NEURAL_NETWORK *EncogNetworkNew();
void EncogNetworkDelete(ENCOG_NEURAL_NETWORK *network);
ENCOG_NEURAL_NETWORK *EncogNetworkFinalizeStructure(NETWORK_LAYER *firstLayer, int freeLayers);
NETWORK_LAYER *EncogNetworkCreateLayer(NETWORK_LAYER *prevLayer, int count, ACTIVATION_FUNCTION af, unsigned char bias);
void EncogNetworkCompute(ENCOG_NEURAL_NETWORK *net,REAL *input, REAL *output);
void EncogNetworkRandomizeRange(ENCOG_NEURAL_NETWORK *net,REAL low, REAL high);
void EncogNetworkImportWeights(ENCOG_NEURAL_NETWORK *net, REAL *weights);
void EncogNetworkExportWeights(ENCOG_NEURAL_NETWORK *net, REAL *weights);
void EncogNetworkDump(ENCOG_NEURAL_NETWORK *net);
void EncogNetworkClearContext(ENCOG_NEURAL_NETWORK *net);
ENCOG_NEURAL_NETWORK *EncogNetworkClone(ENCOG_NEURAL_NETWORK *net);
ENCOG_NEURAL_NETWORK *EncogNetworkLoad(char *name);
void EncogNetworkSave(char *name, ENCOG_NEURAL_NETWORK *network);
ENCOG_NEURAL_NETWORK *EncogNetworkFactory(char *method, char *architecture, int defaultInputCount, int defaultOutputCount);
void EncogNetworkLink(ENCOG_NEURAL_NETWORK *net);

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
char *EncogUtilStrlwr(char *string);
char *EncogUtilStrupr(char *strint);
int EncogUtilStrcmpi(char *s1, char *s2);

void EncogDataCSVSave(char *filename, ENCOG_DATA *data, int decimals);
ENCOG_DATA *EncogDataCSVLoad(char *csvFile, INT inputCount, INT idealCount);
ENCOG_DATA *EncogDataCreate(unsigned int inputCount, unsigned int idealCount, unsigned long records);
void EncogDataDelete(ENCOG_DATA *data);
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
void EncogVectorMulRand(REAL* v, double k, int length);
void EncogVectorMul(REAL* v, double k, int length);
void EncogVectorCopy(REAL* dst, REAL *src, int length);
void EncogVectorRandomise(REAL* v, REAL maxValue, int length);
void EncogVectorRandomiseDefault(REAL* v, int length);
void EncogVectorClampComponents(REAL* v, double maxValue,int length);

float EncogErrorSSE(ENCOG_NEURAL_NETWORK *net, ENCOG_DATA *data);

ENCOG_TRAIN_PSO *EncogTrainPSONew(int populationSize, ENCOG_NEURAL_NETWORK *model, ENCOG_DATA *data);
void EncogTrainPSODelete(ENCOG_TRAIN_PSO *pso);
float EncogTrainPSOIterate(ENCOG_TRAIN_PSO *pso);
void EncogTrainPSOImportBest(ENCOG_TRAIN_PSO *pso, ENCOG_NEURAL_NETWORK *net);

void EncogErrorClear();
void EncogErrorSet(int e);
int EncogErrorGet();
void EncogErrorCheck();
char *EncogErrorMessage();

int EncogStrPopLine(char *line, char *arg, int start, int len);
int EncogStrIsWhiteSpace(char ch);
void EncogStrTrim(char *line);
void EncogStrStripQuotes(char *line);
char *EncogStrParseNV(char *line, char *name, size_t len);
void EncogStrStripCRLF(char *str);
int EncogStrCountValues(char *line);
INT *EncogStrParseIntList(char *line);
int EncogStrParseBoolean(char *line);
double *EncogStrParseDoubleList(char *line);

void EncogFileWriteValueInt(FILE *fp, char *name, INT value);
void EncogFileWriteValueBoolean(FILE *fp, char *name, INT value);
void EncogFileWriteValueIntArray(FILE *fp, char *name, INT *a, INT count);
void EncogFileWriteValueDouble(FILE *fp, char *name, double value);
void EncogFileWriteValueDoubleArray(FILE *fp, char *name, double *a, INT count);

#ifdef ENCOG_CUDA
float EncogNetworkGPUEval(ENCOG_NEURAL_NETWORK *net, ENCOG_DATA *data);
#endif

#ifdef __cplusplus
}
#endif 

#endif
