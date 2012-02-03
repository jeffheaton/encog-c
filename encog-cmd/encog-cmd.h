#ifndef __ENCOG_CMD_H
#define __ENCOG_CMD_H

#ifdef __cplusplus
extern "C" {
#endif

#include "encog.h"

void TestCUDA();

void TrainNetwork(ENCOG_TRAIN_PSO *pso, float maxError,int updateSeconds);

#ifdef __cplusplus
}
#endif 

#endif
