#ifndef __ENCOG_CMD_H
#define __ENCOG_CMD_H

#ifdef __cplusplus
extern "C" {
#endif

#include "encog.h"

int TestVectorAdd();
void TestCUDA();

void TrainNetwork(ENCOG_TRAIN_PSO *pso, float maxError,int updateSeconds);

#ifdef __cplusplus
}
#endif 

#endif
