#ifndef __ENCOG_CMD_H
#define __ENCOG_CMD_H

#ifdef __cplusplus
extern "C" {
#endif

#include "encog.h"

int TestVectorAdd();
void TestCUDA();

void EncogNodeMain(int port);
void EncogNodeRecv(unsigned char b);

#ifdef __cplusplus
}
#endif 

#endif
