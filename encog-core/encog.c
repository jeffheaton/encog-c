#include "encog.h"

ENCOG_CONTEXT encogContext;

void EncogInit() {
	memset(&encogContext,0,sizeof(ENCOG_CONTEXT));

#ifdef ENCOG_CUDA
	encogContext.gpuEnabled = 1;
#endif	
	
	encogContext.versionMajor = 0;
	encogContext.versionMajor = 1;
	strncpy(encogContext.version,"0.1",sizeof(encogContext.version));

	EncogUtilInitRandom();
}

void EncogShutdown() {
}