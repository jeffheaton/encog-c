#include "encog.h"

static int _currentError = ENCOG_ERROR_OK;
static char *_errorMessages[] = 
{
	"Success",				/* 0 */
	"File  not found",		/* 1 */
	"IO Error",				/* 2 */
	"Size mismatch",			/* 3 */
	"Network has not been finalized", /* 4 */
	"Network has already been finalized" /* 5 */
};

void EncogErrorClear() {
	EncogErrorSet(ENCOG_ERROR_OK);
}

void EncogErrorSet(int e) {
	_currentError = e;
}

int EncogErrorGet() {
	return _currentError;
}

void EncogErrorCheck() {
	if( EncogErrorGet() !=ENCOG_ERROR_OK ) {
		printf("** Encog function failure **\n");
		printf("Error Code: %i, Error Message: %s\n",
			EncogErrorGet(),
			EncogErrorMessage());
		exit(1);
	}
}

char *EncogErrorMessage() {
	return _errorMessages[EncogErrorGet()];
}