#include "encog.h"

static int _currentError = ENCOG_ERROR_OK;
static char *_errorMessages[] = 
{
	"Success",							/* 0 */
	"File  not found",					/* 1 */
	"IO Error",							/* 2 */
	"Size mismatch",					/* 3 */
	"Invalid EG File",					/* 4 */
	"Invalid EGB File",					/* 5 */
	"Invalid EGA File",					/* 6 */
	"Network has not been finalized",	/* 7 */
	"Network has already been finalized",	/* 8 */
	"Network must have at least two layers",	/* 9 */
	"Invalid activation function name",	/* 10 */
	"Expected a bias (b) to follow the :",	/* 11 */
	"Invalid layer conditional (?), must have only two"	/* 12 */
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

