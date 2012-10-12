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
#include "encog.h"

static char _arg[MAX_STR];
static int _currentError = ENCOG_ERROR_OK;
static char *_errorMessages[] = 
{
	"Success",							/* 0 */
	"File not found",					/* 1 */
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
	"Invalid layer conditional (?), must have only two",	/* 12 */
	"Encog object error", /* 13 */
	"Encog object type error", /* 14 */
	"Unknown training type error", /* 15 */

};

void EncogErrorClear() {
	EncogErrorSet(ENCOG_ERROR_OK);
}

void EncogErrorSet(int e) {
	_currentError = e;
}

void EncogErrorSetArg(char *arg) {
	strncpy(_arg,arg,sizeof(_arg));
}

char *EncogErrorArgument() {
	return _arg;
}

int EncogErrorGet() {
	return _currentError;
}

void EncogErrorCheck() {
	if( EncogErrorGet() !=ENCOG_ERROR_OK ) {
		printf("** Encog function failure **\n");
		printf("Error Code: %i, Error Message: %s\n%s\n",
			EncogErrorGet(),
			EncogErrorMessage(),
			EncogErrorArgument());
		exit(1);
	}
}

char *EncogErrorMessage() {
	return _errorMessages[EncogErrorGet()];
}

