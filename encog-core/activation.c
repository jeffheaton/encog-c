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

void EncogActivationLinear(REAL *d,int count)
{
}

void EncogActivationSigmoid(REAL *d,int count)
{
    int i;

    for(i=0; i<count; i++)
    {
        *d = (REAL)1.0 / ((REAL)1.0 + (REAL)exp((REAL)-1.0 * *d));
        d++;
    }
}

void EncogActivationTANH(REAL *d,int count)
{
    int i;
    for(i=0; i<count; i++)
    {
        *d = (REAL)tanh(*d);
        d++;
    }
}

REAL EncogDerivativeLinear(REAL b, REAL a)
{
	return 1;
}

REAL EncogDerivativeSigmoid(REAL b, REAL a)
{
	return a * (1.0 - a);
}

REAL EncogDerivativeTANH(REAL b, REAL a)
{
    return (1.0 - a * a);
}
