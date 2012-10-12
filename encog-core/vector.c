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

/**
     * v1 = v1 + v2
     *
     * @param v1    an array of doubles
     * @param v2    an array of doubles
     */
void EncogVectorAdd(REAL *v1, REAL *v2, int length)
{
    int i;
    for (i = 0; i < length; i++)
    {
        v1[i] += v2[i];
    }
}

/**
 * v1 = v1 - v2
 *
 * @param v1    an array of doubles
 * @param v2    an array of doubles
 */
void EncogVectorSub(REAL* v1, REAL* v2, int length)
{
    int i;
    for (i = 0; i < length; i++)
    {
        v1[i] -= v2[i];
    }
}

/**
 * v = -v
 *
 * @param v     an array of doubles
 */
void EncogVectorNeg(REAL* v, int length)
{
    int i;
    for (i = 0; i < length; i++)
    {
        v[i] = -v[i];
    }
}

/**
 * v = k * U(0,1) * v
 *
 * The components of the vector are multiplied
 * by k and a random number.
 * A new random number is generated for each
 * component.
 * Thread-safety depends on Random.nextDouble()
 *
 * @param v     an array of doubles.
 * @param k     a scalar.
 */
void EncogVectorMulRand(REAL* v, REAL k, int length)
{
    int i;
    for (i = 0; i < length; i++)
    {
        v[i] *= k * ((REAL)rand()/(REAL)RAND_MAX);
    }
}

/**
 * v = k * v
 *
 * The components of the vector are multiplied
 * by k.
 *
 * @param v     an array of doubles.
 * @param k     a scalar.
 */
void EncogVectorMul(REAL* v, REAL k, int length)
{
    int i;
    for (i = 0; i < length; i++)
    {
        v[i] *= k;
    }
}

/**
 * dst = src
 * Copy a vector.
 *
 * @param dst   an array of doubles
 * @param src   an array of doubles
 */
void EncogVectorCopy(REAL* dst, REAL *src, int length)
{
    memcpy(dst,src,length*sizeof(REAL));
}

/**
 * v = U(-1, 1) * maxValue
 *
 * Randomise each component of a vector to
 * [-maxValue, maxValue].
 * thread-safety depends on Random.nextDouble().
 *
 * @param v     an array of doubles
 */
void EncogVectorRandomise(REAL* v, REAL maxValue, int length)
{
    int i;
    for ( i = 0; i < length; i++)
    {
        v[i] = EncogUtilRandomRange(-maxValue, maxValue);
    }
}

/**
     * v = U(0, 0.1)
     *
     * @param v     an array of doubles
     */
void EncogVectorRandomiseDefault(REAL* v, int length)
{
    EncogVectorRandomise(v, (REAL)0.1, length);
}


/**
 * For each components, reset their value to maxValue if
 * their absolute value exceeds it.
 *
 * @param v         an array of doubles
 * @param maxValue  if -1 this function does nothing
 */
void EncogVectorClampComponents(REAL* v, REAL maxValue,int length)
{
    int i;
    if (maxValue != -1)
    {
        for (i = 0; i < length; i++)
        {
            if (v[i] > maxValue) v[i] = maxValue;
            if (v[i] < -maxValue) v[i] = -maxValue;
        }
    }
}
