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
#include "encog.h"

#ifdef _MSC_VER
int isnan(double x) 
{ 
	return x != x; 
}

int isinf(double x) 
{ 
	return !isnan(x) && isnan(x - x); 
}
#endif

void EncogUtilInitRandom()
{
    unsigned int iseed;

	/* Clear out any previous errors */
	EncogErrorClear();
	
	iseed = (unsigned int)time(NULL);
    srand (iseed);
}

REAL EncogUtilRandomRange(REAL low, REAL high)
{
    REAL d;
    d = (REAL)rand()/(REAL)RAND_MAX;
    d = (d*(high-low))+low;
    return d;
}

void EncogUtilOutputRealArray(char *tag, REAL *array, INT len)
{
    INT i;
    char temp[MAX_STR];

    puts(tag);
    for(i=0; i<len; i++)
    {
        *temp = 0;
        EncogStrCatDouble(temp,array[i],4,MAX_STR);
        puts(temp);
    }
    printf("\n");
}

void EncogUtilOutputIntArray(char *tag, INT *array, INT len)
{
    INT i;

    puts(tag);
    for(i=0; i<len; i++)
    {
        printf("%i ",array[i]);
    }
    printf("\n");
}

void *EncogUtilAlloc(size_t nelem, size_t elsize)
{
    return calloc(nelem,elsize);
}

void EncogUtilFree(void *m)
{
    if( m!=NULL )
    {
        free(m);
    }
}

void *EncogUtilDuplicateMemory(void *source,size_t elementCount,size_t elementSize)
{
    void *result = EncogUtilAlloc(elementCount,elementSize);
    memcpy(result,source,elementCount*elementSize);
    return result;
}

/**
 * Double to ASCII, based on:
 * http://stackoverflow.com/questions/2302969/how-to-implement-char-ftoafloat-num-without-sprintf-library-function-i
 */
char * dtoa(char *s, double n, int maxDec)
{
	int digit, m, m1, useExp, i, j;
	char *c;
    double percision, weight;
	int neg;

    percision = pow(10.0,-maxDec);

    // handle special cases
    if (isnan(n))
    {
        strcpy(s, "nan");
    }
    else if (isinf(n))
    {
        strcpy(s, "inf");
    }
    else if (n == 0.0)
    {
        strcpy(s, "0");
    }
    else
    {        
        c = s;
        neg = (n < 0);
        if (neg)
            n = -n;
        // calculate magnitude
        m = (int)log10(n);
        useExp = (m >= 14 || (neg && m >= 9) || m <= -9);
        if (neg)
            *(c++) = '-';
        // set up for scientific notation
        if (useExp)
        {
            if (m < 0)
                m -= 1;
            n = n / pow(10.0, m);
            m1 = m;
            m = 0;
        }
        if (m < 1.0)
        {
            m = 0;
        }
        // convert the number
        while (n > percision || m >= 0)
        {
            weight = pow(10.0, m);
            if (weight > 0 && !isinf(weight))
            {
                digit = (int)(floor(n / weight));
                n -= (digit * weight);
                *(c++) = '0' + digit;
            }
            if (m == 0 && n > 0)
                *(c++) = '.';
            m--;
        }
        if (useExp)
        {
            // convert the exponent
            *(c++) = 'e';
            if (m1 > 0)
            {
                *(c++) = '+';
            }
            else
            {
                *(c++) = '-';
                m1 = -m1;
            }
            m = 0;
            while (m1 > 0)
            {
                *(c++) = '0' + m1 % 10;
                m1 /= 10;
                m++;
            }
            c -= m;
            for (i = 0, j = m-1; i<j; i++, j--)
            {
                // swap without temporary
                c[i] ^= c[j];
                c[j] ^= c[i];
                c[i] ^= c[j];
            }
            c += m;
        }
        *(c) = '\0';
    }
    return s;
}


void EncogStrCatChar(char *base, char ch, size_t len)
{
    char temp[2];
    temp[0] = ch;
    temp[1] = 0;
    EncogStrCatStr(base,temp,len);
}

void EncogStrCatStr(char *base, char *str, size_t len )
{
    strncat(base,str,len);
}

void EncogStrCatDouble(char *base, double d, int decimals,size_t len)
{
    char temp[MAX_STR];
    dtoa(temp,d,decimals);
    EncogStrCatStr(base,temp,len);
}

void EncogStrCatInt(char *base, int i,size_t len)
{
    char temp[MAX_STR];
	sprintf(temp,"%i",i);
    EncogStrCatStr(base,temp,len);
}

void EncogStrCatNL(char *base, size_t len)
{
    EncogStrCatStr(base,"\n",len);
}

char *EncogUtilStrlwr(char *string)
{
 char *s;
 if (string)
 {
for (s = string; *s; ++s)
 *s = tolower((int)*s);
 }
 return string;
}

char *EncogUtilStrupr(char *string)
{
 char *s;
 if (string)
 {
for (s = string; *s; ++s)
 *s = toupper((int)*s);
 }
 return string;
}

int EncogUtilStrcmpi(char *s1, char *s2)
{
 int ret = 0;

  while (!(ret = tolower(*(unsigned char *) s1) - tolower(*(unsigned char *) s2)) && *s2) ++s1, ++s2;

  if (ret < 0)

    ret = -1;
  else if (ret > 0)

    ret = 1 ;

  return ret;

}