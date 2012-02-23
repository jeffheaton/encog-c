#include "encog.h"

static int _randomAlgorithm = RANDOM_RTL;
static unsigned long _x=123456789, _y=362436069, _z=521288629;

static unsigned long xorshf96(void) {          //period 2^96-1
unsigned long t;
    _x ^= _x << 16;
    _x ^= _x >> 5;
    _x ^= _x << 1;

   t = _x;
   _x = _y;
   _y = _z;
   _z = t ^ _x ^ _y;

  return _z;
}

void EncogRandomChooseAlgorithm(int a)
{
	_randomAlgorithm = a;
}

void EncogRandomSeed() 
{
	unsigned int iseed;
	iseed = (unsigned int)time(NULL);
	EncogRandomSeedSpecific(iseed);
}

void EncogRandomSeedSpecific(unsigned int iseed) 
{
	switch(_randomAlgorithm) 
	{
		case RANDOM_RTL:
			srand (iseed);
			break;
		case RANDOM_XORSHIFT:
			break;
	}
}

double EncogRandomDouble() 
{
	double result;

	#pragma omp critical 
	{

	switch(_randomAlgorithm) 
	{
		case RANDOM_RTL:			
			result = ((REAL)rand()/(REAL)RAND_MAX);
			break;
		case RANDOM_XORSHIFT:	
			result = ((REAL)xorshf96()/(REAL)RAND_MAX);
			break;
	}
	}
}
	