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


static double _evaluate (ENCOG_TRAIN_NM *nm, int thread, double x[] )
{
	int i, tid;
	float result;
	ENCOG_DATA *data;
	REAL *input, *ideal;
	REAL delta;
	unsigned int j;
	double errorSum;

	EncogNetworkImportWeights(nm->targetNetwork, x);
	data = nm->data;

	errorSum = 0;

	#pragma omp parallel for private(i,j,input,ideal, tid, delta) reduction(+:errorSum) default(shared)
	for(i=0; i<(int)nm->data->recordCount; i++)
	{
		tid = omp_get_thread_num();

		input = EncogDataGetInput(data,i);
		ideal = EncogDataGetIdeal(data,i);

		EncogNetworkCompute(nm->network[tid],input,NULL);
		for(j=0; j<nm->targetNetwork->outputCount; j++)
        {
			delta = nm->network[tid]->layerOutput[j] - ideal[j];
            errorSum+=delta*delta;
        }
	}

	result =  (float)(errorSum/(data->recordCount*data->idealCount));
	
	nm->error = nm->error;

	if( !nm->currentReport.stopRequested )
	{
		nm->currentReport.error = nm->error;
		nm->currentReport.iterations++;
		nm->reportTarget(&nm->currentReport);
	}
	return result;
}

static void _nelmin ( ENCOG_TRAIN_NM *nm, int thread, double start[], double xmin[] )
{
  REAL ccoeff = 0.5;
  REAL del;
  REAL dn;
  REAL dnn;
  REAL ecoeff = 2.0;
  REAL eps = 0.001;
  int i;
  int ihi;
  int ilo;
  int j;
  int jcount;
  int l;
  int nn;
  int n;
  REAL *p;
  REAL *p2star;
  REAL *pbar;
  REAL *pstar;
  REAL rcoeff = 1.0;
  REAL rq;
  REAL x;
  REAL *y;
  REAL y2star;
  REAL ylo;
  REAL ystar;
  REAL z;

  n = nm->n;
/*
  Check the input parameters.
*/
  if ( nm->reqmin <= 0.0 )
  {
    nm->ifault = 1;
    return;
  }

  if ( n < 1 )
  {
    nm->ifault = 1;
    return;
  }

  if ( nm->konvge < 1 )
  {
    nm->ifault = 1;
    return;
  }

  p = ( double * ) malloc ( n * ( n + 1 ) * sizeof ( double ) );
  pstar = ( double * ) malloc ( n * sizeof ( double ) );
  p2star = ( double * ) malloc ( n * sizeof ( double ) );
  pbar = ( double * ) malloc ( n * sizeof ( double ) );
  y = ( double * ) malloc ( ( n + 1 ) * sizeof ( double ) );

  nm->numres = 0;

  jcount = nm->konvge; 
  dn = ( double ) ( n );
  nn = n + 1;
  dnn = ( double ) ( nn );
  del = 1.0;
  rq = nm->reqmin * dn;
/*
  Initial or restarted loop.
*/
  for ( ; ; )
  {
    for ( i = 0; i < n; i++ )
    { 
      p[i+n*n] = start[i];
    }
    y[n] = _evaluate ( nm, thread, start );

    for ( j = 0; j < n; j++ )
    {
      x = start[j];
	  start[j] = start[j] + nm->step * del;
      for ( i = 0; i < n; i++ )
      {
        p[i+j*n] = start[i];
      }
      y[j] = _evaluate ( nm, thread, start );
      start[j] = x;
    }
/*                 
  The simplex construction is complete.
                    
  Find highest and lowest Y values.  YNEWLO = Y(IHI) indicates
  the vertex of the simplex to be replaced.
*/                
    ylo = y[0];
    ilo = 0;

    for ( i = 1; i < nn; i++ )
    {
      if ( y[i] < ylo )
      {
        ylo = y[i];
        ilo = i;
      }
    }
/*
  Inner loop.
*/
    for ( ; ; )
    {
		if ( nm->currentReport.stopRequested )
      {
        break;
      }
		nm->error = (float)y[0];
      ihi = 0;

      for ( i = 1; i < nn; i++ )
      {
		  if ( nm->error < y[i] )
        {
			nm->error = (float)y[i];
          ihi = i;
        }
      }
/*
  Calculate PBAR, the centroid of the simplex vertices
  excepting the vertex with Y value YNEWLO.
*/
      for ( i = 0; i < n; i++ )
      {
        z = 0.0;
        for ( j = 0; j < nn; j++ )
        { 
          z = z + p[i+j*n];
        }
        z = z - p[i+ihi*n];  
        pbar[i] = z / dn;
      }
/*
  Reflection through the centroid.
*/
      for ( i = 0; i < n; i++ )
      {
        pstar[i] = pbar[i] + rcoeff * ( pbar[i] - p[i+ihi*n] );
      }
      ystar = _evaluate ( nm, thread, pstar );
/*
  Successful reflection, so extension.
*/
      if ( ystar < ylo )
      {
        for ( i = 0; i < n; i++ )
        {
          p2star[i] = pbar[i] + ecoeff * ( pstar[i] - pbar[i] );
        }
        y2star = _evaluate ( nm, thread, p2star );
/*
  Check extension.
*/
        if ( ystar < y2star )
        {
          for ( i = 0; i < n; i++ )
          {
            p[i+ihi*n] = pstar[i];
          }
          y[ihi] = ystar;
        }
/*
  Retain extension or contraction.
*/
        else
        {
          for ( i = 0; i < n; i++ )
          {
            p[i+ihi*n] = p2star[i];
          }
          y[ihi] = y2star;
        }
      }
/*
  No extension.
*/
      else
      {
        l = 0;
        for ( i = 0; i < nn; i++ )
        {
          if ( ystar < y[i] )
          {
            l = l + 1;
          }
        }

        if ( 1 < l )
        {
          for ( i = 0; i < n; i++ )
          {
            p[i+ihi*n] = pstar[i];
          }
          y[ihi] = ystar;
        }
/*
  Contraction on the Y(IHI) side of the centroid.
*/
        else if ( l == 0 )
        {
          for ( i = 0; i < n; i++ )
          {
            p2star[i] = pbar[i] + ccoeff * ( p[i+ihi*n] - pbar[i] );
          }
          y2star = _evaluate ( nm, thread, p2star );
/*
  Contract the whole simplex.
*/
          if ( y[ihi] < y2star )
          {
            for ( j = 0; j < nn; j++ )
            {
              for ( i = 0; i < n; i++ )
              {
                p[i+j*n] = ( p[i+j*n] + p[i+ilo*n] ) * 0.5;
                xmin[i] = p[i+j*n];
              }
              y[j] = _evaluate ( nm, thread, xmin );
            }
            ylo = y[0];
            ilo = 0;

            for ( i = 1; i < nn; i++ )
            {
              if ( y[i] < ylo )
              {
                ylo = y[i];
                ilo = i;
              }
            }
            continue;
          }
/*
  Retain contraction.
*/
          else
          {
            for ( i = 0; i < n; i++ )
            {
              p[i+ihi*n] = p2star[i];
            }
            y[ihi] = y2star;
          }
        }
/*
  Contraction on the reflection side of the centroid.
*/
        else if ( l == 1 )
        {
          for ( i = 0; i < n; i++ )
          {
            p2star[i] = pbar[i] + ccoeff * ( pstar[i] - pbar[i] );
          }
          y2star = _evaluate ( nm, thread, p2star );
/*
  Retain reflection?
*/
          if ( y2star <= ystar )
          {
            for ( i = 0; i < n; i++ )
            {
              p[i+ihi*n] = p2star[i];
            }
            y[ihi] = y2star;
          }
          else
          {
            for ( i = 0; i < n; i++ )
            {
              p[i+ihi*n] = pstar[i];
            }
            y[ihi] = ystar;
          }
        }
      }
/*
  Check if YLO improved.
*/
      if ( y[ihi] < ylo )
      {
        ylo = y[ihi];
        ilo = ihi;
      }
      jcount = jcount - 1;

      if ( 0 < jcount )
      {
        continue;
      }
/*
  Check to see if minimum reached.
*/
	  if ( !nm->currentReport.stopRequested )
      {
        jcount = nm->konvge;

        z = 0.0;
        for ( i = 0; i < nn; i++ )
        {
          z = z + y[i];
        }
        x = z / dnn;

        z = 0.0;
        for ( i = 0; i < nn; i++ )
        {
          z = z + pow ( y[i] - x, 2 );
        }

        if ( z <= rq )
        {
          break;
        }
      }
    }
/*
  Factorial tests to check that YNEWLO is a local minimum.
*/
    for ( i = 0; i < n; i++ )
    {
      xmin[i] = p[i+ilo*n];
    }

	if ( nm->currentReport.stopRequested )
    {
      nm->ifault = 2;
      break;
    }

    nm->ifault = 0;

    for ( i = 0; i < n; i++ )
    {
		del = nm->step * eps;
      xmin[i] = xmin[i] + del;
      z = _evaluate ( nm,thread, xmin );
	  if ( z < nm->error )
      {
        nm->ifault = 2;
        break;
      }
      xmin[i] = xmin[i] - del - del;
      z = _evaluate ( nm,thread, xmin );
	  if ( z < nm->error )
      {
        nm->ifault = 2;
        break;
      }
      xmin[i] = xmin[i] + del;
    }

    if ( nm->ifault == 0 )
    {
      break;
    }

/*
  Restart the procedure.
*/
    for ( i = 0; i < n; i++ )
    {
      start[i] = xmin[i];
    }
    del = eps;
    nm->numres++;
  }


  if( !nm->currentReport.stopRequested )
  {
	nm->currentReport.error = nm->error;
	nm->currentReport.iterations++;
	nm->currentReport.stopRequested = 1;
	nm->reportTarget(&nm->currentReport);
  }

  free ( p );
  free ( pstar );
  free ( p2star );
  free ( pbar );
  free ( y );

  return;
}

ENCOG_TRAIN_NM *EncogTrainNMNew(ENCOG_NEURAL_NETWORK *network, ENCOG_DATA *data)
{
	ENCOG_TRAIN_NM *result;
	int maxThread, i;

	/* Clear out any previous errors */
	EncogErrorClear();
	
	maxThread = omp_get_max_threads();
	
	result = (ENCOG_TRAIN_NM *)EncogUtilAlloc(1,sizeof(ENCOG_TRAIN_NM));
	result->threadCount = maxThread;

	result->data = data;
	result->targetNetwork = network;	
	result->reportTarget = &EncogTrainStandardCallback;
	result->error = 1.0;
	result->network = (ENCOG_NEURAL_NETWORK**)EncogUtilAlloc(maxThread,sizeof(ENCOG_NEURAL_NETWORK*));
	result->step = EncogHashGetFloat(encogContext.config,PARAM_STEP,10.0);
	result->reqmin = EncogHashGetFloat(encogContext.config,PARAM_REQMIN, 1.0e-16f);
	result->konvge = EncogHashGetInteger(encogContext.config,PARAM_KONVERGE,100);
	result->ifault = 0;
	memset(&result->currentReport,0,sizeof(ENCOG_TRAINING_REPORT));

	result->n = result->targetNetwork->weightCount;
	result->step = 1;

	for(i=0;i<maxThread;i++) 
	{
		result->network[i] = (ENCOG_NEURAL_NETWORK*)EncogNetworkTransactionClone(network);
	}

	EncogObjectRegister(result, ENCOG_TYPE_NM);
	result->currentReport.trainer = (ENCOG_OBJECT*)result;

	return result;
}

float EncogTrainNMRun(ENCOG_TRAIN_NM *nm)
{
	int n;
	double *start;
	double *xmin;


	/* Clear out any previous errors */
	EncogErrorClear();

	nm->currentReport.iterations = 0;
	nm->currentReport.lastUpdate = 0;
	nm->currentReport.stopRequested = 0;
	nm->currentReport.trainingStarted = time(NULL);

	n = nm->targetNetwork->weightCount;
	start = (double*)EncogUtilDuplicateMemory(nm->targetNetwork->weights,n,sizeof(REAL));
	xmin = (double*)EncogUtilAlloc(n,sizeof(double));

	_nelmin ( nm, 0, start, xmin );

	nm->currentReport.error = nm->error;

    return nm->currentReport.error;
}

