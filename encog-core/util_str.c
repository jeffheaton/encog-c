#include "encog.h"

int EncogStrPopLine(char *line, char *arg, int start, int len) 
{
	char *argPtr,*linePtr;
	int stop;

	argPtr = arg;
	linePtr = line + start;
	stop = len-1;

	while( *linePtr && (argPtr-arg)<stop )
	{
		if( *linePtr==',' ) 
		{
			break;
		} 
		else 
		{
			*(argPtr++) = *(linePtr++);
		}		
	}

	*argPtr = 0;

	return (linePtr - line)+1;
}

int EncogStrIsWhiteSpace(char ch)
{
	return ( ch==' ' || ch=='\t' || ch=='\n' || ch=='\r' );
}

void EncogStrTrim(char *line)
{
	char *ptr;
	/* first trim the end */

	ptr = line + strlen(line)-1;
	while( EncogStrIsWhiteSpace(*ptr) )
	{
		*ptr = 0;
		ptr--;
	}

	/* now trim the beginning */
	ptr = line;
	while( EncogStrIsWhiteSpace(*ptr) )
	{
		ptr++;
	}

	strncpy(ptr,line,strlen(line));
}

void EncogStrStripQuotes(char *line)
{
	char quoteChar = 0;

	if( *line=='\"' || *line=='\'' )
	{
		quoteChar = *line;
		strncpy(line,line+1,strlen(line));		
	}

	if( quoteChar && line[strlen(line)-1]==quoteChar )
	{
		line[strlen(line)-1] = 0;
	}
}

char *EncogStrParseNV(char *line, char *name, size_t len)
{
	char *p = strchr(line,'=');
	
	if( p==NULL )
	{
		*name = 0;
		return name;
	}

	*p = 0;

	strncpy(name,line,len);

	EncogStrTrim(name);

	/* Restore */
	*p = '=';
	p++;

	while( *p && (*p==' ' || *p=='\t') )
		p++;

	return p;
	
}


void EncogStrStripCRLF(char *str) 
{
	while( str[strlen(str)-1]=='\n' || str[strlen(str)-1]=='\r' )
	{
		str[strlen(str)-1] = 0;
	}
}

int EncogStrCountValues(char *line)
{
	int result;
	char *ptr;

	/* count and see how many numbers */
	result = 1;
	ptr = line;
	while(*ptr)
	{
		if( *ptr==',' )
		{			
			result++;
		}
		ptr++;
	}

	return result;
}

INT *EncogStrParseIntList(char *line)
{
	int size, index, i;
	INT *result;
	char arg[MAX_STR];

	size = EncogStrCountValues(line);	
	result = (INT*)malloc(size*sizeof(INT));

	index = 0;
	for(i = 0; i<size; i++ )
	{
		index = EncogStrPopLine(line, arg, index, sizeof(arg));
		result[i] = atoi(arg);
	}
	return result;
}

int EncogStrParseBoolean(char *line) 
{
	return( toupper(*line)=='T' );
}

REAL *EncogStrParseDoubleList(char *line)
{
	int size, index, i;
	REAL *result;
	char arg[MAX_STR];

	size = EncogStrCountValues(line);		
	result = (REAL*)malloc(size*sizeof(REAL));

	index = 0;
	for(i = 0; i<size; i++ )
	{
		index = EncogStrPopLine(line, arg, index, sizeof(arg));
		result[i] = (REAL)atof(arg);
	}
	return result;
}

