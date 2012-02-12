#include "encog.h"

void EncogFileWriteValueInt(FILE *fp, char *name, INT value)
{
	char buffer[MAX_STR];
	*buffer = 0;
	EncogStrCatInt(buffer,value,MAX_STR);
	fputs(name,fp);
	fputc('=',fp);
	fputs(buffer,fp);
	fputs("\n",fp);
}

void EncogFileWriteValueBoolean(FILE *fp, char *name, INT value)
{
	fputs(name,fp);
	fputc('=',fp);
	fputc(value?'T':'F',fp);
	fputs("\n",fp);
}

void EncogFileWriteValueIntArray(FILE *fp, char *name, INT *a, INT count)
{
	char buffer[MAX_STR];
	INT i;

	fputs(name,fp);
	fputc('=',fp);
	for(i=0;i<count;i++)
	{
		if( i>0 )
		{
			fputc(',',fp);
		}
		*buffer = 0;
		EncogStrCatInt(buffer,a[i],MAX_STR);
		fputs(buffer,fp);		
	}
	fputs("\n",fp);
}

void EncogFileWriteValueDouble(FILE *fp, char *name, REAL value)
{
	fputs(name,fp);
	fputc('=',fp);
	fprintf(fp,"%.20g",value);	
	fputs("\n",fp);
}

void EncogFileWriteValueDoubleArray(FILE *fp, char *name, REAL *a, INT count)
{
	INT i, lineCount;

	fputs(name,fp);
	fputc('=',fp);

	if( count>2048 ) {
		fprintf(fp,"##%i\n##double#%i\n",0,count);
		lineCount = 0;

		for(i=0;i<count;i++)
		{
			if( lineCount>0 )
			{
				fputc(',',fp);
			}
			fprintf(fp,"%.20g",a[i]);
			lineCount++;
			if( lineCount>2048 ) {
				lineCount = 0;
				fprintf(fp,"\n");
			}
		}
		fputs("\n##end\n",fp);
	} else {
		for(i=0;i<count;i++)
		{
			if( i>0 )
			{
				fputc(',',fp);
			}
			fprintf(fp,"%.20g",a[i]);	
		}
		fputs("\n",fp);
	}
}
