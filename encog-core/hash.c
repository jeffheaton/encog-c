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
#include <string.h>

ENCOG_HASH *EncogHashNew(INT tableSize, INT ignoreCase)
{
	ENCOG_HASH *result = (ENCOG_HASH *)EncogUtilAlloc(1,sizeof(ENCOG_HASH));

	result->tableSize = tableSize;
	result->ignoreCase = ignoreCase;
	result->table = (ENCOG_HASH_NODE **)EncogUtilAlloc(tableSize,sizeof(ENCOG_HASH_NODE*));

	return result;
}

void EncogHashPut(ENCOG_HASH *hashTable, char *key, void *obj)
{
	char *key2;
	int hashCode;
	ENCOG_HASH_NODE *newNode,*current, *prev;

	prev = NULL;

	key2 = strdup(key);
	if( hashTable->ignoreCase ) 
	{
		EncogUtilStrlwr(key2);
	}

	hashCode = EncogUtilHash((unsigned char*)key2) % hashTable->tableSize;

	newNode = (ENCOG_HASH_NODE*)EncogUtilAlloc(1,sizeof(ENCOG_HASH_NODE));
	newNode->key = key2;
	newNode->hashCode = hashCode;
	newNode->value = obj;

	current = hashTable->table[hashCode];

	while( current!=NULL && strcmp(current->key,key2)<0 )
	{
		prev = current;
		current=current->next;
	}

	if( current == hashTable->table[hashCode] )
	{
		newNode->next = current;
		hashTable->table[hashCode] = newNode;
	}
	else if( prev!=NULL )
	{
		newNode->next = prev->next;
		prev->next = newNode;
	}
}

void *EncogHashGet(ENCOG_HASH *hashTable, char *key, void *defaultValue)
{
	char *key2;
	int hashCode, cp;
	ENCOG_HASH_NODE *current;

	key2 = strdup(key);
	if( hashTable->ignoreCase ) 
	{
		EncogUtilStrlwr(key2);
	}

	hashCode = EncogUtilHash((unsigned char*)key2) % hashTable->tableSize;

	current = hashTable->table[hashCode];

	while(current!=NULL) 
	{
		cp = strcmp(key2,current->key);
		if( !cp )
		{
			free(key2);
			return current->value;
		}
		else if( cp<0 )
		{
			break;
		}
		current = current->next;
	}

	free(key2);
	return defaultValue;
}

int EncogHashContains(ENCOG_HASH *hashTable, char *key)
{
	return EncogHashGet(hashTable, key, NULL)!=NULL;
}

void EncogHashDump(ENCOG_HASH *hashTable)
{
	INT i;
	ENCOG_HASH_NODE *current;

	for(i=0;i<hashTable->tableSize;i++)
	{
		if( hashTable->table[i] )
		{
			printf("Table entry #%i\n",i);
		
			current = hashTable->table[i];

			while(current)
			{
				printf("%s = %s\n", current->key, (char*)current->value);
				current = current->next;
			}
		}
	}
}

int EncogHashGetInteger(ENCOG_HASH *hashTable, char *key, int defaultValue)
{
	char *v;

	v = (char*)EncogHashGet(hashTable,key,NULL);
	if( v==NULL )
		return defaultValue;
	else
		return atoi(v);
}

float EncogHashGetFloat(ENCOG_HASH *hashTable, char *key, float defaultValue)
{
	char *v;

	v = (char*)EncogHashGet(hashTable,key,NULL);

	if( v==NULL)
		return defaultValue;
	else
		return (float)atof(v);
}

