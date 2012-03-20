#include "encog.h"

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
	else
	{
		newNode->next = prev->next;
		prev->next = newNode;
	}
}

void *EncogHashGet(ENCOG_HASH *hashTable, char *key)
{
	char *key2;
	int hashCode, cp;
	ENCOG_HASH_NODE *newNode,*current, *prev;

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
	return NULL;
}

void EncogHashDump(ENCOG_HASH *hashTable)
{
	int i;
	ENCOG_HASH_NODE *current;

	for(i=0;i<hashTable->tableSize;i++)
	{
		if( hashTable->table[i] )
		{
			printf("Table entry #%i\n",i);
		
			current = hashTable->table[i];

			while(current)
			{
				printf("%s = %s\n",current->key,current->value);
				current = current->next;
			}
		}
	}
}

int EncogHashGetInteger(ENCOG_HASH *hashTable, char *key, int defaultValue)
{
	char *v;

	v = (char*)EncogHashGet(hashTable,key);
	if( v==NULL )
		return defaultValue;
	else
		return atoi(v);
}

float EncogHashGetFloat(ENCOG_HASH *hashTable, char *key, float defaultValue)
{
	char *v;

	v = (char*)EncogHashGet(hashTable,key);

	if( v==NULL)
		return defaultValue;
	else
		return atof(v);
}

