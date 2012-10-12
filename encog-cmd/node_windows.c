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
#ifdef _MSC_VER
#include <winsock2.h>
#include <windows.h>
#include <fcntl.h>
#include <string.h>
#include <stdlib.h>
#include <errno.h>
#include <stdio.h>
#include "encog-cmd.h"

#pragma comment(lib, "Ws2_32.lib")

DWORD WINAPI SocketHandler(void*);

void EncogNodeMain(int port) 
{
    unsigned short wVersionRequested;
    WSADATA wsaData;
    int err;
	int hsock;
    int * p_int ;
	struct sockaddr_in my_addr;
	    int* csock;
    struct sockaddr_in sadr;
    int    addr_size;

    wVersionRequested = MAKEWORD( 2, 2 );
    err = WSAStartup( wVersionRequested, &wsaData );
    if ( err != 0 || ( LOBYTE( wsaData.wVersion ) != 2 ||
            HIBYTE( wsaData.wVersion ) != 2 )) {
        fprintf(stderr, "Could not find useable sock dll %d\n",WSAGetLastError());
        goto FINISH;
    }
    
    hsock = socket(AF_INET, SOCK_STREAM, 0);
    if(hsock == -1){
        printf("Error initializing socket %d\n",WSAGetLastError());
        goto FINISH;
    }
    
    p_int = (int*)malloc(sizeof(int));
    *p_int = 1;
    if( (setsockopt(hsock, SOL_SOCKET, SO_REUSEADDR, (char*)p_int, sizeof(int)) == -1 )||
        (setsockopt(hsock, SOL_SOCKET, SO_KEEPALIVE, (char*)p_int, sizeof(int)) == -1 ) ){
        printf("Error setting options %d\n", WSAGetLastError());
        free(p_int);
        goto FINISH;
    }
    free(p_int);

    my_addr.sin_family = AF_INET ;
    my_addr.sin_port = htons(port);
    
    memset(&(my_addr.sin_zero), 0, 8);
    my_addr.sin_addr.s_addr = INADDR_ANY ;
    
    if( bind( hsock, (struct sockaddr*)&my_addr, sizeof(my_addr)) == -1 ){
        fprintf(stderr,"Error binding to socket, make sure nothing else is listening on this port %d\n",WSAGetLastError());
        goto FINISH;
    }
    if(listen( hsock, 10) == -1 ){
        fprintf(stderr, "Error listening %d\n",WSAGetLastError());
        goto FINISH;
    }
    
    //Now lets to the server stuff

    addr_size = sizeof(SOCKADDR);
    
    while(1){
        printf("waiting for a connection\n");
        csock = (int*)malloc(sizeof(int));
        
        if((*csock = accept( hsock, (SOCKADDR*)&sadr, &addr_size))!= INVALID_SOCKET ){
            printf("Received connection from %s",inet_ntoa(sadr.sin_addr));
            CreateThread(0,0,&SocketHandler, (void*)csock , 0,0);
        }
        else{
            fprintf(stderr, "Error accepting %d\n",WSAGetLastError());
        }
    }

FINISH:
;
}

DWORD WINAPI SocketHandler(void* lp){
    int *csock = (int*)lp;
    char buffer[1024];
    int buffer_len = 1024;
    int bytecount;
	int i;

    memset(buffer, 0, buffer_len);
    if((bytecount = recv(*csock, buffer, buffer_len, 0))==SOCKET_ERROR){
        fprintf(stderr, "Error receiving data %d\n", WSAGetLastError());
        goto FINISH;
    }
    printf("Received bytes %d\nReceived string \"%s\"\n", bytecount, buffer);
    strcat(buffer, " SERVER ECHO");

	for(i=0;i<bytecount;i++) {
		EncogNodeRecv(buffer[i]);
	}

    /*if((bytecount = send(*csock, buffer, strlen(buffer), 0))==SOCKET_ERROR){
        fprintf(stderr, "Error sending data %d\n", WSAGetLastError());
        goto FINISH;
    }*/
    
    printf("Sent bytes %d\n", bytecount);


FINISH:
    free(csock);
    return 0;
}
#endif