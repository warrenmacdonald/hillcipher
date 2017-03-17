#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define ASCII_CHARS 128



void printrow(float *msg_v_list) {
    int i;
    for (i = 0; i < 16; i++) {
        printf("PRINTROW: %f\n", msg_v_list[i]);
    }

}

__global__ void mtxEncrypt(float *secretKey, float *msg, float *result, int matrix_dims) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    float Pvalue = 0;
    int i;
    for (i = 0; i < 16; i++) {
       // printf("PRINTROW: %f\n", msg[i]);
    }
    for (int k = 0; k < matrix_dims; ++k) {
        float secretElement = secretKey[ty * matrix_dims + k];
        float msgElement = msg[k * matrix_dims + tx];
        
        printf("\nMultiplying %f by %f: \n", secretElement, msgElement);
        Pvalue  += secretElement * msgElement;
    }
    
    printf("PValue: %f\n", Pvalue);
    result[ty * matrix_dims + tx] = (int)Pvalue%ASCII_CHARS;
    printf("result: %f\n", result[ty * matrix_dims + tx]);
}

void secretKey(float **SKey, float **invSKey, int matrix_dims) {
    int k,z;

    float xSKey[4][4] = {{8, 6, 9, 5},
                {6, 9, 5, 10},
                {5, 8, 4, 9},
                {10, 6, 11, 4}};
    float xinvSKey[4][4] = {{-3, 20, -21, 1},
                    {2, -41, 44, 1},
                    {2, -6, 6, -1},
                    {-1, 28, -30, -1}};

    int i, j;
    for (i = 0; i < 4; i++) {
        for (j = 0; j < 4; j++) {
            SKey[0][i*matrix_dims + j] = xSKey[i][j];
            invSKey[0][i*matrix_dims + j] = xinvSKey[i][j];
            printf("%f :  %f \n", i*matrix_dims + j, SKey[0][i*matrix_dims + j]);
        }
    }
}

void pad_msg(char *msg, int matrix_dims) {
    int i;
    unsigned int msg_len = strlen(msg);
    int matrix_size = matrix_dims*matrix_dims;
    int extra_chars = matrix_size - msg_len%matrix_size;

    char *space = (char *)malloc(extra_chars);
    for (i = 0; i < extra_chars; i++) {
        space[i] = ' ';
    }
    strcat(msg, space);
}

void encode_msg(char *msg, float *msg_vector, int matrix_dims) {
    int matrix_size = matrix_dims * matrix_dims;
    int i = 0, j;

    while (msg[i] != '\0') {
        msg_vector[i] = (int)(msg[i]);
        i++;
    }
}

int main(int argc, char *argv[]) {

    char *msg;
    if (argc > 1) {

        if (fopen(argv[1], "r")) {
            printf("reading file\n\n");
            FILE *msg_file;
            msg_file = fopen(argv[1], "r");
            fseek(msg_file, 0, SEEK_END);
            // ftell() gives current position in the stream
            long msg_file_size = ftell(msg_file);
            // rewind to beginning of file now that we have size
            fseek(msg_file, 0, SEEK_SET);
            // allocate memory for msg var, read file stream into memory
            char *msg_file_text = (char *)malloc(msg_file_size + 1);
            fread(msg_file_text, msg_file_size, 1, msg_file);
            fclose(msg_file);
            // printf("%s", msg_file_text);
            msg = msg_file_text;
        } else {
            msg = (char *)malloc(strlen(argv[1]) + 15);
            strcpy(msg, argv[1]);
        }
    } else {
        const char* jack_msg = "All work and no play makes Jack a dull boy.";
        msg = (char *) malloc(strlen(jack_msg));
        strcpy(msg, jack_msg);
    }
    char *ascii_dict = (char *)malloc(128);

    //printf("%s", msg);
    int matrix_dims = 4;
    int matrix_size = matrix_dims*matrix_dims;
    // int **SKey;


    // malloc() for 4 (int*) pointers to 4 rows, also malloc'd (int)
    int z;
    float **SKey = (float **) malloc(matrix_dims * sizeof(float *));
    for (z = 0; z < matrix_dims; z++) {
        SKey[z] = (float *)malloc(matrix_dims * sizeof(float));
    }
    float **invSKey = (float **)malloc(matrix_dims * sizeof(float *));
    for (z = 0; z < matrix_dims; z++) {
        invSKey[z] = (float *)malloc(matrix_dims * sizeof(float));
    }

    int msg_size = strlen(msg);
    printf("STRLEN:  %d     MESSAGE SIZE: %d", strlen(msg), msg_size);
    // populate secret and inverse secret keys
    secretKey(SKey, invSKey, matrix_dims);

    int i, j;
    printf("%s", msg);
    for (i = 0; i < 4; i++) {
        for (j = 0; j < 4; j++) {
            printf("Skey: %f\n", SKey[0][i*4 + j]);
        }
    }
    // pad message if not divisible by matrix_dims^2
    if (msg_size % matrix_size != 0) {
        pad_msg(msg, matrix_dims);
        msg_size = strlen(msg);
    }
    printf("%s", msg);

    float *msg_vector = (float *)malloc(strlen(msg)*sizeof(float));
    // message encoded, results stored in msg_vector
    encode_msg(msg, msg_vector, matrix_dims);
    printf("ENCODED!\n");

    float msg_vector_list[msg_size/matrix_size][matrix_size];
    for (i = 0; i < msg_size/matrix_size; i++) {
        printf("\n\nCount: %d %d\n", i, msg_size%matrix_size);
        for (j = 0; j < matrix_size; j++) {
            msg_vector_list[i][j] = msg_vector[i*matrix_size + j];
            printf("%f\n", msg_vector_list[i][j], "\n");
        }
    }
    
    
    int nBytes = matrix_size*sizeof(float);
    printf("1\n");
    float **results = (float **)malloc(msg_size * sizeof(float *));
    printf("2\n");
    for (z = 0; z < msg_size/matrix_size; z++) {
        results[z] = (float *)malloc(matrix_size * sizeof(float));
        memset(results[z], 0, nBytes);
    }
   printf("3\n");

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    cudaSetDevice(0);


    float *secretGpu, *msgGpu, *resultGpu;
    cudaMalloc((void **)&secretGpu, nBytes);


    printf("msg_size/matrix_size: %d \n", msg_size/matrix_size);
    printf("\nmsg_vector_list[1]: %d\n", *msg_vector_list[1]);


    cudaMalloc((void **)&resultGpu, nBytes);
    cudaMalloc((void **)&msgGpu, nBytes);


    cudaMemcpy(secretGpu, *SKey, nBytes, cudaMemcpyHostToDevice);
    
    dim3 block(matrix_dims, matrix_dims);
    dim3 grid((matrix_dims+block.x-1)/block.x, (matrix_dims+block.y-1)/block.y);

    for (i = 0; i < msg_size/matrix_size; i++) {
        printf("\nlist: %f", *msg_vector_list[i]);
        // printrow(msg_vector_list[i]); this was a sanity check.  i am sane.
        
        // this seems to be sending the same row over and over.  why?
        cudaMemcpy(msgGpu, msg_vector_list[i], nBytes, cudaMemcpyHostToDevice);
        mtxEncrypt<<<grid, block>>>(secretGpu, msgGpu, resultGpu, matrix_dims);
        cudaMemcpy(results[i], resultGpu, nBytes, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
    }

    //cudaFree(secretGpu);

    printf("\n\n5\n\n");
    for (i = 0; i < msg_size/matrix_size; i++) {
    //    printf("%d %f\n\n\n\n", i, results[i]);
        for (j = 0; j < matrix_size; j++) {
            printf("%c", char((int)results[i][j]));
        }
    }
    
    float **unEncrypted = (float **)malloc(3 * sizeof(float *));
    for (z = 0; z < msg_size/matrix_size; z++) {
        unEncrypted[z] = (float *)malloc(matrix_size * sizeof(float));
        memset(unEncrypted[z], 0, nBytes);
    }
     

    cudaMemcpy(secretGpu, *invSKey, nBytes, cudaMemcpyHostToDevice);
    for (i = 0; i < msg_size/matrix_size; i++) {
        printf("OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO\nOOOOOOOOOOOOO\nOOOOOOOOOO\n");
        cudaMemcpy(msgGpu, results[i], nBytes, cudaMemcpyHostToDevice);
        mtxEncrypt<<<grid, block>>>(secretGpu, msgGpu, resultGpu, matrix_dims);
        cudaMemcpy(unEncrypted[i], resultGpu, nBytes, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
    }
    printf("HELLO");
    for (i = 0; i < msg_size/matrix_size; i++) {
        for (j = 0; j < matrix_size; j++) {
            printf("%c", char((int)unEncrypted[i][j]));
        }
    }
    free(results);
    cudaFree(secretGpu);
    cudaFree(msgGpu);
    cudaFree(resultGpu);
    cudaDeviceReset();

    

}





