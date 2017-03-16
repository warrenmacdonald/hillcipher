#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define ASCII 128

__global__ void mtxEncrypt(float *secretKey, float *msg, float *result, int matrix_size) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    float Pvalue = 0;

    for (int k = 0; k < matrix_size; ++k) {
        float secretElement = secretKey[ty * matrix_size + k];
        float msgElement = msg[k * matrix_size + tx];
        Pvalue  += secretElement * msgElement;
    }
    result[ty * matrix_size + tx] = Pvalue;
}

void secretKey(int **SKey, int **invSKey, int matrix_dims) {
    int k,z;
    //*SKey = malloc((matrix_dims*sizeof(*SKey)) + matrix_dims*matrix_dims*sizeof(int **));
    //*invSKey = malloc(matrix_dims*matrix_dims*sizeof(int *));

    int xSKey[4][4] = {{8, 6, 9, 5},
                {6, 9, 5, 10},
                {5, 8, 4, 9},
                {10, 6, 11, 4}};
    int xinvSKey[4][4] = {{-3, 20, -21, 1},
                    {2, -41, 44, 1},
                    {2, -6, 6, -1},
                    {-1, 28, -30, -1}};

    int i, j;
    for (i = 0; i < 4; i++) {
        for (j = 0; j < 4; j++) {
            SKey[0][i*matrix_dims + j] = xSKey[i][j];
            invSKey[0][i*matrix_dims + j] = xinvSKey[i][j];
            printf("%d:  %d        %d   %d\n", i*matrix_dims + j, SKey[0][i*matrix_dims + j], SKey[0][i*matrix_dims+j], xSKey[i][j]);
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

void encode_msg(char *msg, int *msg_vector, int matrix_dims) {
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
    int **SKey = (int **) malloc(matrix_dims * sizeof(int *));
    for (z = 0; z < matrix_dims; z++) {
        SKey[z] = (int *)malloc(matrix_dims * sizeof(int));
    }
    int **invSKey = (int **)malloc(matrix_dims * sizeof(int *));
    for (z = 0; z < matrix_dims; z++) {
        invSKey[z] = (int *)malloc(matrix_dims * sizeof(int));
    }

    int msg_size = strlen(msg);
    printf("STRLEN:  %d     MESSAGE SIZE: %d", strlen(msg), msg_size);
    // populate secret and inverse secret keys
    secretKey(SKey, invSKey, matrix_dims);

    int i, j;
    printf("%s", msg);
    for (i = 0; i < 4; i++) {
        for (j = 0; j < 4; j++) {
            printf("Skey: %d\n", SKey[0][i*4 + j]);
        }
    }
    // pad message if not divisible by matrix_dims^2
    if (msg_size % matrix_size != 0) {
        pad_msg(msg, matrix_dims);
        msg_size = strlen(msg);
    }
    printf("%s", msg);

    int *msg_vector = (int *)malloc(strlen(msg)*sizeof(int));
    // message encoded, results stored in msg_vector
    encode_msg(msg, msg_vector, matrix_dims);
    printf("ENCODED!\n");
    int msg_vector_list[strlen(msg)/matrix_size][matrix_size];
    for (i = 0; i < msg_size/matrix_size; i++) {
        printf("\n\nCount: %d %d\n", i, msg_size%matrix_size);
        for (j = 0; j < matrix_size; j++) {
            msg_vector_list[i][j] = msg_vector[i*matrix_size + j];
            printf("%d\n", msg_vector_list[i][j], "\n");
        }
    }
    

    int nBytes = matrix_size*sizeof(int);
    int *results = (int *)malloc(nBytes);
    memset(results, 0, nBytes);

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    cudaSetDevice(0);
    float *secretGpu, *msgGpu, *resultGpu;

    cudaMalloc((void **)&secretGpu, nBytes);
    cudaMalloc((void **)&msgGpu, nBytes);
    cudaMalloc((void **)&resultGpu, nBytes);

    cudaMemcpy(secretGpu, SKey, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(msgGpu, msg_vector_list[0], nBytes, cudaMemcpyHostToDevice);
    dim3 block(matrix_dims, matrix_dims);
    dim3 grid((matrix_dims+block.x-1)/block.x, (matrix_dims+block.y-1)/block.y);
    mtxEncrypt<<<grid, block>>>(secretGpu, msgGpu, resultGpu, matrix_size);

    cudaMemcpy(results, resultGpu, nBytes, cudaMemcpyDeviceToHost);
    for (i = 0; i < matrix_size; i++) {
        printf("%d %d\n", i, results[i]);

    }






}

