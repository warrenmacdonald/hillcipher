#!/usr/bin/env python

"""
Encryption - Hill Cipher

This script ingests ASCII text, then encrypts it and decrypts it.
Simple, right? Encryption and decryption is handled partially via
PyCUDA ReductionKernel OR direct CUDA matrix multiplication on the GPU.


Usage:
~$ python hillcipher.py <textfile> <dotprod>
"""

import string
import sys
import os
import numpy as np
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
from pycuda import driver
from pycuda import compiler
from pycuda.reduction import ReductionKernel

class ASCII(object):
    """ Constant val.  Only 100 string.printable chars. """
    CHARS = 100
    def __setattr__(self, *_):
        pass
ASCII = ASCII()

class SecretKey(object):
    """
    SecretKey for Hill's Cipher encryption is a square matrix.
    Decryption requires its inverse.

    SecretKey class member vars are secret key square matrix and inverse.
    Arbitrary 4x4 invertible matrix provided as default secret.
    If a matrix is provided, its inverse is calculated via LU factorization on GPU.
    """
    def __init__(self, secret=None):
        if secret is None:
            self.secret = [[8, 6, 9, 5],
                           [6, 9, 5, 10],
                           [5, 8, 4, 9],
                           [10, 6, 11, 4]]
            self.secret_inverse = [[-3, 20, -21, 1],
                                   [2, -41, 44, 1],
                                   [2, -6, 6, -1],
                                   [-1, 28, -30, -1]]
        else:
            try:
                self.secret_inverse = matrix_invert(secret)
            except:
                print("The secret key is not invertible.")
                sys.exit(0)


    def get_secret_key(self):
        return np.array(self.secret)

    def get_secret_inverse(self):
        return np.array(self.secret_inverse)

    def matrix_invert(self, secret=None):
        #TODO: Matrix inversion.  Resolve floating point issue.
        return self.secret_inverse

def parallel_encrypt(msg_chunk, matrix_dims, encrypt_or_decrypt):
    """parallel_encrypt() encrypts and decrypts messages.

    The msg_chunk is passed in as matrices of w/ rows and cols of
    size matrix_dims.  Each msg_chunk is encrypted by multiplying
    it with SecretKey.secret matrix.  Multiplication is handled
    via CUDA/GPU.

    """

    print(msg_chunk)

    if encrypt_or_decrypt == 'encrypt':
        secret = SecretKey().get_secret_key().astype(np.float32)
    else:
        secret = SecretKey().get_secret_inverse().astype(np.float32)
    print(secret)

    secret_gpu = gpuarray.to_gpu(secret)
    msg_gpu = gpuarray.to_gpu(msg_chunk)
    end_mtx_gpu = gpuarray.empty((matrix_dims, matrix_dims), np.float32)

    kernel_code_template = """
    #include <stdio.h>
    __global__ void mtxEncrypt(float *secretKey, float *msg, float *result) {
        int tx = threadIdx.x;
        int ty = threadIdx.y;
        float Pvalue = 0;

        for (int k = 0; k < %(MATRIX_SIZE)s; ++k) {
            float secretElement = secretKey[ty * %(MATRIX_SIZE)s + k];
            float msgElement = msg[k *  %(MATRIX_SIZE)s + tx];
            Pvalue  += secretElement * msgElement;
      }
        result[ty * %(MATRIX_SIZE)s + tx] = Pvalue;
    }
    """

    MATRIX_SIZE = matrix_dims
    kernel_code = kernel_code_template % {
        'MATRIX_SIZE': MATRIX_SIZE
        }
    mod = compiler.SourceModule(kernel_code)
    matrixmul = mod.get_function("mtxEncrypt")
    matrixmul(secret_gpu, msg_gpu, end_mtx_gpu, block=(matrix_dims, matrix_dims, 1))
    return np.array(end_mtx_gpu.get()%ASCII.CHARS)

def dot_prod_encrypt(vector, numbers_dict, matrix_dims):
    """Encrypts using ReductionKernel w/ simple MapReduce from PyCUDA.

    Dot product calculated sequentially, not in parallel.
    This is not efficient but the method remains available.


    Returns: encrypted message as np.ndarray
    """

    msg = ''
    secret = SecretKey().get_secret_key().astype(np.float32)
    print('vectors: ', matrix_dims)
    new_mtx = np.zeros(shape=(matrix_dims, matrix_dims)).astype(np.float32)
    krnl = ReductionKernel(
        np.float32,
        neutral="0",
        reduce_expr="a+b",
        map_expr="x[i]*y[i]",
        arguments="float *x, float *y"
        )  
    for k in range(0, len(vector)):
        for i in range(0, len(secret)):
            gsecret = gpuarray.to_gpu(secret[i])
            gi = gpuarray.to_gpu(vector[k])
            dot_prod = krnl(gsecret, gi).get() % ASCII.CHARS
            msg += numbers_dict[dot_prod]
            new_mtx[k][i] = int(dot_prod)
    return new_mtx

def dot_prod_decrypt(vector, numbers_dict, matrix_dims):
    """Decrypts using ReductionKernel w/ simple MapReduce from PyCUDA.

    Dot product calculated sequentially, not in parallel.
    This is not efficient but the method remains available.

    Returns: Decrypted matrix.
    """

    msg = ''
    secretinverse = SecretKey().get_secret_inverse().astype(np.float32)
    decrypted_mtx = np.zeros(shape=(matrix_dims, matrix_dims)).astype(np.float32)
    krnl = ReductionKernel(
        np.float32,
        neutral="0",
        reduce_expr="a+b",
        map_expr="x[i]*y[i]",
        arguments="float *x, float *y"
        )
    for k in range(0, len(vector)):
        for i in range(0, len(secretinverse)):
            gsecret = gpuarray.to_gpu(secretinverse[i])
            gi = gpuarray.to_gpu(vector[k])
            dot_prod = krnl(gsecret, gi).get() % ASCII.CHARS
            msg += numbers_dict[dot_prod]
            decrypted_mtx[k][i] = int(dot_prod)
    return decrypted_mtx

def msg_matrix(message, matrix_dims, ascii_dict):
    """Split message into square matrices, encode each string.

    Return list of matrices.
    """
    msg_mtx_size = matrix_dims*matrix_dims
    msg_vector = []
    msg_to_nums = []

    ### ASCII encode message, split into multiple square matrices. ###
    for i in message:
        msg_to_nums.append(ascii_dict[i])
    msg_split = [msg_to_nums[i:i+msg_mtx_size] for i in range(0, len(msg_to_nums), msg_mtx_size)]
    for i in range(0, len(msg_split)):
        msg_vector.append(np.zeros(shape=(matrix_dims, matrix_dims)).astype(np.float32))
    for i in range(0, len(msg_split)):
        num = 0
        for j in range(0, matrix_dims):
            for k in range(0, matrix_dims):
                msg_vector[i][j][k] = msg_split[i][num]
                num += 1
    print("Message:", message)
    print("size: ", msg_mtx_size, matrix_dims)
    return msg_vector

def prnt_msg(result, numbers_dict):
    result_text = ''
    for i in result:
        for j in i:
            for k in j:
                result_text += numbers_dict[int(k)]
    print(result_text)

def main():
    """
    Main function.  Message is ASCII-encoded and stored in a list
    of square matrices.  Each matrix is the same size as the
    secretkey.secret matrix.  By default this is 4x4, but
    the user can provide in argv[].  The message may be padded
    with ending white space if necessary to fill the final
    matrix.

    The list of matrices is then encrypted with parallel_encrypt() and
    appendedto a list until finally the entire message (plus
    potential white space to ensure all matrices are square).  The 
    encrypted matrices are provided as an argument to the 
    parallel_encrypt() method once again (this time with 'decrypt' 
    toggle), and processed with the secret inverse.

    """

    if len(sys.argv) > 1:
        if os.path.isfile(sys.argv[1]):
            msg_file = open(sys.argv[1], 'r')
            message = msg_file.read()
            msg_file.close()
            #message = open(sys.argv[1], 'r').read()
        else:
            message = sys.argv[1]
    else:
        message = 'All work and no play makes Jack a dull boy.'
    use_dot_prod = False
    if len(sys.argv) > 2 and sys.argv[2] == 'dotprod':
        use_dot_prod = True

    ### Pad the end of the message with blank spaces. ###
    ### Matrices all need to share square dimensions. ###
    matrix_dims = len(SecretKey().get_secret_key())
    if len(message) % (matrix_dims*matrix_dims) != 0:
        message += ' '*(matrix_dims*matrix_dims - len(message)%(matrix_dims*matrix_dims))

    ### Build {ASCII : NUM} and {NUM : ASCII} mappings. ###
    ### Note: This is not really ASCII, per se, just printable chars. ###
    ascii_dict = dict(zip(string.printable, range(0, ASCII.CHARS)))
    numbers_dict = {y: x for x, y in ascii_dict.items()}

    msg_vector = msg_matrix(message, matrix_dims, ascii_dict)

    print("\n\n\nEncrypting... \n\n\n")
    encrypted_msg_mtx = []
    for msg_chunk in msg_vector:
        encrypted_msg_mtx.append(
            parallel_encrypt(
                msg_chunk,
                matrix_dims,
                'encrypt'
                )
            )
    print("\n\n\nEncrypted msg matrix: ", encrypted_msg_mtx)

    prnt_msg(encrypted_msg_mtx, numbers_dict)

    print('\n\n\nDecrypting...\n\n\n')
    result = []
    for encrypted_msg_chunk in encrypted_msg_mtx:
        result.append(
            parallel_encrypt(
                encrypted_msg_chunk,
                matrix_dims,
                'decrypt'
                )
            )
    print("\n\n\nDecrypted msg:")
    prnt_msg(result, numbers_dict)

    if use_dot_prod:
        ### Encrypt ###
        dotprode_mtx = []
        for msg_chunk in msg_vector:
            dotprode_mtx.append(
                dot_prod_encrypt(
                    msg_chunk,
                    numbers_dict,
                    matrix_dims
                    )
                )
        prnt_msg(dotprode_mtx, numbers_dict)

        ### Decrypt ###
        dotprodd_mtx = []
        for encrypted_msg_chunk in dotprode_mtx:
            dotprodd_mtx.append(
                dot_prod_decrypt(
                    encrypted_msg_chunk,
                    numbers_dict,
                    matrix_dims
                    )
                )
        prnt_msg(dotprodd_mtx, numbers_dict)

if __name__ == '__main__':
    main()
