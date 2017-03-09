# hillcipher

Encryption - Hill Cipher

This script ingests ASCII text, then encrypts it and decrypts it.
Simple, right! Encryption and decryption is handled partially via
PyCUDA ReductionKernel OR direct CUDA matrix multiplication on the GPU.


Usage:
~$ python hillcipher.py <i>&lt;textfile&gt;</i> <i>&lt;dotprod&gt;</i>

<i>&lt;textfile&gt;</i> optional - encrypt/decrypt the contents of a text file.

<i>&lt;dotprod&gt;</i> optional - also encrypt/decrypt using this script's
(much slower) dot-product functionality, explained below.

<i>&lt;hamlet.txt&gt;</i> included in repo as sample <i>&lt;textfile&gt;</i> for encryption.


Current implementation only capable of encryption/decryption
with Hill Cipher.  Simplest possible cipher chosen for
practice purposes.  More advanced versions will include other
encryption algorithms, including AES and possibly stream cyphers
like RC4.

This script includes several methods for encryption and decryption
on the GPU - two, dot_prod_encrypt() and dot_prod_decrypt() are
much slower and utilize PyCUDA's ReductionKernel to MapReduce a
series of dot products, eventually encrypting and decrypting (slowly).

parallel_encrypt() directly multiplies entire matrices on the
GPU to encrypt and decrypt - this is much faster.
