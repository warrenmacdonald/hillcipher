#!/bin/bash

# don't forget to set CMakeLists.txt to use arch 30 for G2 instances (different for P2)
sudo yum install libmpc-devel mpfr-devel gmp-devel

cd ~/Downloads
curl ftp://ftp.mirrorservice.org/sites/sourceware.org/pub/gcc/releases/gcc-4.9.2/gcc-4.9.2.tar.bz2 -O
tar xvfj gcc-4.9.2.tar.bz2

cd gcc-4.9.2
./configure --disable-multilib --enable-languages=c,c++
make -j 4
make install
