#!/bin/zsh -x

set -e

pushd

cd ~/prg/numpy

CFLAGS=-fopenmp python2.7 setup.py build -j 8 

PYTHONPATH=/home/mazay/prg/numpy_install/lib/python2.7/site-packages python2.7 setup.py install --prefix ~/prg/numpy_install


cd ~/prg

PYTHONPATH=/home/mazay/prg/numpy_install/lib/python2.7/site-packages python2.7 -c 'import numpy as np'

prg1='from time import time; import numpy as np; N = 2**1; M = 2**1; K=2**1; L=2**1; a = np.ones((N, K, M)); b = np.ones((M, L, N)); print "-=-=-=-=-=-=-=-=-"; t0=time(); c = np.tensordot(a, b, ([0,2], [2,0])); print time()-t0; print c.shape; print np.linalg.norm(c)'

export OMP_NUM_THREADS=1

PYTHONPATH=/home/mazay/prg/numpy_install/lib/python2.7/site-packages  /usr/bin/time -v  python2.7  -c $prg1

/usr/bin/time -v  python2.7  -c $prg1

# exit 0

prg2='from time import time; import numpy as np; N = 2**10; M = 2**12; K=2**5; L=2**7; a = np.ones((N, K, M)); b = np.ones((M, L, N)); print "-=-=-=-=-=-=-=-=-"; t0=time(); c = np.tensordot(a, b, ([0,2], [2,0])); print time()-t0; print c.shape; print np.linalg.norm(c)'


export OMP_NUM_THREADS=4

PYTHONPATH=/home/mazay/prg/numpy_install/lib/python2.7/site-packages /usr/bin/time -v  python2.7  -c $prg2

/usr/bin/time -v  python2.7  -c $prg2


popd
