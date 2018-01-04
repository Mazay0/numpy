PYTHONPATH=/home/mazay/prg/numpy_install/lib/python2.7/site-packages OMP_NUM_THREADS=1 /usr/bin/time -v  python2.7  -c 'from time import time; import numpy as np; N = 2**10; M = 2**12; K=2**5; L=2**7; a = np.ones((N, K, M)); b = np.ones((M, L, N)); print "-=-=-=-=-=-=-=-=-"; t0=time(); c = np.tensordot(a, b, ([0,2], [2,0])); print time()-t0; print c.shape; print np.linalg.norm(c)'


OMP_NUM_THREADS=1 /usr/bin/time -v  python2.7  -c 'from time import time; import numpy as np; N = 2**10; M = 2**12; K=2**5; L=2**7; a = np.ones((N, K, M)); b = np.ones((M, L, N)); print "-=-=-=-=-=-=-=-=-"; t0=time(); c = np.tensordot(a, b, ([0,2], [2,0])); print time()-t0; print c.shape; print np.linalg.norm(c)'



CFLAGS=-fopenmp python2.7 setup.py build -j 8 install --prefix $HOME/prg/numpy_install

PYTHONPATH=/home/mazay/prg/numpy_install/lib/python2.7/site-packages CFLAGS=-fopenmp python2.7 setup.py build -j 8 install --prefix $HOME/prg/numpy_install


PYTHONPATH=/home/mazay/prg/numpy_install/lib/python2.7/site-packages CFLAGS=-fopenmp python2.7 -c 'import numpy as np'