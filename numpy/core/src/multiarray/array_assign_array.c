/*
 * This file implements assignment from an ndarray to another ndarray.
 *
 * Written by Mark Wiebe (mwwiebe@gmail.com)
 * Copyright (c) 2011 by Enthought, Inc.
 *
 * See LICENSE.txt for the license.
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE
#include <numpy/ndarraytypes.h>

#include "npy_config.h"
#include "npy_pycompat.h"

#include "convert_datatype.h"
#include "methods.h"
#include "shape.h"
#include "lowlevel_strided_loops.h"

#include "array_assign.h"

#include "omp.h"

#define NPY_MAXTHREADS 128

npy_intp init_parallel_section(int ndim, npy_intp *shape, npy_intp *coord,  char **dataA, npy_intp *stridesA, char ** dataB, npy_intp *stridesB)
{
    int thread_num = omp_get_thread_num();
    int thread_cnt = omp_get_num_threads();
    npy_intp volume[ndim];
    volume[0] = 1; // not shape[0], volume is meassured in 0-th dimension blocks!
    for (int idim=1; idim < ndim; ++idim) {
		volume[idim] = volume[idim-1] * shape[idim];
		//printf("vol[%d] = %d\n", idim, volume[idim]);	fflush(stdout);
    }

    if (volume[ndim-1] == 0) {
		if (thread_num == 0) {
			memset(coord, 0, ndim * sizeof(coord[0]));
			return 1;
		} else {
			return 0;
		}
    }

    npy_intp cnt_per_thread = volume[ndim-1] / thread_cnt;
    npy_intp cnt_per_thread_remainder = volume[ndim-1] % thread_cnt;
    npy_intp current = cnt_per_thread * thread_num;
    npy_intp final = cnt_per_thread * (1 + thread_num);
    if (cnt_per_thread_remainder > thread_num) {
    	current += thread_num;
    	final += 1 + thread_num;
    } else {
    	current += cnt_per_thread_remainder;
    	final += cnt_per_thread_remainder;    	
    }
    
    if (thread_num == thread_cnt-1) {
		final = volume[ndim-1];
    }
    npy_intp cnt = final-current;
    //printf("%d ) vol = %d cur = %d cnt = %d fin = %d\n", thread_num, volume[ndim-1], current, cnt, final);    fflush(stdout);

    npy_intp cur_stripped = current;
    for (int idim=ndim-1; idim > 0; --idim) {
		coord[idim] = cur_stripped / volume[idim-1];
		cur_stripped = cur_stripped % volume[idim-1];
		(*dataA) += stridesA[idim] * coord[idim];
		(*dataB) += stridesB[idim] * coord[idim];
		// printf("%d ) coord[%d] = %d\n", thread_num, idim, coord[idim]);	fflush(stdout);
    }
    coord[0] = 0;
    //printf("%d ) %d ] dataA = %p dataB = %p\n",thread_num,  final-current, *dataA, *dataB);  fflush(stdout);
    return cnt;
}

// /* #pragma omp parallel private (idim, coord, dataA, dataB) */ 

// #define OMP_PARA_TWO _Pragma("omp parallel private ((idim), (coord), (dataA), (dataB))")

#define STR(x) #x
#define STRINGIFY(x) STR(x) 
#define CONCATENATE(X,Y1,Y2,Y3) X ( Y1 , Y2 , Y3 )

#define OMP_PARA_TWO( coord,  dataA,  dataB) \
  _Pragma( STRINGIFY( CONCATENATE(omp parallel firstprivate, coord, dataA, dataB) ) )




/* Start raw iteration */
#define NPY_RAW_PAR_ITER_TWO_START(idim, ndim, coord, shape, \
				dataA, stridesA, dataB, stridesB) \
	{ \
	OMP_PARA_TWO(coord, dataA, dataB) \
	{ \
	    npy_intp cnt = init_parallel_section((ndim), (shape), (coord), &(dataA), (stridesA), &(dataB), (stridesB)); \
	    /* printf("[ cnt = %d ] dataA = %p dataB = %p\n", cnt, dataA, dataB); */ \
	    while (cnt > 0) { 

/* Increment to the next n-dimensional coordinate for two raw arrays */
#define NPY_RAW_PAR_ITER_TWO_NEXT(idim, ndim, coord, shape, \
                              dataA, stridesA, dataB, stridesB) \
            for ((idim) = 1; (idim) < (ndim); ++(idim)) { \
                if (++(coord)[idim] == (shape)[idim]) { \
                    (coord)[idim] = 0; \
                    (dataA) -= ((shape)[idim] - 1) * (stridesA)[idim]; \
                    (dataB) -= ((shape)[idim] - 1) * (stridesB)[idim]; \
                } \
                else { \
                    (dataA) += (stridesA)[idim]; \
                    (dataB) += (stridesB)[idim]; \
                    break; \
                } \
            } \
	    --cnt; \
        }  \
	}  \
	} 




/*
 * Assigns the array from 'src' to 'dst'. The strides must already have
 * been broadcast.
 *
 * Returns 0 on success, -1 on failure.
 */
NPY_NO_EXPORT int
raw_array_assign_array(int ndim, npy_intp *shape,
        PyArray_Descr *dst_dtype, char *dst_data, npy_intp *dst_strides,
        PyArray_Descr *src_dtype, char *src_data, npy_intp *src_strides)
{
    int idim;
    npy_intp shape_it[NPY_MAXDIMS];
    npy_intp dst_strides_it[NPY_MAXDIMS];
    npy_intp src_strides_it[NPY_MAXDIMS];
    npy_intp coord[NPY_MAXDIMS];

    PyArray_StridedUnaryOp *stransfer[NPY_MAXTHREADS];
    NpyAuxData *transferdata[NPY_MAXTHREADS];
    int aligned, needs_api = 0;
    npy_intp src_itemsize = src_dtype->elsize;
    int num_threads, thread_i;

    num_threads = omp_get_max_threads();

    NPY_BEGIN_THREADS_DEF;

    /* Check alignment */
    aligned = raw_array_is_aligned(ndim,
                        dst_data, dst_strides, dst_dtype->alignment) &&
              raw_array_is_aligned(ndim,
                        src_data, src_strides, src_dtype->alignment);

    /* Use raw iteration with no heap allocation */
    if (PyArray_PrepareTwoRawArrayIter(
                    ndim, shape,
                    dst_data, dst_strides,
                    src_data, src_strides,
                    &ndim, shape_it,
                    &dst_data, dst_strides_it,
                    &src_data, src_strides_it) < 0) {
        return -1;
    }

    /*
     * Overlap check for the 1D case. Higher dimensional arrays and
     * opposite strides cause a temporary copy before getting here.
     */
    if (ndim == 1 && src_data < dst_data &&
                src_data + shape_it[0] * src_strides_it[0] > dst_data) {
        src_data += (shape_it[0] - 1) * src_strides_it[0];
        dst_data += (shape_it[0] - 1) * dst_strides_it[0];
        src_strides_it[0] = -src_strides_it[0];
        dst_strides_it[0] = -dst_strides_it[0];
    }

    //printf("raw_array_assign_array: num_threads =  %d\n", num_threads);

    for(thread_i = 0; thread_i < num_threads; ++thread_i) {
        /* Get the function to do the casting */
		if (PyArray_GetDTypeTransferFunction(aligned,
			            src_strides_it[0], dst_strides_it[0],
			            src_dtype, dst_dtype,
			            0,
			            &stransfer[thread_i], &transferdata[thread_i],
			            &needs_api) != NPY_SUCCEED) {
			return -1;
		    }
    }


    if (!needs_api) {
        NPY_BEGIN_THREADS;
    }

    /*for(idim=0; idim < ndim; ++idim)
    {
        printf("raw_array_assign_array: shape_it[%d] = %d\n", idim, shape_it[idim]);
    }*/

	//printf("START dst_data = %p \t src_data = %p  \n",  dst_data, src_data);
	char *dst_data_1 = dst_data;
	char *src_data_1 = src_data;

#if 1

    NPY_RAW_PAR_ITER_TWO_START(idim, ndim, coord, shape_it, dst_data, dst_strides_it, src_data, src_strides_it) {
	    thread_i = omp_get_thread_num();
	    //printf("%d ) coord[1] = %p \t dst_data = %p \t src_data = %p \t %d \t %d  \n", thread_i, coord[1],  dst_data-dst_data_1, src_data-src_data_1 , dst_strides_it[0], src_strides_it[0]);
	    
        (stransfer[thread_i])(dst_data, dst_strides_it[0], src_data, src_strides_it[0],
	                shape_it[0], src_itemsize, transferdata[thread_i]);
	                
    } NPY_RAW_PAR_ITER_TWO_NEXT(idim, ndim, coord, shape_it,
                            dst_data, dst_strides_it,
                            src_data, src_strides_it)
#elif

    NPY_RAW_ITER_START(idim, ndim, coord, shape_it) {
	    thread_i = omp_get_thread_num();
	    //printf("%d ) coord[1] = %p \t dst_data = %p \t src_data = %p \t %d \t %d  \n", thread_i, coord[1],  dst_data-dst_data_1, src_data-src_data_1 , dst_strides_it[0], src_strides_it[0]);
	    
        (stransfer[thread_i])(dst_data, dst_strides_it[0], src_data, src_strides_it[0],
                    shape_it[0], src_itemsize, transferdata[thread_i]);
                    
    } NPY_RAW_ITER_TWO_NEXT(idim, ndim, coord, shape_it,
                            dst_data, dst_strides_it,
							src_data, src_strides_it);

#endif

    NPY_END_THREADS;

    for(thread_i = 0; thread_i < num_threads; ++thread_i) {
		NPY_AUXDATA_FREE(transferdata[thread_i]);
    }

    return (needs_api && PyErr_Occurred()) ? -1 : 0;
}

/*
 * Assigns the array from 'src' to 'dst, wherever the 'wheremask'
 * value is True. The strides must already have been broadcast.
 *
 * Returns 0 on success, -1 on failure.
 */
NPY_NO_EXPORT int
raw_array_wheremasked_assign_array(int ndim, npy_intp *shape,
        PyArray_Descr *dst_dtype, char *dst_data, npy_intp *dst_strides,
        PyArray_Descr *src_dtype, char *src_data, npy_intp *src_strides,
        PyArray_Descr *wheremask_dtype, char *wheremask_data,
        npy_intp *wheremask_strides)
{
    int idim;
    npy_intp shape_it[NPY_MAXDIMS];
    npy_intp dst_strides_it[NPY_MAXDIMS];
    npy_intp src_strides_it[NPY_MAXDIMS];
    npy_intp wheremask_strides_it[NPY_MAXDIMS];
    npy_intp coord[NPY_MAXDIMS];

    PyArray_MaskedStridedUnaryOp *stransfer = NULL;
    NpyAuxData *transferdata = NULL;
    int aligned, needs_api = 0;
    npy_intp src_itemsize = src_dtype->elsize;

    NPY_BEGIN_THREADS_DEF;

    /* Check alignment */
    aligned = raw_array_is_aligned(ndim,
                        dst_data, dst_strides, dst_dtype->alignment) &&
              raw_array_is_aligned(ndim,
                        src_data, src_strides, src_dtype->alignment);

    /* Use raw iteration with no heap allocation */
    if (PyArray_PrepareThreeRawArrayIter(
                    ndim, shape,
                    dst_data, dst_strides,
                    src_data, src_strides,
                    wheremask_data, wheremask_strides,
                    &ndim, shape_it,
                    &dst_data, dst_strides_it,
                    &src_data, src_strides_it,
                    &wheremask_data, wheremask_strides_it) < 0) {
        return -1;
    }

    /*
     * Overlap check for the 1D case. Higher dimensional arrays cause
     * a temporary copy before getting here.
     */
    if (ndim == 1 && src_data < dst_data &&
                src_data + shape_it[0] * src_strides_it[0] > dst_data) {
        src_data += (shape_it[0] - 1) * src_strides_it[0];
        dst_data += (shape_it[0] - 1) * dst_strides_it[0];
        wheremask_data += (shape_it[0] - 1) * wheremask_strides_it[0];
        src_strides_it[0] = -src_strides_it[0];
        dst_strides_it[0] = -dst_strides_it[0];
        wheremask_strides_it[0] = -wheremask_strides_it[0];
    }

    /* Get the function to do the casting */
    if (PyArray_GetMaskedDTypeTransferFunction(aligned,
                        src_strides_it[0],
                        dst_strides_it[0],
                        wheremask_strides_it[0],
                        src_dtype, dst_dtype, wheremask_dtype,
                        0,
                        &stransfer, &transferdata,
                        &needs_api) != NPY_SUCCEED) {
        return -1;
    }

    if (!needs_api) {
        NPY_BEGIN_THREADS;
    }

    NPY_RAW_ITER_START(idim, ndim, coord, shape_it) {
        /* Process the innermost dimension */
        stransfer(dst_data, dst_strides_it[0], src_data, src_strides_it[0],
                    (npy_bool *)wheremask_data, wheremask_strides_it[0],
                    shape_it[0], src_itemsize, transferdata);
    } NPY_RAW_ITER_THREE_NEXT(idim, ndim, coord, shape_it,
                            dst_data, dst_strides_it,
                            src_data, src_strides_it,
                            wheremask_data, wheremask_strides_it);

    NPY_END_THREADS;

    NPY_AUXDATA_FREE(transferdata);

    return (needs_api && PyErr_Occurred()) ? -1 : 0;
}

/*
 * An array assignment function for copying arrays, broadcasting 'src' into
 * 'dst'. This function makes a temporary copy of 'src' if 'src' and
 * 'dst' overlap, to be able to handle views of the same data with
 * different strides.
 *
 * dst: The destination array.
 * src: The source array.
 * wheremask: If non-NULL, a boolean mask specifying where to copy.
 * casting: An exception is raised if the copy violates this
 *          casting rule.
 *
 * Returns 0 on success, -1 on failure.
 */
NPY_NO_EXPORT int
PyArray_AssignArray(PyArrayObject *dst, PyArrayObject *src,
                    PyArrayObject *wheremask,
                    NPY_CASTING casting)
{
    int copied_src = 0;

    npy_intp src_strides[NPY_MAXDIMS];

    /* Use array_assign_scalar if 'src' NDIM is 0 */
    if (PyArray_NDIM(src) == 0) {
        return PyArray_AssignRawScalar(
                            dst, PyArray_DESCR(src), PyArray_DATA(src),
                            wheremask, casting);
    }

    /*
     * Performance fix for expressions like "a[1000:6000] += x".  In this
     * case, first an in-place add is done, followed by an assignment,
     * equivalently expressed like this:
     *
     *   tmp = a[1000:6000]   # Calls array_subscript in mapping.c
     *   np.add(tmp, x, tmp)
     *   a[1000:6000] = tmp   # Calls array_assign_subscript in mapping.c
     *
     * In the assignment the underlying data type, shape, strides, and
     * data pointers are identical, but src != dst because they are separately
     * generated slices.  By detecting this and skipping the redundant
     * copy of values to themselves, we potentially give a big speed boost.
     *
     * Note that we don't call EquivTypes, because usually the exact same
     * dtype object will appear, and we don't want to slow things down
     * with a complicated comparison.  The comparisons are ordered to
     * try and reject this with as little work as possible.
     */
    if (PyArray_DATA(src) == PyArray_DATA(dst) &&
                        PyArray_DESCR(src) == PyArray_DESCR(dst) &&
                        PyArray_NDIM(src) == PyArray_NDIM(dst) &&
                        PyArray_CompareLists(PyArray_DIMS(src),
                                             PyArray_DIMS(dst),
                                             PyArray_NDIM(src)) &&
                        PyArray_CompareLists(PyArray_STRIDES(src),
                                             PyArray_STRIDES(dst),
                                             PyArray_NDIM(src))) {
        /*printf("Redundant copy operation detected\n");*/
        return 0;
    }

    if (PyArray_FailUnlessWriteable(dst, "assignment destination") < 0) {
        goto fail;
    }

    /* Check the casting rule */
    if (!PyArray_CanCastTypeTo(PyArray_DESCR(src),
                                PyArray_DESCR(dst), casting)) {
        PyObject *errmsg;
        errmsg = PyUString_FromString("Cannot cast scalar from ");
        PyUString_ConcatAndDel(&errmsg,
                PyObject_Repr((PyObject *)PyArray_DESCR(src)));
        PyUString_ConcatAndDel(&errmsg,
                PyUString_FromString(" to "));
        PyUString_ConcatAndDel(&errmsg,
                PyObject_Repr((PyObject *)PyArray_DESCR(dst)));
        PyUString_ConcatAndDel(&errmsg,
                PyUString_FromFormat(" according to the rule %s",
                        npy_casting_to_string(casting)));
        PyErr_SetObject(PyExc_TypeError, errmsg);
        Py_DECREF(errmsg);
        goto fail;
    }

    /*
     * When ndim is 1 and the strides point in the same direction,
     * the lower-level inner loop handles copying
     * of overlapping data. For bigger ndim and opposite-strided 1D
     * data, we make a temporary copy of 'src' if 'src' and 'dst' overlap.'
     */
    if (((PyArray_NDIM(dst) == 1 && PyArray_NDIM(src) >= 1 &&
                    PyArray_STRIDES(dst)[0] *
                            PyArray_STRIDES(src)[PyArray_NDIM(src) - 1] < 0) ||
                    PyArray_NDIM(dst) > 1 || PyArray_HASFIELDS(dst)) &&
                    arrays_overlap(src, dst)) {
        PyArrayObject *tmp;

        /*
         * Allocate a temporary copy array.
         */
        tmp = (PyArrayObject *)PyArray_NewLikeArray(dst,
                                        NPY_KEEPORDER, NULL, 0);
        if (tmp == NULL) {
            goto fail;
        }

        if (PyArray_AssignArray(tmp, src, NULL, NPY_UNSAFE_CASTING) < 0) {
            Py_DECREF(tmp);
            goto fail;
        }

        src = tmp;
        copied_src = 1;
    }

    /* Broadcast 'src' to 'dst' for raw iteration */
    if (PyArray_NDIM(src) > PyArray_NDIM(dst)) {
        int ndim_tmp = PyArray_NDIM(src);
        npy_intp *src_shape_tmp = PyArray_DIMS(src);
        npy_intp *src_strides_tmp = PyArray_STRIDES(src);
        /*
         * As a special case for backwards compatibility, strip
         * away unit dimensions from the left of 'src'
         */
        while (ndim_tmp > PyArray_NDIM(dst) && src_shape_tmp[0] == 1) {
            --ndim_tmp;
            ++src_shape_tmp;
            ++src_strides_tmp;
        }

        if (broadcast_strides(PyArray_NDIM(dst), PyArray_DIMS(dst),
                    ndim_tmp, src_shape_tmp,
                    src_strides_tmp, "input array",
                    src_strides) < 0) {
            goto fail;
        }
    }
    else {
        if (broadcast_strides(PyArray_NDIM(dst), PyArray_DIMS(dst),
                    PyArray_NDIM(src), PyArray_DIMS(src),
                    PyArray_STRIDES(src), "input array",
                    src_strides) < 0) {
            goto fail;
        }
    }

    /* optimization: scalar boolean mask */
    if (wheremask != NULL &&
            PyArray_NDIM(wheremask) == 0 &&
            PyArray_DESCR(wheremask)->type_num == NPY_BOOL) {
        npy_bool value = *(npy_bool *)PyArray_DATA(wheremask);
        if (value) {
            /* where=True is the same as no where at all */
            wheremask = NULL;
        }
        else {
            /* where=False copies nothing */
            return 0;
        }
    }

    if (wheremask == NULL) {
        /* A straightforward value assignment */
        /* Do the assignment with raw array iteration */
        if (raw_array_assign_array(PyArray_NDIM(dst), PyArray_DIMS(dst),
                PyArray_DESCR(dst), PyArray_DATA(dst), PyArray_STRIDES(dst),
                PyArray_DESCR(src), PyArray_DATA(src), src_strides) < 0) {
            goto fail;
        }
    }
    else {
        npy_intp wheremask_strides[NPY_MAXDIMS];

        /* Broadcast the wheremask to 'dst' for raw iteration */
        if (broadcast_strides(PyArray_NDIM(dst), PyArray_DIMS(dst),
                    PyArray_NDIM(wheremask), PyArray_DIMS(wheremask),
                    PyArray_STRIDES(wheremask), "where mask",
                    wheremask_strides) < 0) {
            goto fail;
        }

        /* A straightforward where-masked assignment */
         /* Do the masked assignment with raw array iteration */
         if (raw_array_wheremasked_assign_array(
                 PyArray_NDIM(dst), PyArray_DIMS(dst),
                 PyArray_DESCR(dst), PyArray_DATA(dst), PyArray_STRIDES(dst),
                 PyArray_DESCR(src), PyArray_DATA(src), src_strides,
                 PyArray_DESCR(wheremask), PyArray_DATA(wheremask),
                         wheremask_strides) < 0) {
             goto fail;
         }
    }

    if (copied_src) {
        Py_DECREF(src);
    }
    return 0;

fail:
    if (copied_src) {
        Py_DECREF(src);
    }
    return -1;
}
