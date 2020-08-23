
#include <cstdio>
#include <cmath>
#include <iostream>
#include "transform_kern.h"

#define MIN(a, b) (((a)<(b))?(a):(b))

using namespace std;

// begin: compute some common staffs
int H = 0, W = 0;
double distMap[1000000], centerAngleMap[1000000];

void preProc(int h, int w) {
    H = h, W = w;
    double cy = (double)(h - 1) / 2.0, cx = (double)(w - 1) / 2.0;
    int idx = 0;
    for (int i=0; i<h; ++i)
        for (int j=0; j<w; ++j) {
            distMap[idx] = sqrt(((double)i - cy) * ((double)i - cy) + ((double)j - cx) * ((double)j - cx));
            centerAngleMap[idx] = atan2(i - cy, j - cx);
            ++idx;
        }
}
// end: compute some common staffs


PyObject *excep;

static PyObject* rotation(PyObject *self, PyObject *args) {
    PyArrayObject *arr = (PyArrayObject*)PyTuple_GET_ITEM(args, 0);

    double angle = PyFloat_AsDouble(PyTuple_GET_ITEM(args, 1));

    if (!(((arr->flags) & NPY_ARRAY_C_CONTIGUOUS) && (PyArray_TYPE(arr) == NPY_DOUBLE) && (arr->nd == 3))) {
        PyErr_SetString(excep, "Kern error: the array is not C contiguous or the element is not double or the dimension is not 3. We cannot handle this.\n");
        return NULL;
    }

    int c = arr->dimensions[0], h = arr->dimensions[1], w = arr->dimensions[2];
    double *newImg = new double[c*h*w];
    memset(newImg, 0, sizeof(double) * (c * h * w));

    if ((c != 1) && (c != 3)) {
        PyErr_SetString(excep, "Kern error: This optimized kernel only supports 1-channel or 3-channel image.\n");
        return NULL;
    }

    int hw = h * w;

    if ((H != h) || (W != w)) {
        preProc(h, w);
    }

    double cy = (double)(h - 1) / 2.0, cx = (double)(w - 1) / 2.0;
    int idx = 0;
    for (int i=0; i<h; ++i)
        for (int j=0; j<w; ++j) {
            const double dist = distMap[idx];
            const double beta = centerAngleMap[idx] + angle * (double)M_PI / 180.0l;
            const double new_y = cy + dist * sin(beta), new_x = cx + dist * cos(beta);
            if ((new_y < 0) || (new_y > (h-1)) || (new_x < 0) || (new_x > (w - 1))) {
                ++idx;
                continue;
            }
            const int by = MIN(int(new_y), h-2), bx = MIN(int(new_x), w-2);
            const double ky2 = new_y - (double)by, kx2 = new_x - (double)bx;
            const double ky1 = 1.0l - ky2, kx1 = 1.0l - kx2;
            if (c == 1) {
                const int base = by * w + bx;
                newImg[idx] = ((double*)(PyArray_DATA(arr)))[base] * ky1 * kx1 + \
                    ((double*)(PyArray_DATA(arr)))[base + 1] * ky1 * kx2 + \
                    ((double*)(PyArray_DATA(arr)))[base + w] * ky2 * kx1 + \
                    ((double*)(PyArray_DATA(arr)))[base + w + 1] * ky2 * kx2;
            } else
            if (c == 3) {
                const int base1 = by * w + bx, base2 = base1 + hw, base3 = base2 + hw;
                newImg[idx] = ((double*)(PyArray_DATA(arr)))[base1] * ky1 * kx1 + \
                    ((double*)(PyArray_DATA(arr)))[base1 + 1] * ky1 * kx2 + \
                    ((double*)(PyArray_DATA(arr)))[base1 + w] * ky2 * kx1 + \
                    ((double*)(PyArray_DATA(arr)))[base1 + w + 1] * ky2 * kx2;
                newImg[hw + idx] = ((double*)(PyArray_DATA(arr)))[base2] * ky1 * kx1 + \
                    ((double*)(PyArray_DATA(arr)))[base2 + 1] * ky1 * kx2 + \
                    ((double*)(PyArray_DATA(arr)))[base2 + w] * ky2 * kx1 + \
                    ((double*)(PyArray_DATA(arr)))[base2 + w + 1] * ky2 * kx2;
                newImg[hw + hw + idx] = ((double*)(PyArray_DATA(arr)))[base3] * ky1 * kx1 + \
                    ((double*)(PyArray_DATA(arr)))[base3 + 1] * ky1 * kx2 + \
                    ((double*)(PyArray_DATA(arr)))[base3 + w] * ky2 * kx1 + \
                    ((double*)(PyArray_DATA(arr)))[base3 + w + 1] * ky2 * kx2;
            }
            ++idx;
        }

    PyObject *newArr = PyArray_SimpleNewFromData(3, arr->dimensions, NPY_DOUBLE, (void*)newImg);
    PyArray_ENABLEFLAGS((PyArrayObject*)newArr, NPY_ARRAY_OWNDATA);
    return Py_BuildValue("N", newArr);
}


static PyObject* scaling(PyObject *self, PyObject *args) {
    PyArrayObject *arr = (PyArrayObject*)PyTuple_GET_ITEM(args, 0);

    double s = PyFloat_AsDouble(PyTuple_GET_ITEM(args, 1));

    if (!(((arr->flags) & NPY_ARRAY_C_CONTIGUOUS) && (PyArray_TYPE(arr) == NPY_DOUBLE) && (arr->nd == 3))) {
        PyErr_SetString(excep, "Kern error: the array is not C contiguous or the element is not double or the dimension is not 3. We cannot handle this.\n");
        return NULL;
    }

    int c = arr->dimensions[0], h = arr->dimensions[1], w = arr->dimensions[2];
    double *newImg = new double[c*h*w];
    memset(newImg, 0, sizeof(double) * (c * h * w));

    if ((c != 1) && (c != 3)) {
        PyErr_SetString(excep, "Kern error: This optimized kernel only supports 1-channel or 3-channel image.\n");
        return NULL;
    }

    int hw = h * w;

    if ((H != h) || (W != w)) {
        preProc(h, w);
    }

    double cy = (double)(h - 1) / 2.0, cx = (double)(w - 1) / 2.0;
    int idx = 0;
    for (int i=0; i<h; ++i)
        for (int j=0; j<w; ++j) {
            ++idx;
        }

    PyErr_SetString(excep, "Kern error: Not implemented error.\n");
    return NULL;
}

// below are essential things for connecting Python and C++


static char module_docs[] = "TransformKern: simple C accelerated functions for rotation and scaling transformation for Python. "
        "Author: Linyi Li";

static PyMethodDef TransformKernClib_funcs[] = {
    {"rotation", (PyCFunction)rotation, METH_VARARGS, "Rotate the numpy image by given angle."},
    {"scaling", (PyCFunction)scaling, METH_VARARGS, "Scaling the numpy image by given ratio."},
    {NULL, NULL, 0, NULL}
};

static PyModuleDef moduleDef = {
        PyModuleDef_HEAD_INIT,
        "transform_kern",
        module_docs,
        -1,
        TransformKernClib_funcs,
        NULL,
        NULL,
        NULL,
        NULL
};

PyMODINIT_FUNC PyInit_transform_kern(void) {
    _import_array();
    PyObject *module = PyModule_Create(&moduleDef);
    if (module == NULL) {
        return NULL;
    }
    excep = PyErr_NewException("TransformKernel.Error", NULL, NULL);
    if (excep == NULL) {
        Py_DECREF(module);
        return NULL;
    }
    return module;
}

