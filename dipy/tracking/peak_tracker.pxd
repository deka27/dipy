# cython: boundscheck=False
# cython: initializedcheck=False
# cython: wraparound=False
# cython: Nonecheck=False

# Peak-based tracking data structures and declarations.
#
# This module provides the infrastructure for parallel tracking algorithms
# that work with discrete peaks (like EuDX) rather than PMF distributions.

cimport numpy as cnp


# Structure to hold peak-based tracking data for EuDX algorithm.
# This struct encapsulates all the data needed for peak-based direction
# propagation, allowing it to be passed efficiently to parallel tracking
# functions.
cdef struct PeakData:
    double* qa              # QA values, shape (X, Y, Z, Npeaks)
    double* ind             # Peak indices into odf_vertices, shape (X, Y, Z, Npeaks)
    double* odf_vertices    # Sphere vertices, shape (N_vertices, 3)
    cnp.npy_intp* qa_shape      # Shape of QA array [X, Y, Z, Npeaks]
    cnp.npy_intp* qa_strides    # Memory strides for QA array in bytes
    double qa_thr           # QA threshold - peaks below this are ignored
    double ang_thr          # Angular threshold in degrees
    double total_weight     # Minimum total interpolation weight (typically 0.5)
