# cython: boundscheck=False
# cython: initializedcheck=False
# cython: wraparound=False

import numpy as np
cimport numpy as cnp

from dipy.core.interpolation cimport _trilinear_interpolation_iso
from libc.stdlib cimport malloc, free
from libc.math cimport floor

cdef extern from "stdlib.h" nogil:
    void *memset(void *ptr, int value, size_t num)


cdef class PeakDirectionGen:
    """Peak direction data container with trilinear interpolation support.

    This class stores peak directions as 3D vectors and peak values at each voxel,
    providing efficient trilinear interpolation for tracking algorithms.

    Parameters
    ----------
    peak_directions : ndarray, shape (X, Y, Z, npeaks, 3)
        Peak directions as 3D unit vectors at each voxel.
    peak_values : ndarray, shape (X, Y, Z, npeaks)
        Peak strength values at each voxel.
    """

    def __init__(self,
                 double[:, :, :, :, :] peak_directions,
                 double[:, :, :, :] peak_values):
        """Initialize PeakDirectionGen.

        Parameters
        ----------
        peak_directions : ndarray, shape (X, Y, Z, npeaks, 3)
            Peak directions as 3D unit vectors.
        peak_values : ndarray, shape (X, Y, Z, npeaks)
            Peak strength values (e.g., QA, GFA).
        """
        self.peak_dirs = np.asarray(peak_directions, dtype=float, order='C')
        self.peak_values = np.asarray(peak_values, dtype=float, order='C')

        # Validate shapes
        if self.peak_dirs.shape[0] != self.peak_values.shape[0] or \
           self.peak_dirs.shape[1] != self.peak_values.shape[1] or \
           self.peak_dirs.shape[2] != self.peak_values.shape[2] or \
           self.peak_dirs.shape[3] != self.peak_values.shape[3]:
            raise ValueError("peak_directions and peak_values must have matching spatial and peak dimensions")

        if self.peak_dirs.shape[4] != 3:
            raise ValueError("peak_directions must have shape (X, Y, Z, npeaks, 3)")

        self.npeaks = self.peak_values.shape[3]

        # Initialize voxel_size to ones (will be set properly if needed)
        self.voxel_size[0] = 1.0
        self.voxel_size[1] = 1.0    
        self.voxel_size[2] = 1.0

    cdef void get_peak_directions_c(self,
                                    double* point,
                                    double* interp_dirs,
                                    double* interp_values,
                                    double* interp_weights,
                                    int max_peaks) noexcept nogil:
        """Get interpolated peak directions and values at a point.

        Performs trilinear interpolation of peak directions and values from
        the 8 neighboring voxels. Returns data for all peaks from all neighbors.

        Parameters
        ----------
        point : double[3]
            Position in voxel coordinates.
        interp_dirs : double[8 * max_peaks * 3] (output)
            Interpolated peak directions from 8 neighbors.
            Layout: [neighbor0_peak0[3], neighbor0_peak1[3], ...,
                     neighbor1_peak0[3], ...]
        interp_values : double[8 * max_peaks] (output)
            Interpolated peak values from 8 neighbors.
            Layout: [neighbor0_peak0, neighbor0_peak1, ...,
                     neighbor1_peak0, ...]
        interp_weights : double[8] (output)
            Trilinear interpolation weights for each of 8 neighbors.
        max_peaks : int
            Maximum number of peaks per voxel (should match npeaks).
        """
        cdef:
            double w[8]  # Trilinear interpolation weights
            cnp.npy_intp index[24]  # Indices of 8 neighbors (3 coords each)
            cnp.npy_intp x, y, z
            cnp.npy_intp m, j, k
            cnp.npy_intp output_idx
            int npeaks_to_use

        # Get trilinear interpolation weights and neighbor indices
        _trilinear_interpolation_iso(point, <double*> w, <cnp.npy_intp*> index)

        # Determine how many peaks to actually use
        npeaks_to_use = max_peaks if max_peaks <= self.npeaks else self.npeaks

        # For each of the 8 neighboring voxels
        for m in range(8):
            # Get voxel coordinates for this neighbor
            x = index[m * 3 + 0]
            y = index[m * 3 + 1]
            z = index[m * 3 + 2]

            # Store interpolation weight for this neighbor
            interp_weights[m] = w[m]

            # Check bounds
            if (x < 0 or x >= self.peak_dirs.shape[0] or
                y < 0 or y >= self.peak_dirs.shape[1] or
                z < 0 or z >= self.peak_dirs.shape[2]):
                # Outside volume - set peaks to zero
                for j in range(npeaks_to_use):
                    output_idx = m * max_peaks + j
                    interp_values[output_idx] = 0.0
                    for k in range(3):
                        interp_dirs[output_idx * 3 + k] = 0.0
                continue

            # Extract peaks from this neighbor voxel
            for j in range(npeaks_to_use):
                output_idx = m * max_peaks + j

                # Copy peak value
                interp_values[output_idx] = self.peak_values[x, y, z, j]

                # Copy peak direction (3D vector)
                for k in range(3):
                    interp_dirs[output_idx * 3 + k] = self.peak_dirs[x, y, z, j, k]

        # If max_peaks > npeaks, zero out the extra entries
        if max_peaks > npeaks_to_use:
            for m in range(8):
                for j in range(npeaks_to_use, max_peaks):
                    output_idx = m * max_peaks + j
                    interp_values[output_idx] = 0.0
                    for k in range(3):
                        interp_dirs[output_idx * 3 + k] = 0.0
