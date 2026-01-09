cimport numpy as cnp

cdef class PeakDirectionGen:
    cdef:
        double[:, :, :, :, :] peak_dirs    # [X, Y, Z, npeaks, 3]
        double[:, :, :, :] peak_values     # [X, Y, Z, npeaks]
        cnp.npy_intp npeaks
        double[3] voxel_size

    cdef void get_peak_directions_c(self,
                                    double* point,
                                    double* interp_dirs,
                                    double* interp_values,
                                    double* interp_weights,
                                    int max_peaks) noexcept nogil
