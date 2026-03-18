from dipy.direction.pmf cimport PmfGen
from dipy.utils.fast_numpy cimport RNGState


cpdef enum TrackerStatus:
    SUCCESS = 1
    FAIL = -1

ctypedef TrackerStatus (*func_ptr)(double* point,
                                   double* direction,
                                   TrackerParameters params,
                                   double* stream_data,
                                   PmfGen pmf_gen,
                                   RNGState* rng) noexcept nogil

cdef class ParallelTransportTrackerParameters:
    cdef public double angular_separation
    cdef public double data_support_exponent
    cdef public double k_small
    cdef public int probe_count
    cdef public double probe_length
    cdef public double probe_normalizer
    cdef public int probe_quality
    cdef public double probe_radius
    cdef public double probe_step_size
    cdef public int rejection_sampling_max_try
    cdef public int rejection_sampling_nbr_sample

cdef class ShTrackerParameters:
    cdef public double pmf_threshold

cdef class EudxTrackerParameters:
    cdef public double peak_values_threshold
    cdef public double angle_threshold
    cdef public double min_total_weight

cdef class GlideTrackerParameters:
    cdef public double cos_sim_min
    cdef public double cos_sim_max
    cdef public double pmf_threshold
    cdef public double sharpness_power
    cdef public int blend_mode
    cdef public double sigmoid_steepness
    cdef public double sigmoid_midpoint
    cdef public bint has_gm_map
    cdef public double gm_transition_low
    cdef public double gm_transition_high
    cdef public double gm_relaxation_factor
    cdef public double[:,:,:] uncertainty_data
    cdef public double[:,:,:] gm_data
    cdef public bint has_dispersion_map
    cdef public double[:,:,:] dispersion_data
    cdef public bint has_num_fibers_map
    cdef public double[:,:,:] num_fibers_data
    cdef public bint has_wm_map
    cdef public double[:,:,:] wm_data
    cdef public bint has_csf_map
    cdef public double[:,:,:] csf_data
    cdef public double peak_floor_prob

cdef class TrackerParameters:
    cdef func_ptr tracker

    cdef public double cos_similarity
    cdef public double max_angle
    cdef public double max_curvature
    cdef public int max_nbr_pts
    cdef public int min_nbr_pts
    cdef public int random_seed
    cdef public double step_size
    cdef public double average_voxel_size
    cdef public double[3] voxel_size
    cdef public double[3] inv_voxel_size
    cdef public bint return_all

    cdef public ShTrackerParameters sh
    cdef public ParallelTransportTrackerParameters ptt
    cdef public EudxTrackerParameters eudx
    cdef public GlideTrackerParameters glide

    cdef void set_tracker_c(self, func_ptr tracker) noexcept nogil
