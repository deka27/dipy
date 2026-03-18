# cython: boundscheck=False
# cython: initializedcheck=False
# cython: wraparound=False
# cython: Nonecheck=False

from dipy.tracking.propspeed cimport (
    deterministic_propagator,
    eudx_propagator,
    probabilistic_propagator,
    parallel_transport_propagator,
    glide_propagator,
)
from dipy.tracking.utils import min_radius_curvature_from_angle

import numpy as np
cimport numpy as cnp


def generate_tracking_parameters(algo_name, *,
    int max_len=500, int min_len=2, double step_size=0.2, double[:] voxel_size,
    double max_angle=20, bint return_all=True, double pmf_threshold=0.1,
    double probe_length=0.5, double probe_radius=0, int probe_quality=3,
    int probe_count=1, double data_support_exponent=1, int random_seed=0,
    double peak_values_threshold=0.0239, double angle_threshold=60,
    double min_total_weight=0.5,
    double max_angle_min=20.0, double max_angle_max=60.0,
    double sharpness_power=4.0, int blend_mode=1,
    double sigmoid_steepness=6.0, double sigmoid_midpoint=0.5,
    double gm_transition_low=0.1, double gm_transition_high=0.7,
    double gm_relaxation_factor=1.3,
    double peak_floor_prob=0.0,
    object uncertainty_data=None, object gm_data=None,
    object dispersion_data=None, object num_fibers_data=None,
    object wm_data=None, object csf_data=None):

    cdef TrackerParameters params

    algo_name = algo_name.lower()

    if algo_name in ['deterministic', 'det']:
        params = TrackerParameters(max_len=max_len,
                                   min_len=min_len,
                                   step_size=step_size,
                                   voxel_size=voxel_size,
                                   pmf_threshold=pmf_threshold,
                                   max_angle=max_angle,
                                   random_seed=random_seed,
                                   return_all=return_all)
        params.set_tracker_c(deterministic_propagator)
        return params
    elif algo_name in ['probabilistic', 'prob']:
        params = TrackerParameters(max_len=max_len,
                                   min_len=min_len,
                                   step_size=step_size,
                                   voxel_size=voxel_size,
                                   pmf_threshold=pmf_threshold,
                                   max_angle=max_angle,
                                   random_seed=random_seed,
                                   return_all=return_all)
        params.set_tracker_c(probabilistic_propagator)
        return params
    elif algo_name == 'ptt':
        params = TrackerParameters(max_len=max_len,
                                   min_len=min_len,
                                   step_size=step_size,
                                   voxel_size=voxel_size,
                                   pmf_threshold=pmf_threshold,
                                   max_angle=max_angle,
                                   probe_length=probe_length,
                                   probe_radius=probe_radius,
                                   probe_quality=probe_quality,
                                   probe_count=probe_count,
                                   data_support_exponent=data_support_exponent,
                                   random_seed=random_seed,
                                   return_all=return_all)
        params.set_tracker_c(parallel_transport_propagator)
        return params
    elif algo_name == 'eudx':
        params = TrackerParameters(max_len=max_len,
                                   min_len=min_len,
                                   step_size=step_size,
                                   voxel_size=voxel_size,
                                   max_angle=angle_threshold,
                                   peak_values_threshold=peak_values_threshold,
                                   angle_threshold=angle_threshold,
                                   min_total_weight=min_total_weight,
                                   random_seed=random_seed,
                                   return_all=return_all)
        params.set_tracker_c(eudx_propagator)
        return params
    elif algo_name == 'glide':
        params = TrackerParameters(max_len=max_len,
                                   min_len=min_len,
                                   step_size=step_size,
                                   voxel_size=voxel_size,
                                   pmf_threshold=pmf_threshold,
                                   max_angle=max_angle_max,
                                   random_seed=random_seed,
                                   return_all=return_all)
        params.glide = GlideTrackerParameters(
            max_angle_min=max_angle_min,
            max_angle_max=max_angle_max,
            pmf_threshold=pmf_threshold,
            sharpness_power=sharpness_power,
            blend_mode=blend_mode,
            sigmoid_steepness=sigmoid_steepness,
            sigmoid_midpoint=sigmoid_midpoint,
            gm_transition_low=gm_transition_low,
            gm_transition_high=gm_transition_high,
            gm_relaxation_factor=gm_relaxation_factor,
            peak_floor_prob=peak_floor_prob,
            uncertainty_data=uncertainty_data,
            gm_data=gm_data,
            dispersion_data=dispersion_data,
            num_fibers_data=num_fibers_data,
            wm_data=wm_data,
            csf_data=csf_data,
        )
        params.set_tracker_c(glide_propagator)
        return params
    else:
        raise ValueError("Invalid algorithm name")


cdef class TrackerParameters:

    def __init__(self, max_len, min_len, step_size, voxel_size,
                 max_angle, return_all, pmf_threshold=None, probe_length=None,
                 probe_radius=None, probe_quality=None, probe_count=None,
                 data_support_exponent=None, random_seed=None,
                 peak_values_threshold=None, angle_threshold=None,
                 min_total_weight=None):
        cdef cnp.npy_intp i

        self.max_nbr_pts = int(max_len/step_size)
        self.min_nbr_pts = int(min_len/step_size)
        self.return_all = return_all
        self.random_seed = random_seed
        self.step_size = step_size
        self.average_voxel_size = 0
        for i in range(3):
            self.voxel_size[i] = voxel_size[i]
            self.inv_voxel_size[i] = 1. / voxel_size[i]
            self.average_voxel_size += voxel_size[i] / 3

        self.max_angle = np.deg2rad(max_angle)
        self.cos_similarity = np.cos(self.max_angle)
        self.max_curvature = 1 / min_radius_curvature_from_angle(
            self.max_angle,
            self.step_size / self.average_voxel_size)

        self.sh = None
        self.ptt = None
        self.eudx = None
        self.glide = None

        if pmf_threshold is not None:
            self.sh = ShTrackerParameters(pmf_threshold)

        if probe_length is not None and probe_radius is not None and probe_quality is not None and probe_count is not None and data_support_exponent is not None:
            self.ptt = ParallelTransportTrackerParameters(probe_length, probe_radius, probe_quality, probe_count, data_support_exponent)

        if (
            peak_values_threshold is not None
            and angle_threshold is not None
            and min_total_weight is not None
        ):
            self.eudx = EudxTrackerParameters(
                peak_values_threshold, angle_threshold, min_total_weight
            )

    cdef void set_tracker_c(self, func_ptr tracker) noexcept nogil:
        self.tracker = tracker


cdef class ShTrackerParameters:

    def __init__(self, pmf_threshold):
        self.pmf_threshold = pmf_threshold

cdef class ParallelTransportTrackerParameters:

    def __init__(self, double probe_length, double probe_radius,
                int probe_quality, int probe_count, double data_support_exponent):
        self.probe_length = probe_length
        self.probe_radius = probe_radius
        self.probe_quality = probe_quality
        self.probe_count = probe_count
        self.data_support_exponent = data_support_exponent

        self.probe_step_size = self.probe_length / (self.probe_quality - 1)
        self.probe_normalizer = 1.0 / (self.probe_quality * self.probe_count)
        self.k_small = 0.0001

        # Adaptively set in Trekker
        self.rejection_sampling_nbr_sample = 10
        self.rejection_sampling_max_try = 100


cdef class EudxTrackerParameters:
    """EUDX tracking parameters.

    Parameters
    ----------
    peak_values_threshold : double
        Minimum peak-values threshold. Peaks below this
        value are ignored. Default: 0.0239.
    angle_threshold : double
        Maximum angle (in degrees) between successive tracking steps.
        Default: 60.
    min_total_weight : double
        Minimum fraction of interpolation support required to continue
        tracking. Must be between 0 and 1. Default: 0.5 (50%).
    """

    def __init__(self, double peak_values_threshold, double angle_threshold,
                 double min_total_weight):
        self.peak_values_threshold = peak_values_threshold
        self.angle_threshold = angle_threshold
        self.min_total_weight = min_total_weight


cdef class GlideTrackerParameters:
    """GLIDE tracking parameters for uncertainty-adaptive hybrid tractography.

    Parameters
    ----------
    max_angle_min : double
        Minimum max angle (degrees) used when uncertainty is low (tight).
    max_angle_max : double
        Maximum max angle (degrees) used when uncertainty is high (relaxed).
    pmf_threshold : double
        PMF threshold for filtering weak directions.
    sharpness_power : double
        Base exponent for PMF sharpening.
    blend_mode : int
        Blending mode: 0=linear, 1=sigmoid, 2=step.
    sigmoid_steepness : double
        Steepness of sigmoid blending curve.
    sigmoid_midpoint : double
        Midpoint of sigmoid blending curve.
    gm_transition_low : double
        Lower bound of GM transition zone for gyral bias correction.
    gm_transition_high : double
        Upper bound of GM transition zone for gyral bias correction.
    gm_relaxation_factor : double
        Angular relaxation factor in GM transition zone.
    uncertainty_data : ndarray
        3D uncertainty map (float64, contiguous).
    gm_data : ndarray, optional
        3D GM fraction map (float64, contiguous).
    """

    def __init__(self, double max_angle_min, double max_angle_max,
                 double pmf_threshold, double sharpness_power,
                 int blend_mode, double sigmoid_steepness,
                 double sigmoid_midpoint, double gm_transition_low,
                 double gm_transition_high, double gm_relaxation_factor,
                 object uncertainty_data, double peak_floor_prob=0.0,
                 object gm_data=None,
                 object dispersion_data=None, object num_fibers_data=None,
                 object wm_data=None, object csf_data=None):
        self.cos_sim_min = np.cos(np.deg2rad(max_angle_max))
        self.cos_sim_max = np.cos(np.deg2rad(max_angle_min))
        self.pmf_threshold = pmf_threshold
        self.sharpness_power = sharpness_power
        self.blend_mode = blend_mode
        self.sigmoid_steepness = sigmoid_steepness
        self.sigmoid_midpoint = sigmoid_midpoint
        self.gm_transition_low = gm_transition_low
        self.gm_transition_high = gm_transition_high
        self.gm_relaxation_factor = gm_relaxation_factor
        self.peak_floor_prob = peak_floor_prob
        self.uncertainty_data = np.ascontiguousarray(
            uncertainty_data, dtype=np.float64)
        self.has_gm_map = gm_data is not None
        if gm_data is not None:
            self.gm_data = np.ascontiguousarray(gm_data, dtype=np.float64)

        self.has_dispersion_map = dispersion_data is not None
        if dispersion_data is not None:
            self.dispersion_data = np.ascontiguousarray(
                dispersion_data, dtype=np.float64)

        self.has_num_fibers_map = num_fibers_data is not None
        if num_fibers_data is not None:
            self.num_fibers_data = np.ascontiguousarray(
                num_fibers_data, dtype=np.float64)

        self.has_wm_map = wm_data is not None
        if wm_data is not None:
            self.wm_data = np.ascontiguousarray(wm_data, dtype=np.float64)

        self.has_csf_map = csf_data is not None
        if csf_data is not None:
            self.csf_data = np.ascontiguousarray(csf_data, dtype=np.float64)
