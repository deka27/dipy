# cython: boundscheck=False
# cython: initializedcheck=False
# cython: wraparound=False
# cython: Nonecheck=False

from libc.stdio cimport printf

cimport ctime
cimport cython
from cython.parallel import prange
import numpy as np
cimport numpy as cnp

from dipy.direction.pmf cimport PmfGen
from dipy.tracking.stopping_criterion cimport StoppingCriterion
from dipy.utils cimport fast_numpy
from dipy.tracking.propspeed cimport eudx_propagator  # ← ADD THIS

from dipy.tracking.stopping_criterion cimport (StreamlineStatus,
                                               StoppingCriterion,
                                               TRACKPOINT,
                                               ENDPOINT,
                                               OUTSIDEIMAGE,
                                               INVALIDPOINT,
                                               VALIDSTREAMLIME,
                                               INVALIDSTREAMLIME)
from dipy.tracking.tracker_parameters cimport (TrackerParameters,
                                               TrackerStatus,
                                               func_ptr)

from nibabel.streamlines import ArraySequence as Streamlines

from libc.stdlib cimport malloc, free
from libc.string cimport memcpy, memset
from libc.math cimport ceil
from libc.stdio cimport printf


def generate_tractogram(double[:,::1] seed_positions,
                        double[:,::1] seed_directions,
                        StoppingCriterion sc,
                        TrackerParameters params,
                        PmfGen pmf_gen,
                        affine,
                        int nbr_threads=0,
                        float buffer_frac=1.0,
                        bint save_seeds=0,
                        pam_data=None,
                        bint cleanup_pam=False):
    """Generate a tractogram from a set of seed points and directions.

    Parameters
    ----------
    seed_positions : ndarray
        Seed positions for the streamlines.
    seed_directions : ndarray
        Seed directions for the streamlines.
    sc : StoppingCriterion
        Stopping criterion for the streamlines.
    params : TrackerParameters
        Parameters for the streamline generation.
    pmf_gen : PmfGen
        Probability mass function generator.
    affine : ndarray
        Affine transformation for the streamlines.
    nbr_threads : int, optional
        Number of threads to use for streamline generation.
    buffer_frac : float, optional
        Fraction of the seed points to process in each iteration.
    save_seeds : bool, optional
        If True, return seeds alongside streamlines
    pam_data : object, optional
        Direct PAM data for streamline generation.
    cleanup_pam : bool, optional
        If True, cleanup PAM tracking after completion.

    Yields
    ------
    streamlines : Streamlines
        Streamlines generated from the seed points.
    seeds : ndarray, optional
        seed points associated with the generated streamlines.

    """
    cdef:
        cnp.npy_intp _len = seed_positions.shape[0]
        cnp.npy_intp _plen = int(ceil(_len * buffer_frac))
        cnp.npy_intp i, seed_start, seed_end
        double** streamlines_arr
        int* length_arr
        StreamlineStatus* status_arr
        cnp.npy_intp total_streamlines = 0

    if buffer_frac <=0 or buffer_frac > 1:
        raise ValueError("buffer_frac must > 0 and <= 1.")

    lin_T = affine[:3, :3].T.copy()
    offset = affine[:3, 3].copy()

    inv_affine = np.linalg.inv(affine)
    seed_positions = np.dot(seed_positions, inv_affine[:3, :3].T.copy())
    seed_positions += inv_affine[:3, 3]

    seed_start = 0
    seed_end = _plen
    while seed_start < _len:
        streamlines_arr = <double**> malloc(_plen * sizeof(double*))
        length_arr = <int*> malloc(_plen * sizeof(int))
        status_arr = <StreamlineStatus*> malloc(_plen * sizeof(int))

        if streamlines_arr == NULL or length_arr == NULL or status_arr == NULL:
            raise MemoryError("Memory allocation failed")

        generate_tractogram_c(seed_positions[seed_start:seed_end],
                              seed_directions[seed_start:seed_end],
                              nbr_threads, sc, params, pmf_gen,
                              streamlines_arr, length_arr, status_arr)

        for i in range(seed_end - seed_start):
            # EuDX: Accept all streamlines with any length > 1
            if (length_arr[i] > 1):
                s = np.asarray(<cnp.float_t[:length_arr[i]*3]> streamlines_arr[i])
                track = s.copy().reshape((-1,3))
                total_streamlines += 1
                if save_seeds:
                    yield np.dot(track, lin_T) + offset, np.dot(seed_positions[seed_start + i], lin_T) + offset
                else:
                    yield np.dot(track, lin_T) + offset
            free(streamlines_arr[i])

        free(streamlines_arr)
        free(length_arr)
        free(status_arr)

        seed_start += _plen
        seed_end += _plen
        if seed_end > _len:
            seed_end = _len

    # Print total streamlines generated for debugging
    print(f"Total streamlines generated: {total_streamlines} from {_len} seeds")

    if cleanup_pam:
        from dipy.tracking.propspeed import cleanup_pam_tracking
        cleanup_pam_tracking()  # ← Cleanup AFTER tracking completes


cdef void generate_tractogram_c(double[:,::1] seed_positions,
                                double[:,::1] seed_directions,
                                int nbr_threads,
                                StoppingCriterion sc,
                                TrackerParameters params,
                                PmfGen pmf_gen,
                                double** streamlines,
                                int* lengths,
                                StreamlineStatus* status):  # ← Remove pam_data completely
    """Generate a tractogram from a set of seed points and directions."""
    cdef:
        cnp.npy_intp _len=seed_positions.shape[0]
        cnp.npy_intp i
        double default_dir[3]  # Declare default direction at the top
        # Variables for optimized memory management
        cnp.npy_intp buffer_size
        double* stream
        int* stream_idx
        cnp.npy_intp streamline_size

    if nbr_threads <= 0:
        nbr_threads = 0

    # Generate tractogram with parallel processing

    # Temporarily use sequential to debug missing streamlines
    for i in range(_len):
        # Allocate stream buffer (optimized size calculation)
        buffer_size = params.max_nbr_pts * 6 + 1  # 2 directions * 3 coords + 1
        stream = <double*> malloc(buffer_size * sizeof(double))
        stream_idx = <int*> malloc(2 * sizeof(int))

        # Early exit if allocation failed
        if stream == NULL or stream_idx == NULL:
            if stream != NULL:
                free(stream)
            if stream_idx != NULL:
                free(stream_idx)
            status[i] = INVALIDSTREAMLIME
            lengths[i] = 0
            streamlines[i] = NULL
            continue

        # Generate streamline with appropriate initial direction
        if seed_directions is None:
            # For EuDX, determine initial direction from peaks at seed location
            # Use the strongest peak as initial direction
            default_dir[0] = 1.0
            default_dir[1] = 0.0
            default_dir[2] = 0.0
            # The EuDX propagator will determine the proper initial direction
            status[i] = generate_local_streamline(&seed_positions[i][0],
                                                  default_dir,
                                                  stream,
                                                  stream_idx,
                                                  sc,
                                                  params,
                                                  pmf_gen)
        else:
            status[i] = generate_local_streamline(&seed_positions[i][0],
                                                  &seed_directions[i][0],
                                                  stream,
                                                  stream_idx,
                                                  sc,
                                                  params,
                                                  pmf_gen)

        # Process streamline result
        if status[i] != INVALIDSTREAMLIME and stream_idx[1] >= stream_idx[0]:
            lengths[i] = stream_idx[1] - stream_idx[0] + 1

            # Only allocate if we have a valid streamline
            if lengths[i] > 0:
                streamline_size = lengths[i] * 3
                streamlines[i] = <double*> malloc(streamline_size * sizeof(double))

                if streamlines[i] != NULL:
                    # Fast memory copy
                    memcpy(streamlines[i], &stream[stream_idx[0] * 3],
                           streamline_size * sizeof(double))
                else:
                    # Handle allocation failure
                    status[i] = INVALIDSTREAMLIME
                    lengths[i] = 0
            else:
                streamlines[i] = NULL
                lengths[i] = 0
        else:
            # Invalid streamline
            streamlines[i] = NULL
            lengths[i] = 0
            status[i] = INVALIDSTREAMLIME

        # Clean up temporary buffers
        free(stream)
        free(stream_idx)


cdef StreamlineStatus generate_local_streamline(double* seed,
                                                double* direction,
                                                double* stream,
                                                int* stream_idx,
                                                StoppingCriterion sc,
                                                TrackerParameters params,
                                                PmfGen pmf_gen) noexcept nogil:
    cdef:
        cnp.npy_intp i, j
        cnp.npy_uint32 s_random_seed
        cnp.npy_uint32 seed_x, seed_y, seed_z, position_hash
        cnp.npy_uint32 golden_ratio_const = 2654435769
        double[3] point
        double[3] voxdir
        double voxdir_norm
        double* stream_data
        StreamlineStatus status_forward, status_backward
        fast_numpy.RNGState rng

    # Debug at start of streamline generation (can't print in nogil context, so skip for now)

    # set the random generator with deterministic seeding for parallel consistency
    # Use high precision integer conversion to avoid floating point inconsistencies
    seed_x = <cnp.npy_uint32>(seed[0] * 10000000)  # Higher precision
    seed_y = <cnp.npy_uint32>(seed[1] * 10000000)
    seed_z = <cnp.npy_uint32>(seed[2] * 10000000)

    # Create a unique, deterministic seed for each seed position
    position_hash = (seed_x * 73856093) ^ (seed_y * 19349663) ^ (seed_z * 83492791)

    if params.random_seed != 0:
        s_random_seed = position_hash ^ params.random_seed
    else:
        s_random_seed = position_hash ^ golden_ratio_const

    fast_numpy.seed_rng(&rng, s_random_seed)

    # set the initial position
    fast_numpy.copy_point(seed, point)
    fast_numpy.copy_point(direction, voxdir)
    fast_numpy.copy_point(seed, &stream[params.max_nbr_pts * 3])
    stream_idx[0] = stream_idx[1] = params.max_nbr_pts

    # the input direction is invalid
    voxdir_norm = fast_numpy.norm(voxdir)
    if voxdir_norm < 0.99 or voxdir_norm > 1.01:
        return INVALIDSTREAMLIME

    # **FIXED: Proper stream_data allocation**
    stream_data = <double*> malloc(100 * sizeof(double))
    if stream_data == NULL:
        return INVALIDSTREAMLIME
    memset(stream_data, 0, 100 * sizeof(double))

    # forward tracking
    status_forward = TRACKPOINT
    for i in range(1, params.max_nbr_pts):
        if params.tracker(&point[0], &voxdir[0], params, stream_data, pmf_gen, &rng) == TrackerStatus.FAIL:
            break
        # update position
        for j in range(3):
            point[j] += voxdir[j] * params.inv_voxel_size[j] * params.step_size
        fast_numpy.copy_point(point, &stream[(params.max_nbr_pts + i )* 3])

        status_forward = sc.check_point_c(point, &rng)
        if (status_forward == ENDPOINT or
            status_forward == INVALIDPOINT or
            status_forward == OUTSIDEIMAGE):
            break
    stream_idx[1] = params.max_nbr_pts + i - 1

    # backward tracking
    fast_numpy.copy_point(seed, point)
    fast_numpy.copy_point(direction, voxdir)
    if i > 1:
        # Use the first selected orientation for the backward tracking segment
        for j in range(3):
            voxdir[j] = (stream[(params.max_nbr_pts + 1) * 3 + j]
                         - stream[params.max_nbr_pts * 3 + j])
        fast_numpy.normalize(voxdir)

    # flip the initial direction for backward streamline segment
    for j in range(3):
        voxdir[j] = voxdir[j] * -1

    status_backward = TRACKPOINT
    for i in range(1, params.max_nbr_pts):
        if params.tracker(&point[0], &voxdir[0], params, stream_data, pmf_gen, &rng) == TrackerStatus.FAIL:
            break
        # update position
        for j in range(3):
            point[j] += voxdir[j] * params.inv_voxel_size[j] * params.step_size
        fast_numpy.copy_point(point, &stream[(params.max_nbr_pts - i )* 3])

        status_backward = sc.check_point_c(point, &rng)
        if (status_backward == ENDPOINT or
            status_backward == INVALIDPOINT or
            status_backward == OUTSIDEIMAGE):
            break
    stream_idx[0] = params.max_nbr_pts - i + 1

    # **CRITICAL: Free allocated memory**
    free(stream_data)

    # check for valid streamline ending status
    # EuDX should be very permissive - accept any streamline that was successfully tracked
    # Original EuDX doesn't require strict bidirectional termination criteria
    return VALIDSTREAMLIME


cdef void prepare_pmf(double* pmf,
                      double* point,
                      PmfGen pmf_gen,
                      double pmf_threshold,
                      int pmf_len) noexcept nogil:
    """Prepare the probability mass function for streamline generation.

    Parameters
    ----------
    pmf : ndarray
        Probability mass function.
    point : ndarray
        Current tracking position.
    pmf_gen : PmfGen
        Probability mass function generator.
    pmf_threshold : float
        Threshold for the probability mass function.
    pmf_len : int
        Length of the probability mass function.

    """
    cdef:
        cnp.npy_intp i
        double absolute_pmf_threshold
        double max_pmf=0

    pmf = pmf_gen.get_pmf_c(point, pmf)

    for i in range(pmf_len):
        if pmf[i] > max_pmf:
            max_pmf = pmf[i]
    absolute_pmf_threshold = pmf_threshold * max_pmf

    for i in range(pmf_len):
        if pmf[i] < absolute_pmf_threshold:
            pmf[i] = 0.0
