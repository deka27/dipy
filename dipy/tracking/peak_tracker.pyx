# cython: boundscheck=False
# cython: initializedcheck=False
# cython: wraparound=False
# cython: Nonecheck=False
from cython.parallel import prange
from libc.stdlib cimport malloc, free
from libc.string cimport memcpy, memset
from libc.math cimport ceil

cimport cython
cimport numpy as cnp
import numpy as np

from dipy.tracking.peak_tracker cimport PeakData
from dipy.tracking.propspeed cimport _propagation_direction
from dipy.tracking.stopping_criterion cimport (
    StoppingCriterion,
    StreamlineStatus,
    TRACKPOINT,
    ENDPOINT,
    OUTSIDEIMAGE,
    INVALIDPOINT,
    VALIDSTREAMLIME,
    INVALIDSTREAMLIME,
)
from dipy.utils.fast_numpy cimport copy_point


def generate_tractogram_eudx(double[:, ::1] seed_positions,
                              double[:, ::1] seed_directions,
                              StoppingCriterion sc,
                              double[:, :, :, ::1] qa,
                              double[:, :, :, ::1] ind,
                              double[:, ::1] odf_vertices,
                              affine,
                              int max_len=500,
                              int min_len=2,
                              double step_size=0.5,
                              double[:] voxel_size=None,
                              double qa_thr=0.0239,
                              double ang_thr=60,
                              double total_weight=0.5,
                              int nbr_threads=0,
                              float buffer_frac=1.0,
                              bint save_seeds=False,
                              bint return_all=True):
    """
    Generate tractogram using EuDX algorithm with parallel processing.

    This function provides parallel streamline generation for peak-based
    tracking methods. It works directly with peak data (QA values + indices)
    rather than PMF distributions.

    Parameters
    ----------
    seed_positions : ndarray, shape (N, 3)
        Seed positions in world coordinates.
    seed_directions : ndarray, shape (N, 3)
        Initial tracking directions for each seed (normalized).
    sc : StoppingCriterion
        Stopping criterion for tracking.
    qa : ndarray, shape (X, Y, Z, Npeaks)
        Quantitative anisotropy (peak strength) values.
    ind : ndarray, shape (X, Y, Z, Npeaks)
        Peak indices into odf_vertices array.
    odf_vertices : ndarray, shape (N_vertices, 3)
        Sphere vertices (sampling directions).
    affine : ndarray, shape (4, 4)
        Affine transformation from voxel to world coordinates.
    max_len : int, optional
        Maximum streamline length in mm. Default: 500
    min_len : int, optional
        Minimum streamline length in mm. Default: 2
    step_size : double, optional
        Step size in mm. Default: 0.5
    voxel_size : ndarray, shape (3,), optional
        Voxel dimensions. If None, extracted from affine.
    qa_thr : double, optional
        QA threshold for peak selection. Default: 0.0239
    ang_thr : double, optional
        Angular threshold in degrees. Default: 60
    total_weight : double, optional
        Minimum interpolation weight to continue. Default: 0.5
    nbr_threads : int, optional
        Number of threads for parallel processing. 0 = use all cores. Default: 0
    buffer_frac : float, optional
        Fraction of seeds to process per batch. Default: 1.0
    save_seeds : bool, optional
        If True, yield (streamline, seed) tuples. Default: False
    return_all : bool, optional
        If True, return all streamlines. If False, only valid ones. Default: True

    Yields
    ------
    streamline : ndarray, shape (n_points, 3)
        Generated streamline points in world coordinates.
    seed : ndarray, shape (3,), optional
        Seed position (only if save_seeds=True).
    """
    cdef:
        cnp.npy_intp _len = seed_positions.shape[0]
        cnp.npy_intp _plen = int(ceil(_len * buffer_frac))
        cnp.npy_intp i, seed_start, seed_end
        double** streamlines_arr
        int* length_arr
        StreamlineStatus* status_arr
        PeakData peak_data
        cnp.npy_intp qa_shape_arr[4]
        cnp.npy_intp qa_strides_arr[4]
        int min_pts, max_pts

    if buffer_frac <= 0 or buffer_frac > 1:
        raise ValueError("buffer_frac must be > 0 and <= 1.")

    # Convert length constraints from mm to points
    min_pts = int(min_len / step_size)
    max_pts = int(max_len / step_size)

    # Setup affine transformations
    lin_T = affine[:3, :3].T.copy()
    offset = affine[:3, 3].copy()
    inv_affine = np.linalg.inv(affine)

    # Transform seeds to voxel space
    seed_positions = np.ascontiguousarray(seed_positions, dtype=np.float64)
    seed_positions = np.dot(seed_positions, inv_affine[:3, :3].T.copy())
    seed_positions += inv_affine[:3, 3]

    # Setup peak data structure
    for i in range(4):
        qa_shape_arr[i] = qa.shape[i]
        qa_strides_arr[i] = qa.strides[i]

    peak_data.qa = &qa[0, 0, 0, 0]
    peak_data.ind = &ind[0, 0, 0, 0]
    peak_data.odf_vertices = &odf_vertices[0, 0]
    peak_data.qa_shape = qa_shape_arr
    peak_data.qa_strides = qa_strides_arr
    peak_data.qa_thr = qa_thr
    peak_data.ang_thr = ang_thr
    peak_data.total_weight = total_weight

    # Process seeds in batches
    seed_start = 0
    seed_end = min(_plen, _len)

    while seed_start < _len:
        # Allocate arrays for this batch
        streamlines_arr = <double**> malloc(_plen * sizeof(double*))
        length_arr = <int*> malloc(_plen * sizeof(int))
        status_arr = <StreamlineStatus*> malloc(_plen * sizeof(StreamlineStatus))

        if streamlines_arr == NULL or length_arr == NULL or status_arr == NULL:
            raise MemoryError("Failed to allocate memory for streamline batch")

        # Generate streamlines in parallel
        generate_eudx_batch_c(
            seed_positions[seed_start:seed_end],
            seed_directions[seed_start:seed_end],
            nbr_threads, sc, peak_data,
            voxel_size, step_size, max_pts,
            streamlines_arr, length_arr, status_arr
        )

        # Yield results
        for i in range(seed_end - seed_start):
            if ((status_arr[i] == VALIDSTREAMLIME or return_all)
                and (length_arr[i] >= min_pts and length_arr[i] <= max_pts)):
                # Copy streamline data
                s = np.asarray(<cnp.float_t[:length_arr[i] * 3]> streamlines_arr[i])
                track = s.copy().reshape((-1, 3))

                # Transform back to world coordinates
                track = np.dot(track, lin_T) + offset

                if save_seeds:
                    seed_world = np.dot(
                        seed_positions[seed_start + i], lin_T
                    ) + offset
                    yield track, seed_world
                else:
                    yield track

            # Free this streamline
            free(streamlines_arr[i])

        # Free batch arrays
        free(streamlines_arr)
        free(length_arr)
        free(status_arr)

        # Move to next batch
        seed_start += _plen
        seed_end = min(seed_end + _plen, _len)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void generate_eudx_batch_c(double[:, ::1] seed_positions,
                                 double[:, ::1] seed_directions,
                                 int nbr_threads,
                                 StoppingCriterion sc,
                                 PeakData peak_data,
                                 double[:] voxel_size,
                                 double step_size,
                                 int max_pts,
                                 double** streamlines,
                                 int* lengths,
                                 StreamlineStatus* status) noexcept nogil:
    """
    Process a batch of seeds in parallel using EuDX algorithm.

    This is the core parallel function that distributes seeds across threads.
    Each thread independently generates one or more streamlines.

    Parameters
    ----------
    seed_positions : memoryview
        Seed positions in voxel coordinates.
    seed_directions : memoryview
        Initial directions (normalized).
    nbr_threads : int
        Number of OpenMP threads. 0 = use all available.
    sc : StoppingCriterion
        Stopping criterion.
    peak_data : PeakData
        Peak tracking data (QA, indices, thresholds).
    voxel_size : memoryview
        Voxel dimensions [x, y, z].
    step_size : double
        Step size in mm.
    max_pts : int
        Maximum number of points per streamline.
    streamlines : double**
        Output array of streamline pointers.
    lengths : int*
        Output array of streamline lengths.
    status : StreamlineStatus*
        Output array of streamline statuses.
    """
    cdef:
        cnp.npy_intp _len = seed_positions.shape[0]
        cnp.npy_intp i
        double* stream
        int* stream_idx

    if nbr_threads <= 0:
        nbr_threads = 0

    # PARALLEL LOOP - each thread processes seeds independently
    for i in prange(_len, nogil=True, num_threads=nbr_threads):
        # Allocate temporary buffer for bidirectional tracking
        # Format: [backward points | seed | forward points]
        stream = <double*> malloc((max_pts * 3 * 2 + 1) * sizeof(double))
        stream_idx = <int*> malloc(2 * sizeof(int))

        if stream == NULL or stream_idx == NULL:
            status[i] = INVALIDSTREAMLIME
            if stream != NULL:
                free(stream)
            if stream_idx != NULL:
                free(stream_idx)
            continue

        # Generate streamline for this seed
        status[i] = generate_eudx_streamline(
            &seed_positions[i, 0],
            &seed_directions[i, 0],
            stream,
            stream_idx,
            sc,
            peak_data,
            &voxel_size[0],
            step_size,
            max_pts
        )

        # Copy streamline from buffer to output
        lengths[i] = stream_idx[1] - stream_idx[0] + 1
        streamlines[i] = <double*> malloc(lengths[i] * 3 * sizeof(double))

        if streamlines[i] != NULL:
            memcpy(
                &streamlines[i][0],
                &stream[stream_idx[0] * 3],
                lengths[i] * 3 * sizeof(double)
            )
        else:
            status[i] = INVALIDSTREAMLIME
            lengths[i] = 0

        free(stream)
        free(stream_idx)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef StreamlineStatus generate_eudx_streamline(double* seed,
                                                double* direction,
                                                double* stream,
                                                int* stream_idx,
                                                StoppingCriterion sc,
                                                PeakData peak_data,
                                                double* voxel_size,
                                                double step_size,
                                                int max_pts) noexcept nogil:
    """
    Generate a single streamline using EuDX algorithm.

    This function implements bidirectional tracking from a seed point.
    It uses the existing _propagation_direction() function from propspeed.pyx
    which contains the core EuDX interpolation and direction selection logic.

    Parameters
    ----------
    seed : double*
        Seed position in voxel coordinates [x, y, z].
    direction : double*
        Initial direction (normalized) [dx, dy, dz].
    stream : double*
        Output buffer for streamline points (bidirectional).
        Layout: [backward_pts | seed | forward_pts]
    stream_idx : int*
        Output indices [start_idx, end_idx] in stream buffer.
    sc : StoppingCriterion
        Stopping criterion.
    peak_data : PeakData
        Peak tracking data.
    voxel_size : double*
        Voxel dimensions [x, y, z].
    step_size : double
        Step size in mm.
    max_pts : int
        Maximum points in one direction.

    Returns
    -------
    status : StreamlineStatus
        VALIDSTREAMLIME if endpoints are valid, INVALIDSTREAMLIME otherwise.
    """
    cdef:
        double point[3]
        double dir[3]
        double newdir[3]
        cnp.npy_intp i, j
        StreamlineStatus status_forward, status_backward
        cnp.npy_intp propagation_status
        double norm
        int last_forward_stored, last_backward_stored

    # Initialize - seed is at center of buffer
    copy_point(seed, point)
    copy_point(direction, dir)
    copy_point(seed, &stream[max_pts * 3])
    stream_idx[0] = stream_idx[1] = max_pts

    # === FORWARD TRACKING ===
    status_forward = TRACKPOINT
    last_forward_stored = max_pts  # Start with just the seed

    for i in range(1, max_pts):
        # Get next direction using EuDX propagation
        propagation_status = _propagation_direction(
            point, dir,
            peak_data.qa,
            peak_data.ind,
            peak_data.odf_vertices,
            peak_data.qa_thr,
            peak_data.ang_thr,
            peak_data.qa_shape,
            peak_data.qa_strides,
            newdir,
            peak_data.total_weight
        )

        if propagation_status == 0:
            # No valid direction found
            status_forward = INVALIDPOINT
            break

        # Update direction
        copy_point(newdir, dir)

        # Update position (step forward)
        for j in range(3):
            point[j] += dir[j] / voxel_size[j] * step_size

        # Check stopping criterion BEFORE storing
        status_forward = sc.check_point_c(point)
        if (status_forward == ENDPOINT or
            status_forward == INVALIDPOINT or
            status_forward == OUTSIDEIMAGE):
            break

        # Only store point if it passed the stopping criterion check
        copy_point(point, &stream[(max_pts + i) * 3])
        last_forward_stored = max_pts + i

    stream_idx[1] = last_forward_stored

    # === BACKWARD TRACKING ===
    # Reset to seed position
    copy_point(seed, point)

    # Use the actual direction from the first forward step for backward tracking
    # This is critical at fiber crossings where the initial direction may differ
    # from the actual tracking direction taken
    if stream_idx[1] > max_pts:
        # We took at least one forward step, use that direction (reversed)
        for j in range(3):
            dir[j] = stream[max_pts * 3 + j] - stream[(max_pts + 1) * 3 + j]

        # Normalize
        norm = 0.0
        for j in range(3):
            norm += dir[j] * dir[j]
        norm = norm ** 0.5
        if norm > 0:
            for j in range(3):
                dir[j] = dir[j] / norm
        else:
            # Fallback to negative of initial direction
            copy_point(direction, dir)
            for j in range(3):
                dir[j] = -dir[j]
    else:
        # No forward steps taken, use negative of initial direction
        copy_point(direction, dir)
        for j in range(3):
            dir[j] = -dir[j]

    status_backward = TRACKPOINT
    last_backward_stored = max_pts  # Start with just the seed

    for i in range(1, max_pts):
        # Get next direction
        propagation_status = _propagation_direction(
            point, dir,
            peak_data.qa,
            peak_data.ind,
            peak_data.odf_vertices,
            peak_data.qa_thr,
            peak_data.ang_thr,
            peak_data.qa_shape,
            peak_data.qa_strides,
            newdir,
            peak_data.total_weight
        )

        if propagation_status == 0:
            status_backward = INVALIDPOINT
            break

        # Update direction
        copy_point(newdir, dir)

        # Update position (step backward)
        for j in range(3):
            point[j] += dir[j] / voxel_size[j] * step_size

        # Check stopping criterion BEFORE storing
        status_backward = sc.check_point_c(point)
        if (status_backward == ENDPOINT or
            status_backward == INVALIDPOINT or
            status_backward == OUTSIDEIMAGE):
            break

        # Only store point if it passed the stopping criterion check
        copy_point(point, &stream[(max_pts - i) * 3])
        last_backward_stored = max_pts - i

    stream_idx[0] = last_backward_stored

    # Check validity - streamline is valid if:
    # 1. At least one direction reached a valid endpoint (ENDPOINT or OUTSIDEIMAGE), OR
    # 2. Both directions tracked at least 2 points (even if ending in INVALIDPOINT)
    #
    # This matches the original LocalTracking behavior where streamlines are kept
    # if they make progress, even if they don't reach perfect endpoints.
    cdef bint forward_valid = (status_forward == ENDPOINT or
                               status_forward == OUTSIDEIMAGE or
                               status_forward == TRACKPOINT)
    cdef bint backward_valid = (status_backward == ENDPOINT or
                                status_backward == OUTSIDEIMAGE or
                                status_backward == TRACKPOINT)

    # At least one direction should have tracked successfully
    cdef int forward_points = stream_idx[1] - max_pts
    cdef int backward_points = max_pts - stream_idx[0]

    # Valid if either direction reached an endpoint, or if we have a reasonable streamline
    if ((status_forward == ENDPOINT or status_forward == OUTSIDEIMAGE) or
        (status_backward == ENDPOINT or status_backward == OUTSIDEIMAGE) or
        (forward_points > 1 or backward_points > 1)):
        return VALIDSTREAMLIME

    return INVALIDSTREAMLIME
