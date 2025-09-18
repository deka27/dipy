# cython: boundscheck=False
# cython: initializedcheck=False
# cython: wraparound=False
# cython: Nonecheck=False

"""Track propagation performance functions."""

# cython: profile=True
# cython: embedsignature=True

cimport cython
from cython.parallel import prange

import numpy as np
cimport numpy as cnp

from dipy.core.interpolation cimport _trilinear_interpolation_iso, offset
from dipy.direction.pmf cimport PmfGen
from dipy.utils.fast_numpy cimport (
    copy_point,
    cross,
    cumsum,
    norm,
    normalize,
    random_float,
    random_perpendicular_vector,
    random_point_within_circle,
    RNGState,
    where_to_insert,
)
from dipy.tracking.tractogen cimport prepare_pmf
from dipy.tracking.tracker_parameters cimport TrackerParameters, TrackerStatus

from libc.stdlib cimport malloc, free
from libc.math cimport M_PI, pow, sin, cos, fabs
from libc.stdio cimport printf

cdef extern from "dpy_math.h" nogil:
    double floor(double x)
    float acos(float x )
    double sqrt(double x)
    double DPY_PI


DEF PEAK_NO=5

# initialize numpy runtime
cnp.import_array()

# Global PAM storage for nogil access
cdef double* global_pam_peak_values = NULL
cdef double* global_pam_peak_dirs = NULL
cdef cnp.npy_intp global_pam_shape[4]
cdef cnp.npy_intp global_pam_strides[3]


def ndarray_offset(cnp.ndarray[cnp.npy_intp, ndim=1] indices,
                   cnp.ndarray[cnp.npy_intp, ndim=1] strides,
                   int lenind,
                   int typesize):
    """ Find offset in an N-dimensional ndarray using strides

    Parameters
    ----------
    indices : array, npy_intp shape (N,)
        Indices of the array which we want to find the offset.
    strides : array, shape (N,)
        Strides of array.
    lenind : int
        len of the `indices` array.
    typesize : int
        Number of bytes for data type e.g. if 8 for double, 4 for int32

    Returns
    -------
    offset : integer
        Index position in flattened array

    Examples
    --------
    >>> import numpy as np
    >>> from dipy.tracking.propspeed import ndarray_offset
    >>> I=np.array([1,1])
    >>> A=np.array([[1,0,0],[0,2,0],[0,0,3]])
    >>> S=np.array(A.strides)
    >>> ndarray_offset(I,S,2,A.dtype.itemsize)
    4
    >>> A.ravel()[4]==A[1,1]
    True
    """
    if not cnp.PyArray_CHKFLAGS(indices, cnp.NPY_ARRAY_C_CONTIGUOUS):
        raise ValueError("indices is not C contiguous")
    if not cnp.PyArray_CHKFLAGS(strides, cnp.NPY_ARRAY_C_CONTIGUOUS):
        raise ValueError("strides is not C contiguous")
    return offset(<cnp.npy_intp*> cnp.PyArray_DATA(indices),
                  <cnp.npy_intp*> cnp.PyArray_DATA(strides),
                  lenind,
                  typesize)


cdef cnp.npy_intp _nearest_direction(double* dx,
                                     double* qa,
                                     double *ind,
                                     cnp.npy_intp peaks,
                                     double *odf_vertices,
                                     double qa_thr, double ang_thr,
                                     double *direction) noexcept nogil:
    """ Give the nearest direction to a point, checking threshold and angle

    Parameters
    ----------
    dx : double array shape (3,)
        Moving direction of the current tracking.
    qa : double array shape (Np,)
        Quantitative anisotropy matrix, where ``Np`` is the number of peaks.
    ind : array, float64 shape(x, y, z, Np)
        Index of the track orientation.
    peaks : npy_intp
    odf_vertices : double array shape (N, 3)
        Sampling directions on the sphere.
    qa_thr : float
        Threshold for QA, we want everything higher than this threshold.
    ang_thr : float
        Angle threshold, we only select fiber orientation within this range.
    direction : double array shape (3,)
        The fiber orientation to be considered in the interpolation.  The array
        gets modified in-place.

    Returns
    -------
    delta : bool
        Delta function: if 1 we give it weighting, if it is 0 we don't give any
        weighting.
    """
    cdef:
        double max_dot = 0
        double angl,curr_dot
        double odfv[3]
        cnp.npy_intp i, j, max_doti = 0

    # calculate the cos with radians
    angl = cos((DPY_PI * ang_thr) / 180.)
    # if the maximum peak is lower than the threshold then there is no point
    # continuing tracking
    if qa[0] <= qa_thr:
        return 0
    # for all peaks find the minimum angle between odf_vertices and dx
    for i from 0 <= i < peaks:
        # if the current peak is smaller than the threshold then jump out
        if qa[i] <= qa_thr:
            break
        # copy odf_vertices
        for j from 0 <= j < 3:
            odfv[j]=odf_vertices[3 * <cnp.npy_intp>ind[i] + j]
        # calculate the absolute dot product between dx and odf_vertices
        curr_dot = dx[0] * odfv[0] + dx[1] * odfv[1] + dx[2] * odfv[2]
        if curr_dot < 0: #abs check
            curr_dot = -curr_dot
        # maximum dot means minimum angle
        # store the maximum dot and the corresponding index from the
        # neighboring voxel in maxdoti
        if curr_dot > max_dot:
            max_dot=curr_dot
            max_doti = i
    # if maxdot smaller than our angular *dot* threshold stop tracking
    if max_dot < angl:
        return 0
    # copy the odf_vertices for the voxel qa indices which have the smaller
    # angle
    for j from 0 <= j < 3:
        odfv[j] = odf_vertices[3 * <cnp.npy_intp>ind[max_doti] + j]
    # if the dot product is negative then return the opposite direction
    # otherwise return the same direction
    if dx[0] * odfv[0] + dx[1] * odfv[1] + dx[2] * odfv[2] < 0:
        for j from 0 <= j < 3:
            direction[j] = -odf_vertices[3 * <cnp.npy_intp>ind[max_doti] + j]
        return 1
    for j from 0 <= j < 3:
        direction[j]= odf_vertices[3 * <cnp.npy_intp>ind[max_doti] + j]
    return 1


@cython.cdivision(True)
cdef cnp.npy_intp _propagation_direction(double *point,
                                         double* dx,
                                         double* qa,
                                         double *ind,
                                         double *odf_vertices,
                                         double qa_thr,
                                         double ang_thr,
                                         cnp.npy_intp *qa_shape,
                                         cnp.npy_intp* strides,
                                         double *direction,
                                         double total_weight) noexcept nogil:
    cdef:
        double total_w = 0 # total weighting useful for interpolation
        double delta = 0 # store delta function (stopping function) result
        double new_direction[3] # new propagation direction
        double w[8]
        double qa_tmp[PEAK_NO]
        double ind_tmp[PEAK_NO]
        cnp.npy_intp index[24]
        cnp.npy_intp xyz[4]
        cnp.npy_intp i, j, m
        double normd
        # number of allowed peaks e.g. for fa is 1 for gqi.qa is 5
        cnp.npy_intp peaks = qa_shape[3]

    # Calculate qa & ind of each of the 8 neighboring voxels.
    # To do that we use trilinear interpolation and return the weights and the
    # indices for the weights i.e. xyz in qa[x,y,z]
    _trilinear_interpolation_iso(point, <double *> w, <cnp.npy_intp *> index)
    # check if you are outside of the volume
    for i from 0 <= i < 3:
        new_direction[i] = 0
        if index[7 * 3 + i] >= qa_shape[i] or index[i] < 0:
            return 0
    # for every weight sum the total weighting
    for m from 0 <= m < 8:
        for i from 0 <= i < 3:
            xyz[i]=index[m * 3 + i]
        # fill qa_tmp and ind_tmp
        for j from 0 <= j < peaks:
            xyz[3] = j
            off = offset(<cnp.npy_intp*> xyz, strides, 4, 8)
            qa_tmp[j] = qa[off]
            ind_tmp[j] = ind[off]
        # return the nearest direction by searching in all peaks
        delta=_nearest_direction(dx,
                                 qa_tmp,
                                 ind_tmp,
                                 peaks,
                                 odf_vertices,
                                 qa_thr,
                                 ang_thr,
                                 direction)
        # if delta is 0 then that means that there was no good direction
        # (obeying the thresholds) from that neighboring voxel, so this voxel
        # is not adding to the total weight
        if delta == 0:
            continue
        # add in total
        total_w += w[m]
        for i from 0 <= i < 3:
            new_direction[i] += w[m] * direction[i]
    # if less than half the volume is time to stop propagating
    if total_w < total_weight: # termination
        return 0
    # all good return normalized weighted next direction
    normd = new_direction[0]**2 + new_direction[1]**2 + new_direction[2]**2
    normd = 1 / sqrt(normd)
    for i from 0 <= i < 3:
        direction[i] = new_direction[i] * normd
    return 1


cdef cnp.npy_intp _initial_direction(double* seed,double *qa,
                                     double* ind, double* odf_vertices,
                                     double qa_thr,
                                     cnp.npy_intp* strides,
                                     cnp.npy_intp ref,
                                     double* direction) noexcept nogil:
    """ First direction that we get from a seeding point
    """
    cdef:
        cnp.npy_intp point[4]
        cnp.npy_intp off
        cnp.npy_intp i
        double qa_tmp,ind_tmp
    # Very tricky/cool addition/flooring that helps create a valid neighborhood
    # (grid) for the trilinear interpolation to run smoothly.
    # Find the index for qa
    for i from 0 <= i < 3:
        point[i] = <cnp.npy_intp>floor(seed[i] + .5)
    point[3] = ref
    # Find the offset in memory to access the qa value
    off = offset(<cnp.npy_intp*>point,strides, 4, 8)
    qa_tmp = qa[off]
    # Check for scalar threshold
    if qa_tmp < qa_thr:
        return 0
    # Find the correct direction from the indices
    ind_tmp = ind[off] # similar to ind[point] in numpy syntax
    # Return initial direction through odf_vertices by ind
    for i from 0 <= i < 3:
        direction[i] = odf_vertices[3 * <cnp.npy_intp>ind_tmp + i]
    return 1


def eudx_both_directions(cnp.ndarray[double, ndim=1] seed,
                         cnp.npy_intp ref,
                         cnp.ndarray[double, ndim=4] qa,
                         cnp.ndarray[double, ndim=4] ind,
                         cnp.ndarray[double, ndim=2] odf_vertices,
                         double qa_thr,
                         double ang_thr,
                         double step_sz,
                         double total_weight,
                         cnp.npy_intp max_points):
    """
    Parameters
    ----------
    seed : array, float64 shape (3,)
        Point where the tracking starts.
    ref : cnp.npy_intp int
        Index of peak to follow first.
    qa : array, float64 shape (X, Y, Z, Np)
        Anisotropy matrix, where ``Np`` is the number of maximum allowed peaks.
    ind : array, float64 shape(x, y, z, Np)
        Index of the track orientation.
    odf_vertices : double array shape (N, 3)
        Sampling directions on the sphere.
    qa_thr : float
        Threshold for QA, we want everything higher than this threshold.
    ang_thr : float
        Angle threshold, we only select fiber orientation within this range.
    step_sz : double
    total_weight : double
    max_points : cnp.npy_intp

    Returns
    -------
    track : array, shape (N,3)
    """
    cdef:
        double *ps = <double *> cnp.PyArray_DATA(seed)
        double *pqa = <double*> cnp.PyArray_DATA(qa)
        double *pin = <double*> cnp.PyArray_DATA(ind)
        double *pverts = <double*> cnp.PyArray_DATA(odf_vertices)
        cnp.npy_intp *pstr = <cnp.npy_intp *> cnp.PyArray_STRIDES(qa)
        cnp.npy_intp *qa_shape = <cnp.npy_intp *> cnp.PyArray_DIMS(qa)
        cnp.npy_intp *pvstr = <cnp.npy_intp *> cnp.PyArray_STRIDES(odf_vertices)
        cnp.npy_intp d, i, j, cnt
        double direction[3]
        double dx[3]
        double idirection[3]
        double ps2[3]
        double tmp, ftmp
    if not cnp.PyArray_CHKFLAGS(seed, cnp.NPY_ARRAY_C_CONTIGUOUS):
        raise ValueError("seed is not C contiguous")
    if not cnp.PyArray_CHKFLAGS(qa, cnp.NPY_ARRAY_C_CONTIGUOUS):
        raise ValueError("qa is not C contiguous")
    if not cnp.PyArray_CHKFLAGS(ind, cnp.NPY_ARRAY_C_CONTIGUOUS):
        raise ValueError("ind is not C contiguous")
    if not cnp.PyArray_CHKFLAGS(odf_vertices, cnp.NPY_ARRAY_C_CONTIGUOUS):
        raise ValueError("odf_vertices is not C contiguous")

    cnt = 0
    d = _initial_direction(ps, pqa, pin, pverts, qa_thr, pstr, ref, idirection)
    if d == 0:
        return None
    for i from 0 <= i < 3:
        # store the initial direction
        dx[i] = idirection[i]
        # ps2 is for downwards and ps for upwards propagation
        ps2[i] = ps[i]
    point = seed.copy()
    track = [point.copy()]
    # track towards one direction
    while d:
        d = _propagation_direction(ps, dx, pqa, pin, pverts, qa_thr, ang_thr,
                                   qa_shape, pstr, direction, total_weight)
        if d == 0:
            break
        if cnt > max_points:
            break
        # update the track
        for i from 0 <= i < 3:
            dx[i] = direction[i]
            # check for boundaries
            tmp = ps[i] + step_sz * dx[i]
            if tmp > qa_shape[i] - 1 or tmp < 0.:
                 d = 0
                 break
            # propagate
            ps[i] = tmp
            point[i] = ps[i]

        if d == 1:
            track.append(point.copy())
            cnt += 1
    d = 1
    for i from 0 <= i < 3:
        dx[i] = -idirection[i]

    cnt = 0
    # track towards the opposite direction
    while d:
        d = _propagation_direction(ps2, dx, pqa, pin, pverts, qa_thr, ang_thr,
                                   qa_shape, pstr, direction, total_weight)
        if d == 0:
            break
        if cnt > max_points:
            break
        # update the track
        for i from 0 <= i < 3:
            dx[i] = direction[i]
            # check for boundaries
            tmp=ps2[i] + step_sz*dx[i]
            if tmp > qa_shape[i] - 1 or tmp < 0.:
                 d = 0
                 break
            # propagate
            ps2[i] = tmp
            point[i] = ps2[i] # to be changed
        # add track point
        if d == 1:
            track.insert(0, point.copy())
            cnt += 1
    # prepare to return final track for the current seed
    tmp_track = np.array(track, dtype=np.float32)

    # Sometimes one of the ends takes small negative values; needs to be
    # investigated further

    # Return track for the current seed point and ref
    return tmp_track


cdef int initialize_ptt(TrackerParameters params,
                        double* stream_data,
                        PmfGen pmf_gen,
                        double* seed_point,
                        double* seed_direction,
                        RNGState* rng) noexcept nogil:
        """Sample an initial curve by rejection sampling.

        Parameters
        ----------
        params : TrackerParameters
            PTT tracking parameters.
        stream_data : double*
            Streamline data persitant across tracking steps.
        pmf_gen : PmfGen
            Orientation data.
        seed_point : double[3]
            Initial point
        seed_direction : double[3]
            Initial direction
        rng : RNGState*
            Random number generator state. (Threadsafe)

        Returns
        -------
        status : int
            Returns 0 if the initialization was successful, or
            1 otherwise.
        """
        cdef double data_support = 0
        cdef double max_posterior = 0
        cdef int tries

        # position
        stream_data[19] = seed_point[0]
        stream_data[20] = seed_point[1]
        stream_data[21] = seed_point[2]

        for tries in range(params.ptt.rejection_sampling_nbr_sample):
            initialize_ptt_candidate(params, stream_data, pmf_gen, seed_direction, rng)
            data_support = calculate_ptt_data_support(params, stream_data, pmf_gen)
            if data_support > max_posterior:
                max_posterior = data_support

        # Compensation for underestimation of max posterior estimate
        max_posterior = pow(2.0 * max_posterior, params.ptt.data_support_exponent)

        # Initialization is successful if a suitable candidate can be sampled
        # within the trial limit
        for tries in range(params.ptt.rejection_sampling_max_try):
            initialize_ptt_candidate(params, stream_data, pmf_gen, seed_direction, rng)
            if (random_float(rng) * max_posterior <= calculate_ptt_data_support(params, stream_data, pmf_gen)):
                stream_data[22] = stream_data[23] # last_val = last_val_cand
                return 0
        return 1


cdef void initialize_ptt_candidate(TrackerParameters params,
                                   double* stream_data,
                                   PmfGen pmf_gen,
                                   double* init_dir,
                                   RNGState* rng) noexcept nogil:
    """
    Initialize the parallel transport frame.

    After initial position is set, a parallel transport frame is set using
    the initial direction (a walking frame, i.e., 3 orthonormal vectors,
    plus 2 scalars, i.e. k1 and k2).

    A point and parallel transport frame parametrizes a curve that is named
    the "probe". Using probe parameters (probe_length, probe_radius,
    probe_quality, probe_count), a short fiber bundle segment is modelled.

    Parameters
    ----------
    params : TrackerParameters
        PTT tracking parameters.
    stream_data : double*
        Streamline data persitant across tracking steps.
    pmf_gen : PmfGen
        Orientation data.
    init_dir : double[3]
        Initial tracking direction (tangent)
    rng : RNGState*
        Random number generator state. (Threadsafe)

    Returns
    -------

    """
    cdef double[3] position
    cdef int count
    cdef int i
    cdef double* pmf

    # Initialize Frame
    stream_data[1] = init_dir[0]
    stream_data[2] = init_dir[1]
    stream_data[3] = init_dir[2]
    random_perpendicular_vector(&stream_data[7],
                                &stream_data[1],
                                rng)  # frame2, frame0
    cross(&stream_data[4],
          &stream_data[7],
          &stream_data[1])  # frame1, frame2, frame0
    stream_data[24], stream_data[25] = \
        random_point_within_circle(params.max_curvature, rng)

    stream_data[22] = 0  # last_val

    if params.ptt.probe_count == 1:
        stream_data[22] = pmf_gen.get_pmf_value_c(&stream_data[19],
                                                  &stream_data[1])  # position, frame[0]
    else:
        for count in range(params.ptt.probe_count):
            for i in range(3):
                position[i] = (stream_data[19 + i]
                              + stream_data[4 + i]
                              * params.ptt.probe_radius
                              * cos(count * params.ptt.angular_separation)
                              * params.inv_voxel_size[i]
                              + stream_data[7 + i]
                              * params.ptt.probe_radius
                              * sin(count * params.ptt.angular_separation)
                              * params.inv_voxel_size[i])

            stream_data[22] += pmf_gen.get_pmf_value_c(&stream_data[19],
                                                       &stream_data[1])


cdef void prepare_ptt_propagator(TrackerParameters params,
                                 double* stream_data,
                                 double arclength) noexcept nogil:
    """Prepare the propagator.

    The propagator used for transporting the moving frame forward.

    Parameters
    ----------
    params : TrackerParameters
        PTT tracking parameters.
    stream_data : double*
        Streamline data persitant across tracking steps.
    arclength : double
        Arclenth, which is equivalent to step size along the arc

    """
    cdef double tmp_arclength
    stream_data[10] = arclength  # propagator[0]

    if (fabs(stream_data[24]) < params.ptt.k_small
        and fabs(stream_data[25]) < params.ptt.k_small):
        stream_data[11] = 0
        stream_data[12] = 0
        stream_data[13] = 1
        stream_data[14] = 0
        stream_data[15] = 0
        stream_data[16] = 0
        stream_data[17] = 0
        stream_data[18] = 1
    else:
        if fabs(stream_data[24]) < params.ptt.k_small:  # k1
            stream_data[24] = params.ptt.k_small
        if fabs(stream_data[25]) < params.ptt.k_small:  # k2
            stream_data[25] = params.ptt.k_small

        tmp_arclength  = arclength * arclength / 2.0

        # stream_data[10:18] -> propagator
        stream_data[11] = stream_data[24] * tmp_arclength
        stream_data[12] = stream_data[25] * tmp_arclength
        stream_data[13] = (1 - stream_data[25]
                          * stream_data[25] * tmp_arclength
                          - stream_data[24] * stream_data[24]
                          * tmp_arclength)
        stream_data[14] = stream_data[24] * arclength
        stream_data[15] = stream_data[25] * arclength
        stream_data[16] = -stream_data[25] * arclength
        stream_data[17] = (-stream_data[24] * stream_data[25]
                          * tmp_arclength)
        stream_data[18] = (1 - stream_data[25] * stream_data[25]
                          * tmp_arclength)


cdef double calculate_ptt_data_support(TrackerParameters params,
                                       double* stream_data,
                                       PmfGen pmf_gen) noexcept nogil:
    """Calculates data support for the candidate probe.

    Parameters
    ----------
    params : TrackerParameters
        PTT tracking parameters.
    stream_data : double*
        Streamline data persitant across tracking steps.
    pmf_gen : PmfGen
        Orientation data.
    """

    cdef double fod_amp
    cdef double[3] position
    cdef double[3][3] frame
    cdef double[3] tangent
    cdef double[3] normal
    cdef double[3] binormal
    cdef double[3] new_position
    cdef double likelihood
    cdef int c, i, j, q
    cdef double* pmf

    prepare_ptt_propagator(params, stream_data, params.ptt.probe_step_size)

    for i in range(3):
        position[i] = stream_data[19+i]
        for j in range(3):
            frame[i][j] = stream_data[1 + i * 3 + j]

    likelihood = stream_data[22]
    for q in range(1, params.ptt.probe_quality):
        for i in range(3):
            # stream_data[10:18] : propagator
            position[i] = \
                (stream_data[10] * frame[0][i] * params.voxel_size[i]
                + stream_data[11] * frame[1][i] * params.voxel_size[i]
                + stream_data[12] * frame[2][i] * params.voxel_size[i]
                + position[i])
            tangent[i] = (stream_data[13] * frame[0][i]
                         + stream_data[14] * frame[1][i]
                         + stream_data[15] * frame[2][i])
        normalize(&tangent[0])

        if q < (params.ptt.probe_quality - 1):
            for i in range(3):
                binormal[i] = (stream_data[16] * frame[0][i]
                              + stream_data[17] * frame[1][i]
                              + stream_data[18] * frame[2][i])
            cross(&normal[0], &binormal[0], &tangent[0])

            copy_point(&tangent[0], &frame[0][0])
            copy_point(&normal[0], &frame[1][0])
            copy_point(&binormal[0], &frame[2][0])
        if params.ptt.probe_count == 1:
            fod_amp = pmf_gen.get_pmf_value_c(position, tangent)
            fod_amp = fod_amp if fod_amp > params.sh.pmf_threshold else 0
            stream_data[23] = fod_amp  # last_val_cand
            likelihood += stream_data[23]  # last_val_cand
        else:
            stream_data[23] = 0  # last_val_cand
            if q == params.ptt.probe_quality - 1:
                for i in range(3):
                    binormal[i] = (stream_data[16] * frame[0][i]
                                  + stream_data[17] * frame[1][i]
                                  + stream_data[18] * frame[2][i])
                cross(&normal[0], &binormal[0], &tangent[0])

            for c in range(params.ptt.probe_count):
                for i in range(3):
                    new_position[i] = (position[i]
                                      + normal[i] * params.ptt.probe_radius
                                      * cos(c * params.ptt.angular_separation)
                                      * params.inv_voxel_size[i]
                                      + binormal[i] * params.ptt.probe_radius
                                      * sin(c * params.ptt.angular_separation)
                                      * params.inv_voxel_size[i])
                fod_amp = pmf_gen.get_pmf_value_c(new_position, tangent)
                fod_amp = fod_amp if fod_amp > params.sh.pmf_threshold else 0
                stream_data[23] += fod_amp  # last_val_cand

            likelihood += stream_data[23]  # last_val_cand

    likelihood *= params.ptt.probe_normalizer
    if params.ptt.data_support_exponent != 1:
        likelihood = pow(likelihood, params.ptt.data_support_exponent)

    return likelihood

cdef cnp.npy_intp _eudx_nearest_direction(double* dx,
                                          double* peak_values,
                                          double* peak_dirs,
                                          cnp.npy_intp num_peaks,
                                          double qa_thr,
                                          double ang_thr,
                                          double* direction) noexcept nogil:
    """EuDX version of _nearest_direction using PAM peaks"""
    cdef:
        double max_dot = 0
        double angl, curr_dot
        double peak_dir[3]
        cnp.npy_intp i, j, max_idx = 0
        cnp.npy_intp valid_peak_found = 0

    # Calculate angular threshold in radians
    angl = cos((DPY_PI * ang_thr) / 180.)

    # Check if we have any peaks
    if num_peaks == 0:
        global global_eudx_direction_fails
        global_eudx_direction_fails += 1
        return 0

    # Find the peak that best matches current direction
    for i in range(num_peaks):
        # Skip very low peaks (PAM values are normalized [0,1])
        if peak_values[i] <= 0.01:  # Very permissive threshold
            continue

        # Get peak direction
        for j in range(3):
            peak_dir[j] = peak_dirs[i * 3 + j]

        # Calculate dot product
        curr_dot = dx[0] * peak_dir[0] + dx[1] * peak_dir[1] + dx[2] * peak_dir[2]
        if curr_dot < 0:
            curr_dot = -curr_dot

        # Find best match
        if curr_dot > max_dot:
            max_dot = curr_dot
            max_idx = i
            valid_peak_found = 1

    # Check if we found any valid peak
    if not valid_peak_found:
        global global_eudx_direction_fails
        global_eudx_direction_fails += 1
        return 0

    # Check angular constraint
    if max_dot < angl:
        return 0

    # Set output direction with correct sign
    for j in range(3):
        peak_dir[j] = peak_dirs[max_idx * 3 + j]

    if dx[0] * peak_dir[0] + dx[1] * peak_dir[1] + dx[2] * peak_dir[2] < 0:
        for j in range(3):
            direction[j] = -peak_dir[j]
    else:
        for j in range(3):
            direction[j] = peak_dir[j]

    return 1


cdef cnp.npy_intp _eudx_nearest_direction_optimized(double* dx,
                                                   double* peak_values,
                                                   double* peak_dirs,
                                                   cnp.npy_intp num_peaks,
                                                   double qa_thr,
                                                   double ang_thr,
                                                   double* direction) noexcept nogil:
    """Optimized EuDX direction selection - no memory copying, direct pointer access"""
    cdef:
        double max_dot = 0
        double angl, curr_dot
        cnp.npy_intp i, max_idx = 0

    # Early exit for invalid regions (minimal check only)
    if num_peaks == 0 or peak_values[0] <= qa_thr * 0.05:
        return 0

    # Calculate angular threshold in radians once
    angl = cos((DPY_PI * ang_thr) / 180.)

    # Find best matching peak using direct pointer access (loop will be optimized by compiler)
    for i in range(num_peaks):
        # Calculate dot product directly from global data
        curr_dot = (dx[0] * peak_dirs[i * 3] +
                   dx[1] * peak_dirs[i * 3 + 1] +
                   dx[2] * peak_dirs[i * 3 + 2])

        # Take absolute value for bidirectional matching
        if curr_dot < 0:
            curr_dot = -curr_dot

        # Track best match
        if curr_dot > max_dot:
            max_dot = curr_dot
            max_idx = i

    # Aggressive fallback for failed angular constraint
    if max_dot < angl and num_peaks > 0 and peak_values[0] > qa_thr * 0.1:
        max_idx = 0
        max_dot = 1.0

    # Set output direction with correct sign (direct calculation)
    curr_dot = (dx[0] * peak_dirs[max_idx * 3] +
               dx[1] * peak_dirs[max_idx * 3 + 1] +
               dx[2] * peak_dirs[max_idx * 3 + 2])

    if curr_dot < 0:
        direction[0] = -peak_dirs[max_idx * 3]
        direction[1] = -peak_dirs[max_idx * 3 + 1]
        direction[2] = -peak_dirs[max_idx * 3 + 2]
    else:
        direction[0] = peak_dirs[max_idx * 3]
        direction[1] = peak_dirs[max_idx * 3 + 1]
        direction[2] = peak_dirs[max_idx * 3 + 2]

    return 1


cdef cnp.npy_intp _eudx_propagation_direction(double* point,
                                              double* dx,
                                              double* pam_peak_values,
                                              double* pam_peak_dirs,
                                              double qa_thr,
                                              double ang_thr,
                                              cnp.npy_intp* pam_shape,
                                              cnp.npy_intp* strides,
                                              double* direction,
                                              double total_weight) noexcept nogil:
    """EuDX propagation using PAM data with trilinear interpolation"""
    cdef:
        double total_w = 0
        double delta = 0
        double new_direction[3]
        double w[8]
        double peak_values_tmp[PEAK_NO]
        double peak_dirs_tmp[PEAK_NO * 3]
        cnp.npy_intp index[24]
        cnp.npy_intp xyz[4]
        cnp.npy_intp i, j, m, k
        cnp.npy_intp off
        double normd
        cnp.npy_intp num_peaks = pam_shape[3]

    # Trilinear interpolation
    _trilinear_interpolation_iso(point, <double*> w, <cnp.npy_intp*> index)
    
    # Check bounds
    global global_eudx_bounds_fails
    for i in range(3):
        new_direction[i] = 0
        if index[7 * 3 + i] >= pam_shape[i] or index[i] < 0:
            global_eudx_bounds_fails += 1
            return 0
    
    # For each neighboring voxel
    for m in range(8):
        for i in range(3):
            xyz[i] = index[m * 3 + i]
        
        # Extract peaks for this voxel
        for j in range(num_peaks):
            # Calculate offset directly using our pre-computed strides
            # strides[0] = y*z*peaks, strides[1] = z*peaks, strides[2] = peaks
            off = xyz[0] * strides[0] + xyz[1] * strides[1] + xyz[2] * strides[2] + j
            peak_values_tmp[j] = pam_peak_values[off]

            # Extract peak directions - directions are stored sequentially
            # Each peak has 3 components (x, y, z)
            for k in range(3):
                peak_dirs_tmp[j * 3 + k] = pam_peak_dirs[off * 3 + k]
        
        # Find nearest direction
        delta = _eudx_nearest_direction(dx, peak_values_tmp, peak_dirs_tmp,
                                        num_peaks, qa_thr, ang_thr, direction)
        
        if delta == 0:
            continue
            
        total_w += w[m]
        for i in range(3):
            new_direction[i] += w[m] * direction[i]
    
    # Check if enough support
    global global_eudx_weight_fails
    if total_w < total_weight:
        global_eudx_weight_fails += 1
        return 0
    
    # Normalize result
    normd = new_direction[0]**2 + new_direction[1]**2 + new_direction[2]**2
    normd = 1 / sqrt(normd)
    for i in range(3):
        direction[i] = new_direction[i] * normd
    
    return 1


cdef int global_eudx_calls = 0
cdef int global_eudx_successes = 0
cdef int global_eudx_bounds_fails = 0
cdef int global_eudx_weight_fails = 0
cdef int global_eudx_direction_fails = 0

cdef TrackerStatus eudx_propagator(double* point,
                                   double* direction,
                                   TrackerParameters params,
                                   double* stream_data,
                                   PmfGen pmf_gen,
                                   RNGState* rng) noexcept nogil:
    """EuDX propagator using trilinear interpolation like original implementation"""
    global global_eudx_calls
    global global_eudx_successes

    cdef:
        cnp.npy_intp result
        # PAM peak values are normalized [0,1], not QA values
        # Use a threshold appropriate for normalized peak strengths
        double qa_thr = 0.01  # Very low threshold for PAM values
        double ang_thr = 80.0   # More permissive angular threshold for EuDX
        double total_weight = 0.3  # Lower weight threshold to allow more streamlines

    global_eudx_calls += 1

    # Check if PAM data is available
    if global_pam_peak_values == NULL or global_pam_peak_dirs == NULL:
        return TrackerStatus.FAIL

    # Check if point is within bounds before calling propagation
    cdef cnp.npy_intp i
    for i in range(3):
        if point[i] < 0 or point[i] >= global_pam_shape[i]:
            # Point is outside volume
            return TrackerStatus.FAIL

    # Use trilinear interpolation like the original EuDX implementation
    result = _eudx_propagation_direction(point, direction,
                                        global_pam_peak_values,
                                        global_pam_peak_dirs,
                                        qa_thr, ang_thr,
                                        global_pam_shape, global_pam_strides,
                                        direction, total_weight)

    if result:
        global_eudx_successes += 1
        return TrackerStatus.SUCCESS
    else:
        return TrackerStatus.FAIL


cdef TrackerStatus deterministic_propagator(double* point,
                                            double* direction,
                                            TrackerParameters params,
                                            double* stream_data,
                                            PmfGen pmf_gen,
                                            RNGState* rng) noexcept nogil:
    """
    Propagate the position by step_size amount.

    The propagation use the direction of a sphere with the highest probability
    mass function (pmf).

    Parameters
    ----------
    point : double[3]
        Current tracking position.
    direction : double[3]
        Previous tracking direction.
    params : TrackerParameters
        Deterministic Tractography parameters.
    stream_data : double*
        Streamline data persitant across tracking steps.
    pmf_gen : PmfGen
        Orientation data.
    rng : RNGState*
        Random number generator state. (thread safe)

    Returns
    -------
    status : TrackerStatus
        Returns SUCCESS if the propagation was successful, or
        FAIL otherwise.
    """
    cdef:
        cnp.npy_intp i, max_idx
        double max_value=0
        double* newdir
        double* pmf
        double cos_sim
        cnp.npy_intp len_pmf=pmf_gen.pmf.shape[0]

    if norm(direction) == 0:
        return TrackerStatus.FAIL
    normalize(direction)

    pmf = <double*> malloc(len_pmf * sizeof(double))
    prepare_pmf(pmf, point, pmf_gen, params.sh.pmf_threshold, len_pmf)

    for i in range(len_pmf):
        cos_sim = pmf_gen.vertices[i][0] * direction[0] \
                + pmf_gen.vertices[i][1] * direction[1] \
                + pmf_gen.vertices[i][2] * direction[2]
        if cos_sim < 0:
            cos_sim = cos_sim * -1
        if cos_sim > params.cos_similarity and pmf[i] > max_value:
            max_idx = i
            max_value = pmf[i]

    if max_value <= 0:
        free(pmf)
        return TrackerStatus.FAIL

    newdir = &pmf_gen.vertices[max_idx][0]
    # Update direction
    if (direction[0] * newdir[0]
        + direction[1] * newdir[1]
        + direction[2] * newdir[2] > 0):
        copy_point(newdir, direction)
    else:
        copy_point(newdir, direction)
        direction[0] = direction[0] * -1
        direction[1] = direction[1] * -1
        direction[2] = direction[2] * -1
    free(pmf)
    return TrackerStatus.SUCCESS


cdef TrackerStatus probabilistic_propagator(double* point,
                                            double* direction,
                                            TrackerParameters params,
                                            double* stream_data,
                                            PmfGen pmf_gen,
                                            RNGState* rng) noexcept nogil:
    """
    Propagates the position by step_size amount. The propagation use randomly samples
    direction of a sphere based on probability mass function (pmf).

    Parameters
    ----------
    point : double[3]
        Current tracking position.
    direction : double[3]
        Previous tracking direction.
    params : TrackerParameters
        Parallel Transport Tractography (PTT) parameters.
    stream_data : double*
        Streamline data persitant across tracking steps.
    pmf_gen : PmfGen
        Orientation data.
    rng : RNGState*
        Random number generator state. (thread safe)

    Returns
    -------
    status : TrackerStatus
        Returns SUCCESS if the propagation was successful, or
        FAIL otherwise.
    """

    cdef:
        cnp.npy_intp i, idx
        double* newdir
        double* pmf
        double last_cdf, cos_sim
        cnp.npy_intp len_pmf=pmf_gen.pmf.shape[0]


    if norm(direction) == 0:
        return TrackerStatus.FAIL
    normalize(direction)

    pmf = <double*> malloc(len_pmf * sizeof(double))
    prepare_pmf(pmf, point, pmf_gen, params.sh.pmf_threshold, len_pmf)

    for i in range(len_pmf):
        cos_sim = pmf_gen.vertices[i][0] * direction[0] \
                + pmf_gen.vertices[i][1] * direction[1] \
                + pmf_gen.vertices[i][2] * direction[2]
        if cos_sim < 0:
            cos_sim = cos_sim * -1
        if cos_sim < params.cos_similarity:
            pmf[i] = 0

    cumsum(pmf, pmf, len_pmf)
    last_cdf = pmf[len_pmf - 1]
    if last_cdf == 0:
        free(pmf)
        return TrackerStatus.FAIL

    idx = where_to_insert(pmf, random_float(rng) * last_cdf, len_pmf)
    newdir = &pmf_gen.vertices[idx][0]
    # Update direction
    if (direction[0] * newdir[0]
        + direction[1] * newdir[1]
        + direction[2] * newdir[2] > 0):
        copy_point(newdir, direction)
    else:
        copy_point(newdir, direction)
        direction[0] = direction[0] * -1
        direction[1] = direction[1] * -1
        direction[2] = direction[2] * -1
    free(pmf)
    return TrackerStatus.SUCCESS


cdef TrackerStatus parallel_transport_propagator(double* point,
                                              double* direction,
                                              TrackerParameters params,
                                              double* stream_data,
                                              PmfGen pmf_gen,
                                              RNGState* rng) noexcept nogil:
    """
    Propagates the position by step_size amount. The propagation is using
    the parameters of the last candidate curve. Then, randomly generate
    curve parametrization from the current position. The walking frame
    is the same, only the k1 and k2 parameters are randomly picked.
    Rejection sampling is used to pick the next curve using the data
    support (likelihood).

    stream_data:
        0    : initialized
        1-10 : frame1,2,3
        10-19: propagator
        19-22: position
        22   : last_val
        23   : last_val_cand
        24   : k1
        25   : k2

    Parameters
    ----------
    point : double[3]
        Current tracking position.
    direction : double[3]
        Previous tracking direction.
    params : TrackerParameters
        Parallel Transport Tractography (PTT) parameters.
    stream_data : double*
        Streamline data persitant across tracking steps.
    pmf_gen : PmfGen
        Orientation data.
    rng : RNGState*
        Random number generator state. (thread safe)

    Returns
    -------
    status : TrackerStatus
        Returns SUCCESS if the propagation was successful, or
        FAIL otherwise.
    """

    cdef double max_posterior = 0
    cdef double data_support = 0
    cdef double[3] tangent
    cdef int tries
    cdef int i

    if stream_data[0] == 0:
        initialize_ptt(params, stream_data, pmf_gen, point, direction, rng)
        stream_data[0] = 1  # initialized

    prepare_ptt_propagator(params, stream_data, params.step_size)

    for i in range(3):
        #  position
        stream_data[19 + i] = (stream_data[10] * stream_data[1 + i]
                               * params.inv_voxel_size[i]
                               + stream_data[11] * stream_data[4 + i]
                               * params.inv_voxel_size[i]
                               + stream_data[12] * stream_data[7 + i]
                               * params.inv_voxel_size[i]
                               + stream_data[19 + i])
        tangent[i] = (stream_data[13] * stream_data[1 + i]
                      + stream_data[14] * stream_data[4 + i]
                      + stream_data[15] * stream_data[7 + i])
        stream_data[7 + i] = \
            (stream_data[16] * stream_data[1 + i]
            + stream_data[17] * stream_data[4 + i]
            + stream_data[18] * stream_data[7 + i])
    normalize(&tangent[0])
    cross(&stream_data[4], &stream_data[7], &tangent[0])  # frame1, frame2
    normalize(&stream_data[4])  # frame1
    cross(&stream_data[7], &tangent[0], &stream_data[4])  # frame2, tangent, frame1
    stream_data[1] = tangent[0]
    stream_data[2] = tangent[1]
    stream_data[3] = tangent[2]

    for tries in range(params.ptt.rejection_sampling_nbr_sample):
        # k1, k2
        stream_data[24], stream_data[25] = \
            random_point_within_circle(params.max_curvature, rng)
        data_support = calculate_ptt_data_support(params, stream_data, pmf_gen)
        if data_support > max_posterior:
            max_posterior = data_support

    # Compensation for underestimation of max posterior estimate
    max_posterior = pow(2.0 * max_posterior, params.ptt.data_support_exponent)


    for tries in range(params.ptt.rejection_sampling_max_try):
        # k1, k2
        stream_data[24], stream_data[25] = \
            random_point_within_circle(params.max_curvature, rng)
        if random_float(rng) * max_posterior < calculate_ptt_data_support(params, stream_data, pmf_gen):
            stream_data[22] = stream_data[23] # last_val = last_val_cand
            # Propagation is successful if a suitable candidate can be sampled
            # within the trial limit
            # update the point and return
            copy_point(&stream_data[19], point)
            return TrackerStatus.SUCCESS

    return TrackerStatus.FAIL


def setup_pam_for_tracking(pam):
    """Optimized setup of global PAM data for nogil tracking access"""
    global global_pam_peak_values, global_pam_peak_dirs, global_pam_shape, global_pam_strides

    cdef:
        cnp.npy_intp total_values_size
        cnp.npy_intp total_dirs_size
        cnp.npy_intp i
        # Pre-compute strides for faster indexing
        cnp.npy_intp stride_1, stride_2, stride_3, stride_4
        cnp.npy_intp src_idx, dst_idx
        cnp.npy_intp x, y, z, p
        double* values_ptr
        double* dirs_ptr

    # Store PAM shape (peak_values is 4D: x, y, z, peaks)
    for i in range(4):
        global_pam_shape[i] = pam.peak_values.shape[i]

    # Precompute strides for fast indexing (eliminates multiplication in tight loops)
    global_pam_strides[0] = global_pam_shape[1] * global_pam_shape[2] * global_pam_shape[3]  # x stride
    global_pam_strides[1] = global_pam_shape[2] * global_pam_shape[3]  # y stride
    global_pam_strides[2] = global_pam_shape[3]  # z stride

    # Calculate total data sizes
    total_values_size = (global_pam_shape[0] * global_pam_shape[1] *
                        global_pam_shape[2] * global_pam_shape[3])
    total_dirs_size = total_values_size * 3

    # Allocate aligned memory for better cache performance
    global_pam_peak_values = <double*> malloc(total_values_size * sizeof(double))
    global_pam_peak_dirs = <double*> malloc(total_dirs_size * sizeof(double))

    if global_pam_peak_values == NULL or global_pam_peak_dirs == NULL:
        raise MemoryError("Failed to allocate PAM data")

    # Fast copy of peak values (contiguous memory copy)
    values_ptr = <double*> cnp.PyArray_DATA(pam.peak_values)
    for i in range(total_values_size):
        global_pam_peak_values[i] = values_ptr[i]

    # Optimized copy of peak directions with pre-computed strides
    dirs_ptr = <double*> cnp.PyArray_DATA(pam.peak_dirs)

    # Pre-compute strides for 5D array access (x, y, z, peaks, 3)
    stride_4 = pam.peak_dirs.shape[4]  # 3
    stride_3 = pam.peak_dirs.shape[3] * stride_4  # peaks * 3
    stride_2 = pam.peak_dirs.shape[2] * stride_3  # z * peaks * 3
    stride_1 = pam.peak_dirs.shape[1] * stride_2  # y * z * peaks * 3

    # Optimized nested loop with stride-based indexing
    dst_idx = 0
    for x in range(global_pam_shape[0]):
        for y in range(global_pam_shape[1]):
            for z in range(global_pam_shape[2]):
                for p in range(global_pam_shape[3]):
                    # Calculate source index once for all 3 direction components
                    src_idx = x * stride_1 + y * stride_2 + z * stride_3 + p * stride_4

                    # Copy all 3 direction components in sequence (cache-friendly)
                    global_pam_peak_dirs[dst_idx] = dirs_ptr[src_idx]
                    global_pam_peak_dirs[dst_idx + 1] = dirs_ptr[src_idx + 1]
                    global_pam_peak_dirs[dst_idx + 2] = dirs_ptr[src_idx + 2]
                    dst_idx += 3

    print(f"PAM SETUP: Optimized copy of {total_values_size} values and {total_dirs_size} direction components")


def cleanup_pam_tracking():
    """Clean up global PAM data"""
    global global_pam_peak_values, global_pam_peak_dirs
    global global_eudx_calls, global_eudx_successes
    global global_eudx_bounds_fails, global_eudx_weight_fails, global_eudx_direction_fails

    print(f"DEBUG: EuDX propagator: {global_eudx_calls} calls, {global_eudx_successes} successes")
    print(f"DEBUG: Failures - bounds: {global_eudx_bounds_fails}, weight: {global_eudx_weight_fails}, direction: {global_eudx_direction_fails}")

    if global_pam_peak_values != NULL:
        free(global_pam_peak_values)
        global_pam_peak_values = NULL

    if global_pam_peak_dirs != NULL:
        free(global_pam_peak_dirs)
        global_pam_peak_dirs = NULL

    print("PAM CLEANUP: Global data freed")

    # Reset counters
    global_eudx_calls = 0
    global_eudx_successes = 0
    global_eudx_bounds_fails = 0
    global_eudx_weight_fails = 0
    global_eudx_direction_fails = 0
