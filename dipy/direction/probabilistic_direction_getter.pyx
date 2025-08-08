# cython: boundscheck=False
# cython: initializedcheck=False
# cython: wraparound=False
# cython: nonecheck=False

"""
Implementation of a probabilistic direction getter based on sampling from
discrete distribution (pmf) at each step of the tracking.
"""
from random import random

import numpy as np
cimport numpy as cnp

from dipy.direction.closest_peak_direction_getter cimport PmfGenDirectionGetter
from dipy.utils.fast_numpy cimport (copy_point, cumsum, norm, normalize,
                                     where_to_insert)


cdef class ProbabilisticDirectionGetter(PmfGenDirectionGetter):
    """Randomly samples direction of a sphere based on probability mass
    function (pmf).

    The main constructors for this class are current from_pmf and from_shcoeff.
    The pmf gives the probability that each direction on the sphere should be
    chosen as the next direction. To get the true pmf from the "raw pmf"
    directions more than ``max_angle`` degrees from the incoming direction are
    set to 0 and the result is normalized.
    """
    def __init__(self, pmf_gen, max_angle, sphere, pmf_threshold=.1, **kwargs):
        """Direction getter from a pmf generator.

        Parameters
        ----------
        pmf_gen : PmfGen
            Used to get probability mass function for selecting tracking
            directions.
        max_angle : float, [0, 90]
            The maximum allowed angle between incoming direction and new
            direction.
        sphere : Sphere
            The set of directions to be used for tracking.
        pmf_threshold : float [0., 1.]
            Used to remove direction from the probability mass function for
            selecting the tracking direction.
        relative_peak_threshold : float in [0., 1.]
            Used for extracting initial tracking directions. Passed to
            peak_directions.
        min_separation_angle : float in [0, 90]
            Used for extracting initial tracking directions. Passed to
            peak_directions.

        See Also
        --------
        dipy.direction.peaks.peak_directions

        """
        PmfGenDirectionGetter.__init__(self, pmf_gen, max_angle, sphere,
                                       pmf_threshold=pmf_threshold, **kwargs)
        # The vertices need to be in a contiguous array
        self.vertices = self.sphere.vertices.copy()


    cdef int get_direction_c(self, double[::1] point, double[::1] direction):
        """Samples a pmf to updates ``direction`` array with a new direction.

        Parameters
        ----------
        point : memory-view (or ndarray), shape (3,)
            The point in an image at which to lookup tracking directions.
        direction : memory-view (or ndarray), shape (3,)
            Previous tracking direction.

        Returns
        -------
        status : int
            Returns 0 `direction` was updated with a new tracking direction, or
            1 otherwise.

        """
        cdef:
            cnp.npy_intp i, idx, _len
            double[:] newdir
            double* pmf
            double last_cdf, cos_sim

        _len = self.len_pmf
        pmf = self._get_pmf(point)


        if norm(&direction[0]) == 0:
            return 1
        normalize(&direction[0])

        with nogil:
            for i in range(_len):
                cos_sim = self.vertices[i][0] * direction[0] \
                        + self.vertices[i][1] * direction[1] \
                        + self.vertices[i][2] * direction[2]
                if cos_sim < 0:
                    cos_sim = cos_sim * -1
                if cos_sim < self.cos_similarity:
                    pmf[i] = 0

            cumsum(pmf, pmf, _len)
            last_cdf = pmf[_len - 1]
            if last_cdf == 0:
                return 1

        idx = where_to_insert(pmf, random() * last_cdf, _len)

        newdir = self.vertices[idx]
        # Update direction and return 0 for error
        if (direction[0] * newdir[0]
            + direction[1] * newdir[1]
            + direction[2] * newdir[2] > 0):
            copy_point(&newdir[0], &direction[0])
        else:
            newdir[0] = newdir[0] * -1
            newdir[1] = newdir[1] * -1
            newdir[2] = newdir[2] * -1
            copy_point(&newdir[0], &direction[0])
        return 0


cdef class DeterministicMaximumDirectionGetter(ProbabilisticDirectionGetter):
    """Return direction of a sphere with the highest probability mass
    function (pmf).
    """
    def __init__(self, pmf_gen, max_angle, sphere, pmf_threshold=.1, **kwargs):
        ProbabilisticDirectionGetter.__init__(self, pmf_gen, max_angle, sphere,
                                              pmf_threshold=pmf_threshold, **kwargs)

    # cdef int get_direction_c(self, double[::1] point, double[::1] direction):
    #     """Find direction with the highest pmf to updates ``direction`` array
    #     with a new direction.
    #     Parameters
    #     ----------
    #     point : memory-view (or ndarray), shape (3,)
    #         The point in an image at which to lookup tracking directions.
    #     direction : memory-view (or ndarray), shape (3,)
    #         Previous tracking direction.
    #     Returns
    #     -------
    #     status : int
    #         Returns 0 `direction` was updated with a new tracking direction, or
    #         1 otherwise.
    #     """
    #     cdef:
    #         cnp.npy_intp _len, max_idx
    #         double[:] newdir
    #         double* pmf
    #         double max_value, cos_sim

    #     pmf = self._get_pmf(point)
    #     _len = self.len_pmf
    #     max_idx = 0
    #     max_value = 0.0

    #     if norm(&direction[0]) == 0:
    #         return 1
    #     normalize(&direction[0])

    #     with nogil:
    #         for i in range(_len):
    #             cos_sim = self.vertices[i][0] * direction[0] \
    #                     + self.vertices[i][1] * direction[1] \
    #                     + self.vertices[i][2] * direction[2]
    #             if cos_sim < 0:
    #                 cos_sim = cos_sim * -1
    #             if cos_sim > self.cos_similarity and pmf[i] > max_value:
    #                 max_idx = i
    #                 max_value = pmf[i]

    #         if max_value <= 0:
    #             return 1

    #         newdir = self.vertices[max_idx]
    #         # Update direction and return 0 for error
    #         if (direction[0] * newdir[0]
    #             + direction[1] * newdir[1]
    #             + direction[2] * newdir[2] > 0):
    #             copy_point(&newdir[0], &direction[0])
    #         else:
    #             newdir[0] = newdir[0] * -1
    #             newdir[1] = newdir[1] * -1
    #             newdir[2] = newdir[2] * -1
    #             copy_point(&newdir[0], &direction[0])
    #     return 0
    cdef int get_direction_c(self, double[::1] point, double[::1] direction):
        """
        Find direction with the highest pmf using trilinear interpolation of PMFs
        from the 8 neighboring voxels.

        Parameters
        ----------
        point : shape (3,)
            The point in an image at which to lookup tracking directions.
        direction : shape (3,)
            Previous tracking direction.

        Returns
        -------
        status : int
            0 if `direction` was updated, 1 otherwise.
        """
        cdef:
            cnp.npy_intp _len, max_idx
            double[:] newdir
            double* pmf_c
            double max_value, cos_sim
            int x0, y0, z0, x1, y1, z1
            double fx, fy, fz
            double w000, w001, w010, w011, w100, w101, w110, w111
            Py_ssize_t i
            # temporary PMFs for each corner
            double* pmf_corner
            double blended_val

        # Get base voxel indices and fractional offsets
        x0 = <int>point[0]
        y0 = <int>point[1]
        z0 = <int>point[2]

        if x0 < 0 or y0 < 0 or z0 < 0 or \
        x0 >= self.pmf_gen.shape[0] or \
        y0 >= self.pmf_gen.shape[1] or \
        z0 >= self.pmf_gen.shape[2]:
            return 1  # outside image

        x1 = min(x0 + 1, self.pmf_gen.shape[0] - 1)
        y1 = min(y0 + 1, self.pmf_gen.shape[1] - 1)
        z1 = min(z0 + 1, self.pmf_gen.shape[2] - 1)

        fx = point[0] - x0
        fy = point[1] - y0
        fz = point[2] - z0

        # Trilinear weights
        w000 = (1-fx)*(1-fy)*(1-fz)
        w001 = (1-fx)*(1-fy)*fz
        w010 = (1-fx)*fy*(1-fz)
        w011 = (1-fx)*fy*fz
        w100 = fx*(1-fy)*(1-fz)
        w101 = fx*(1-fy)*fz
        w110 = fx*fy*(1-fz)
        w111 = fx*fy*fz

        _len = self.len_pmf
        # Allocate blended PMF array
        cdef np.ndarray[np.float64_t, ndim=1] blended_pmf = np.zeros(_len, dtype=np.float64)

        # Blend PMFs from each corner voxel
        pmf_corner = self._get_pmf((x0, y0, z0))
        for i in range(_len):
            blended_pmf[i] += w000 * pmf_corner[i]
        pmf_corner = self._get_pmf((x0, y0, z1))
        for i in range(_len):
            blended_pmf[i] += w001 * pmf_corner[i]
        pmf_corner = self._get_pmf((x0, y1, z0))
        for i in range(_len):
            blended_pmf[i] += w010 * pmf_corner[i]
        pmf_corner = self._get_pmf((x0, y1, z1))
        for i in range(_len):
            blended_pmf[i] += w011 * pmf_corner[i]
        pmf_corner = self._get_pmf((x1, y0, z0))
        for i in range(_len):
            blended_pmf[i] += w100 * pmf_corner[i]
        pmf_corner = self._get_pmf((x1, y0, z1))
        for i in range(_len):
            blended_pmf[i] += w101 * pmf_corner[i]
        pmf_corner = self._get_pmf((x1, y1, z0))
        for i in range(_len):
            blended_pmf[i] += w110 * pmf_corner[i]
        pmf_corner = self._get_pmf((x1, y1, z1))
        for i in range(_len):
            blended_pmf[i] += w111 * pmf_corner[i]

        # Now proceed as original: find the best direction index
        if norm(&direction[0]) == 0:
            return 1
        normalize(&direction[0])

        max_idx = 0
        max_value = 0.0

        with nogil:
            for i in range(_len):
                cos_sim = self.vertices[i][0] * direction[0] \
                        + self.vertices[i][1] * direction[1] \
                        + self.vertices[i][2] * direction[2]
                if cos_sim < 0:
                    cos_sim = -cos_sim
                if cos_sim > self.cos_similarity and blended_pmf[i] > max_value:
                    max_idx = i
                    max_value = blended_pmf[i]

            if max_value <= 0:
                return 1

            newdir = self.vertices[max_idx]
            # Update direction (flip if necessary)
            if (direction[0] * newdir[0]
                + direction[1] * newdir[1]
                + direction[2] * newdir[2] > 0):
                copy_point(&newdir[0], &direction[0])
            else:
                newdir[0] = -newdir[0]
                newdir[1] = -newdir[1]
                newdir[2] = -newdir[2]
                copy_point(&newdir[0], &direction[0])

        return 0
