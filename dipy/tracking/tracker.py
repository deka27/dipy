from nibabel.affines import voxel_sizes
import numpy as np

from dipy.data import default_sphere
from dipy.direction import (
    BootDirectionGetter,
    ClosestPeakDirectionGetter,
    PeakDirectionGen,
    ProbabilisticDirectionGetter,
)
from dipy.direction.peaks import peaks_from_positions
from dipy.direction.pmf import SHCoeffPmfGen, SimplePmfGen
from dipy.tracking.local_tracking import LocalTracking, ParticleFilteringTracking
from dipy.tracking.tracker_parameters import generate_tracking_parameters
from dipy.tracking.tractogen import generate_tractogram, generate_tractogram_glide
from dipy.tracking.utils import seeds_directions_pairs


def generic_tracking(
    seed_positions,
    seed_directions,
    sc,
    params,
    *,
    affine=None,
    sh=None,
    peaks=None,
    sf=None,
    sphere=None,
    basis_type=None,
    legacy=True,
    nbr_threads=0,
    seed_buffer_fraction=1.0,
    save_seeds=False,
):
    affine = affine if affine is not None else np.eye(4)

    pmf_type = [
        {"name": "sh", "value": sh, "cls": SHCoeffPmfGen},
        {"name": "peaks", "value": peaks, "cls": SimplePmfGen},
        {"name": "sf", "value": sf, "cls": SimplePmfGen},
    ]

    initialized_pmf = [
        d_selected for d_selected in pmf_type if d_selected["value"] is not None
    ]
    if len(initialized_pmf) > 1:
        selected_pmf = ", ".join([p["name"] for p in initialized_pmf])
        raise ValueError(
            "Only one pmf type should be initialized. "
            f"Variables initialized: {', '.join(selected_pmf)}"
        )
    if len(initialized_pmf) == 0:
        available_pmf = ", ".join([d["name"] for d in pmf_type])
        raise ValueError(
            f"No PMF found. One of this variable ({available_pmf}) should be"
            " initialized."
        )

    selected_pmf = initialized_pmf[0]

    if selected_pmf["name"] == "sf" and sphere is None:
        raise ValueError("A sphere should be defined when using SF (an ODF).")

    if selected_pmf["name"] == "peaks":
        raise NotImplementedError("Peaks are not yet implemented.")

    sphere = sphere or default_sphere

    kwargs = {}
    if selected_pmf["name"] == "sh":
        kwargs = {"basis_type": basis_type, "legacy": legacy}

    pmf_gen = selected_pmf["cls"](
        np.asarray(selected_pmf["value"], dtype=float), sphere, **kwargs
    )

    if seed_directions is not None:
        if not isinstance(seed_directions, (np.ndarray, list)):
            raise ValueError("seed_directions should be a numpy array or a list.")
        elif isinstance(seed_directions, list):
            seed_directions = np.array(seed_directions)

        if not np.array_equal(seed_directions.shape, seed_positions.shape):
            raise ValueError(
                "seed_directions and seed_positions should have the same shape."
            )
    else:
        peaks = peaks_from_positions(
            seed_positions, None, None, npeaks=1, affine=affine, pmf_gen=pmf_gen
        )
        seed_positions, seed_directions = seeds_directions_pairs(
            seed_positions, peaks, max_cross=1
        )

    return generate_tractogram(
        seed_positions,
        seed_directions,
        sc,
        params,
        pmf_gen,
        affine=affine,
        nbr_threads=nbr_threads,
        buffer_frac=seed_buffer_fraction,
        save_seeds=save_seeds,
    )


def probabilistic_tracking(
    seed_positions,
    sc,
    affine,
    *,
    seed_directions=None,
    sh=None,
    peaks=None,
    sf=None,
    min_len=2,
    max_len=500,
    step_size=0.2,
    voxel_size=None,
    max_angle=20,
    pmf_threshold=0.1,
    sphere=None,
    basis_type=None,
    legacy=True,
    nbr_threads=0,
    random_seed=0,
    seed_buffer_fraction=1.0,
    return_all=True,
    save_seeds=False,
):
    """Probabilistic tracking algorithm.

    Parameters
    ----------
    seed_positions : ndarray
        Seed positions in world space.
    sc : StoppingCriterion
        Stopping criterion.
    affine : ndarray
        Affine matrix.
    seed_directions : ndarray, optional
        Seed directions.
    sh : ndarray, optional
       Spherical Harmonics (SH).
    peaks : ndarray, optional
        Peaks array.
    sf : ndarray, optional
        Spherical Function (SF).
    min_len : int, optional
        Minimum length (mm) of the streamlines.
    max_len : int, optional
        Maximum length (mm) of the streamlines.
    step_size : float, optional
        Step size of the tracking.
    voxel_size : ndarray, optional
        Voxel size.
    max_angle : float, optional
        Maximum angle.
    pmf_threshold : float, optional
        PMF threshold.
    sphere : Sphere, optional
        Sphere.
    basis_type : name of basis
        The basis that ``shcoeff`` are associated with.
        ``dipy.reconst.shm.real_sh_descoteaux`` is used by default.
    legacy: bool, optional
        True to use a legacy basis definition for backward compatibility
        with previous ``tournier07`` and ``descoteaux07`` implementations.
    nbr_threads: int, optional
        Number of threads to use for the processing. By default, all available threads
        will be used.
    random_seed: int, optional
        Seed for the random number generator, must be >= 0. A value of greater than 0
        will all produce the same streamline trajectory for a given seed coordinate.
        A value of 0 may produces various streamline tracjectories for a given seed
        coordinate.
    seed_buffer_fraction: float, optional
        Fraction of the seed buffer to use. A value of 1.0 will use the entire seed
        buffer. A value of 0.5 will use half of the seed buffer then the other half.
        a way to reduce memory usage.
    return_all: bool, optional
        True to return all the streamlines, False to return only the streamlines that
        reached the stopping criterion.
    save_seeds: bool, optional
        True to return the seeds with the associated streamline.

    Returns
    -------
    Tractogram

    """
    voxel_size = voxel_size if voxel_size is not None else voxel_sizes(affine)

    params = generate_tracking_parameters(
        "prob",
        min_len=min_len,
        max_len=max_len,
        step_size=step_size,
        voxel_size=voxel_size,
        max_angle=max_angle,
        pmf_threshold=pmf_threshold,
        random_seed=random_seed,
        return_all=return_all,
    )

    return generic_tracking(
        seed_positions,
        seed_directions,
        sc,
        params,
        affine=affine,
        sh=sh,
        peaks=peaks,
        sf=sf,
        sphere=sphere,
        basis_type=basis_type,
        legacy=legacy,
        nbr_threads=nbr_threads,
        seed_buffer_fraction=seed_buffer_fraction,
        save_seeds=save_seeds,
    )


def deterministic_tracking(
    seed_positions,
    sc,
    affine,
    *,
    seed_directions=None,
    sh=None,
    peaks=None,
    sf=None,
    min_len=2,
    max_len=500,
    step_size=0.2,
    voxel_size=None,
    max_angle=20,
    pmf_threshold=0.1,
    sphere=None,
    basis_type=None,
    legacy=True,
    nbr_threads=0,
    random_seed=0,
    seed_buffer_fraction=1.0,
    return_all=True,
    save_seeds=False,
):
    """Deterministic tracking algorithm.

    Parameters
    ----------
    seed_positions : ndarray
        Seed positions in world space.
    sc : StoppingCriterion
        Stopping criterion.
    affine : ndarray
        Affine matrix.
    seed_directions : ndarray, optional
        Seed directions.
    sh : ndarray, optional
        Spherical Harmonics (SH).
    peaks : ndarray, optional
        Peaks array.
    sf : ndarray, optional
        Spherical Function (SF).
    min_len : int, optional
        Minimum length (mm) of the streamlines.
    max_len : int, optional
        Maximum length (mm) of the streamlines.
    step_size : float, optional
        Step size of the tracking.
    voxel_size : ndarray, optional
        Voxel size.
    max_angle : float, optional
        Maximum angle.
    pmf_threshold : float, optional
        PMF threshold.
    sphere : Sphere, optional
        Sphere.
    basis_type : name of basis
        The basis that ``shcoeff`` are associated with.
        ``dipy.reconst.shm.real_sh_descoteaux`` is used by default.
    legacy: bool, optional
        True to use a legacy basis definition for backward compatibility
        with previous ``tournier07`` and ``descoteaux07`` implementations.
    nbr_threads: int, optional
        Number of threads to use for the processing. By default, all available threads
        will be used.
    random_seed: int, optional
        Seed for the random number generator, must be >= 0. A value of greater than 0
        will all produce the same streamline trajectory for a given seed coordinate.
        A value of 0 may produces various streamline tracjectories for a given seed
        coordinate.
    seed_buffer_fraction: float, optional
        Fraction of the seed buffer to use. A value of 1.0 will use the entire seed
        buffer. A value of 0.5 will use half of the seed buffer then the other half.
        a way to reduce memory usage.
    return_all: bool, optional
        True to return all the streamlines, False to return only the streamlines that
        reached the stopping criterion.
    save_seeds: bool, optional
        True to return the seeds with the associated streamline.

    Returns
    -------
    Tractogram

    """
    voxel_size = voxel_size if voxel_size is not None else voxel_sizes(affine)

    params = generate_tracking_parameters(
        "det",
        min_len=min_len,
        max_len=max_len,
        step_size=step_size,
        voxel_size=voxel_size,
        max_angle=max_angle,
        pmf_threshold=pmf_threshold,
        random_seed=random_seed,
        return_all=return_all,
    )
    return generic_tracking(
        seed_positions,
        seed_directions,
        sc,
        params,
        affine=affine,
        sh=sh,
        peaks=peaks,
        sf=sf,
        sphere=sphere,
        basis_type=basis_type,
        legacy=legacy,
        nbr_threads=nbr_threads,
        seed_buffer_fraction=seed_buffer_fraction,
        save_seeds=save_seeds,
    )


def ptt_tracking(
    seed_positions,
    sc,
    affine,
    *,
    seed_directions=None,
    sh=None,
    peaks=None,
    sf=None,
    min_len=2,
    max_len=500,
    step_size=0.5,
    voxel_size=None,
    max_angle=10,
    pmf_threshold=0.1,
    probe_length=1.5,
    probe_radius=0,
    probe_quality=7,
    probe_count=1,
    data_support_exponent=1,
    sphere=None,
    basis_type=None,
    legacy=True,
    nbr_threads=0,
    random_seed=0,
    seed_buffer_fraction=1.0,
    return_all=True,
    save_seeds=False,
):
    """Parallel Transport Tractography (PTT) tracking algorithm.

    Parameters
    ----------
    seed_positions : ndarray
        Seed positions in world space.
    sc : StoppingCriterion
        Stopping criterion.
    affine : ndarray
        Affine matrix.
    seed_directions : ndarray, optional
        Seed directions.
    sh : ndarray, optional
        Spherical Harmonics (SH) data.
    peaks : ndarray, optional
        Peaks array
    sf : ndarray, optional
        Spherical Function (SF).
    min_len : int, optional
        Minimum length (mm) of the streamlines.
    max_len : int, optional
        Maximum length (mm) of the streamlines.
    step_size : float, optional
        Step size of the tracking.
    voxel_size : ndarray, optional
        Voxel size.
    max_angle : float, optional
        Maximum angle.
    pmf_threshold : float, optional
        PMF threshold.
    probe_length : float, optional
        Probe length.
    probe_radius : float, optional
        Probe radius.
    probe_quality : int, optional
        Probe quality.
    probe_count : int, optional
        Probe count.
    data_support_exponent : int, optional
        Data support exponent.
    sphere : Sphere, optional
        Sphere.
    basis_type : name of basis
        The basis that ``shcoeff`` are associated with.
        ``dipy.reconst.shm.real_sh_descoteaux`` is used by default.
    legacy: bool, optional
        True to use a legacy basis definition for backward compatibility
        with previous ``tournier07`` and ``descoteaux07`` implementations.
    nbr_threads: int, optional
        Number of threads to use for the processing. By default, all available threads
        will be used.
    random_seed: int, optional
        Seed for the random number generator, must be >= 0. A value of greater than 0
        will all produce the same streamline trajectory for a given seed coordinate.
        A value of 0 may produces various streamline tracjectories for a given seed
        coordinate.
    seed_buffer_fraction: float, optional
        Fraction of the seed buffer to use. A value of 1.0 will use the entire seed
        buffer. A value of 0.5 will use half of the seed buffer then the other half.
        a way to reduce memory usage.
    return_all: bool, optional
        True to return all the streamlines, False to return only the streamlines that
        reached the stopping criterion.
    save_seeds: bool, optional
        True to return the seeds with the associated streamline.
    Returns
    -------
    Tractogram

    """
    voxel_size = voxel_size if voxel_size is not None else voxel_sizes(affine)

    params = generate_tracking_parameters(
        "ptt",
        min_len=min_len,
        max_len=max_len,
        step_size=step_size,
        voxel_size=voxel_size,
        max_angle=max_angle,
        pmf_threshold=pmf_threshold,
        random_seed=random_seed,
        probe_length=probe_length,
        probe_radius=probe_radius,
        probe_quality=probe_quality,
        probe_count=probe_count,
        data_support_exponent=data_support_exponent,
        return_all=return_all,
    )
    return generic_tracking(
        seed_positions,
        seed_directions,
        sc,
        params,
        affine=affine,
        sh=sh,
        peaks=peaks,
        sf=sf,
        sphere=sphere,
        basis_type=basis_type,
        legacy=legacy,
        nbr_threads=nbr_threads,
        seed_buffer_fraction=seed_buffer_fraction,
        save_seeds=save_seeds,
    )


def closestpeak_tracking(
    seed_positions,
    sc,
    affine,
    *,
    seed_directions=None,
    sh=None,
    peaks=None,
    sf=None,
    min_len=2,
    max_len=500,
    step_size=0.5,
    voxel_size=None,
    max_angle=60,
    pmf_threshold=0.1,
    sphere=None,
    basis_type=None,
    legacy=True,
    nbr_threads=0,
    random_seed=0,
    seed_buffer_fraction=1.0,
    return_all=True,
    save_seeds=False,
):
    """Closest peak tracking algorithm.

    Parameters
    ----------
    seed_positions : ndarray
        Seed positions in world space.
    sc : StoppingCriterion
        Stopping criterion.
    affine : ndarray
        Affine matrix.
    seed_directions : ndarray, optional
        Seed directions.
    sh : ndarray, optional
        Spherical Harmonics (SH).
    peaks : ndarray, optional
        Peaks array.
    sf : ndarray, optional
        Spherical Function (SF).
    min_len : int, optional
        Minimum length (mm) of the streamlines.
    max_len : int, optional
        Maximum length (mm) of the streamlines.
    step_size : float, optional
        Step size of the tracking.
    voxel_size : ndarray, optional
        Voxel size.
    max_angle : float, optional
        Maximum angle.
    pmf_threshold : float, optional
        PMF threshold.
    sphere : Sphere, optional
        Sphere.
    basis_type : name of basis
        The basis that ``shcoeff`` are associated with.
        ``dipy.reconst.shm.real_sh_descoteaux`` is used by default.
    legacy: bool, optional
        True to use a legacy basis definition for backward compatibility
        with previous ``tournier07`` and ``descoteaux07`` implementations.
    nbr_threads: int, optional
        Number of threads to use for the processing. By default, all available threads
        will be used.
    random_seed: int, optional
        Seed for the random number generator, must be >= 0. A value of greater than 0
        will all produce the same streamline trajectory for a given seed coordinate.
        A value of 0 may produces various streamline tracjectories for a given seed
        coordinate.
    seed_buffer_fraction: float, optional
        Fraction of the seed buffer to use. A value of 1.0 will use the entire seed
        buffer. A value of 0.5 will use half of the seed buffer then the other half.
        a way to reduce memory usage.
    return_all: bool, optional
        True to return all the streamlines, False to return only the streamlines that
        reached the stopping criterion.
    save_seeds: bool, optional
        True to return the seeds with the associated streamline.

    Returns
    -------
    Tractogram

    """
    dg = None
    sphere = sphere if sphere is not None else default_sphere
    if sh is not None:
        dg = ClosestPeakDirectionGetter.from_shcoeff(
            sh,
            sphere=sphere,
            max_angle=max_angle,
            pmf_threshold=pmf_threshold,
            basis_type=basis_type,
            legacy=legacy,
        )
    elif sf is not None:
        dg = ClosestPeakDirectionGetter.from_pmf(
            sf, sphere=sphere, max_angle=max_angle, pmf_threshold=pmf_threshold
        )
    else:
        raise ValueError("SH or SF should be defined. Not implemented yet for peaks.")

    # convert length in mm to number of points
    min_len = int(min_len / step_size)
    max_len = int(max_len / step_size)

    return LocalTracking(
        dg,
        sc,
        seed_positions,
        affine,
        step_size=step_size,
        minlen=min_len,
        maxlen=max_len,
        random_seed=random_seed,
        return_all=return_all,
        initial_directions=seed_directions,
        save_seeds=save_seeds,
    )


def bootstrap_tracking(
    seed_positions,
    sc,
    affine,
    *,
    seed_directions=None,
    data=None,
    model=None,
    sh=None,
    peaks=None,
    sf=None,
    min_len=2,
    max_len=500,
    step_size=0.5,
    voxel_size=None,
    max_angle=60,
    pmf_threshold=0.1,
    sphere=None,
    basis_type=None,
    legacy=True,
    nbr_threads=0,
    random_seed=0,
    seed_buffer_fraction=1.0,
    return_all=True,
    save_seeds=False,
):
    """Bootstrap tracking algorithm.

    seed_positions : ndarray
        Seed positions in world space.
    sc : StoppingCriterion
        Stopping criterion.
    affine : ndarray
        Affine matrix.
    seed_directions : ndarray, optional
        Seed directions.
    data : ndarray, optional
        Diffusion data.
    model : Model, optional
        Reconstruction model.
    sh : ndarray, optional
        Spherical Harmonics (SH).
    peaks : ndarray, optional
        Peaks array.
    sf : ndarray, optional
        Spherical Function (SF).
    min_len : int, optional
        Minimum length (mm) of the streamlines.
    max_len : int, optional
        Maximum length (mm) of the streamlines.
    step_size : float, optional
        Step size of the tracking.
    voxel_size : ndarray, optional
        Voxel size.
    max_angle : float, optional
        Maximum angle.
    pmf_threshold : float, optional
        PMF threshold.
    sphere : Sphere, optional
        Sphere.
    basis_type : name of basis
        The basis that ``shcoeff`` are associated with.
        ``dipy.reconst.shm.real_sh_descoteaux`` is used by default.
    legacy: bool, optional
        True to use a legacy basis definition for backward compatibility
        with previous ``tournier07`` and ``descoteaux07`` implementations.
    nbr_threads: int, optional
        Number of threads to use for the processing. By default, all available threads
        will be used.
    random_seed: int, optional
        Seed for the random number generator, must be >= 0. A value of greater than 0
        will all produce the same streamline trajectory for a given seed coordinate.
        A value of 0 may produces various streamline tracjectories for a given seed
        coordinate.
    seed_buffer_fraction: float, optional
        Fraction of the seed buffer to use. A value of 1.0 will use the entire seed
        buffer. A value of 0.5 will use half of the seed buffer then the other half.
        a way to reduce memory usage.
    return_all: bool, optional
        True to return all the streamlines, False to return only the streamlines that
        reached the stopping criterion.
    save_seeds: bool, optional
        True to return the seeds with the associated streamline.

    Returns
    -------
    Tractogram

    """
    sphere = sphere if sphere is not None else default_sphere
    if data is None or model is None:
        raise ValueError("Data and model should be defined.")

    dg = BootDirectionGetter.from_data(
        data,
        model,
        max_angle=max_angle,
    )

    # convert length in mm to number of points
    min_len = int(min_len / step_size)
    max_len = int(max_len / step_size)

    return LocalTracking(
        dg,
        sc,
        seed_positions,
        affine,
        step_size=step_size,
        minlen=min_len,
        maxlen=max_len,
        random_seed=random_seed,
        return_all=return_all,
        initial_directions=seed_directions,
        save_seeds=save_seeds,
    )


def eudx_tracking(
    seed_positions,
    sc,
    affine,
    *,
    seed_directions=None,
    sh=None,
    peaks=None,
    sf=None,
    pam=None,
    min_len=2,
    max_len=500,
    step_size=0.5,
    voxel_size=None,
    max_angle=60,
    pmf_threshold=0.1,
    sphere=None,
    basis_type=None,
    legacy=True,
    nbr_threads=0,
    random_seed=0,
    seed_buffer_fraction=1.0,
    return_all=True,
    save_seeds=False,
):
    """EuDX tracking algorithm.

    seed_positions : ndarray
        Seed positions in world space.
    sc : StoppingCriterion
        Stopping criterion.
    affine : ndarray
        Affine matrix.
    seed_directions : ndarray, optional
        Seed directions.
    sh : ndarray, optional
        Spherical Harmonics (SH).
    peaks : ndarray, optional
        Peaks array.
    sf : ndarray, optional
        Spherical Function (SF).
    pam : PeakAndMetrics, optional
        Peaks and Metrics object
    min_len : int, optional
        Minimum length (mm) of the streamlines.
    max_len : int, optional
        Maximum length (mm) of the streamlines.
    step_size : float, optional
        Step size of the tracking.
    voxel_size : ndarray, optional
        Voxel size.
    max_angle : float, optional
        Maximum angle.
    pmf_threshold : float, optional
        PMF threshold.
    sphere : Sphere, optional
        Sphere.
    basis_type : name of basis
        The basis that ``shcoeff`` are associated with.
        ``dipy.reconst.shm.real_sh_descoteaux`` is used by default.
    legacy: bool, optional
        True to use a legacy basis definition for backward compatibility
        with previous ``tournier07`` and ``descoteaux07`` implementations.
    nbr_threads: int, optional
        Number of threads to use for the processing. By default, all available threads
        will be used.
    random_seed: int, optional
        Seed for the random number generator, must be >= 0. A value of greater than 0
        will all produce the same streamline trajectory for a given seed coordinate.
        A value of 0 may produces various streamline tracjectories for a given seed
        coordinate.
    seed_buffer_fraction: float, optional
        Fraction of the seed buffer to use. A value of 1.0 will use the entire seed
        buffer. A value of 0.5 will use half of the seed buffer then the other half.
        a way to reduce memory usage.
    return_all: bool, optional
        True to return all the streamlines, False to return only the streamlines that
        reached the stopping criterion.
    save_seeds: bool, optional
        True to return the seeds with the associated streamline.

    Returns
    -------
    Tractogram

    """
    sphere = sphere if sphere is not None else default_sphere
    if pam is None:
        raise ValueError("PAM should be defined.")

    # convert length in mm to number of points
    min_len = int(min_len / step_size)
    max_len = int(max_len / step_size)

    return LocalTracking(
        pam,
        sc,
        seed_positions,
        affine,
        step_size=step_size,
        minlen=min_len,
        maxlen=max_len,
        random_seed=random_seed,
        return_all=return_all,
        initial_directions=seed_directions,
        save_seeds=save_seeds,
    )


def pft_tracking(
    seed_positions,
    sc,
    affine,
    *,
    seed_directions=None,
    sh=None,
    peaks=None,
    sf=None,
    pam=None,
    max_cross=None,
    min_len=2,
    max_len=500,
    step_size=0.2,
    voxel_size=None,
    max_angle=20,
    pmf_threshold=0.1,
    sphere=None,
    basis_type=None,
    legacy=True,
    nbr_threads=0,
    random_seed=0,
    seed_buffer_fraction=1.0,
    return_all=True,
    pft_back_tracking_dist=2,
    pft_front_tracking_dist=1,
    pft_max_trial=20,
    particle_count=15,
    save_seeds=False,
    min_wm_pve_before_stopping=0,
    unidirectional=False,
    randomize_forward_direction=False,
):
    """Particle Filtering Tracking (PFT) tracking algorithm.

    seed_positions : ndarray
        Seed positions in world space.
    sc : StoppingCriterion
        Stopping criterion.
    affine : ndarray
        Affine matrix.
    seed_directions : ndarray, optional
        Seed directions.
    sh : ndarray, optional
        Spherical Harmonics (SH).
    peaks : ndarray, optional
        Peaks array.
    sf : ndarray, optional
        Spherical Function (SF).
    pam : PeakAndMetrics, optional
        Peaks and Metrics object.
    max_cross : int, optional
        Maximum number of crossing fibers.
    min_len : int, optional
        Minimum length (mm) of the streamlines.
    max_len : int, optional
        Maximum length (mm) of the streamlines.
    step_size : float, optional
        Step size of the tracking.
    voxel_size : ndarray, optional
        Voxel size.
    max_angle : float, optional
        Maximum angle.
    pmf_threshold : float, optional
        PMF threshold.
    sphere : Sphere, optional
        Sphere.
    basis_type : name of basis
        The basis that ``shcoeff`` are associated with.
        ``dipy.reconst.shm.real_sh_descoteaux`` is used by default.
    legacy: bool, optional
        True to use a legacy basis definition for backward compatibility
        with previous ``tournier07`` and ``descoteaux07`` implementations.
    nbr_threads: int, optional
        Number of threads to use for the processing. By default, all available threads
        will be used.
    random_seed: int, optional
        Seed for the random number generator, must be >= 0. A value of greater than 0
        will all produce the same streamline trajectory for a given seed coordinate.
        A value of 0 may produces various streamline tracjectories for a given seed
        coordinate.
    seed_buffer_fraction: float, optional
        Fraction of the seed buffer to use. A value of 1.0 will use the entire seed
        buffer. A value of 0.5 will use half of the seed buffer then the other half.
        a way to reduce memory usage.
    return_all: bool, optional
        True to return all the streamlines, False to return only the streamlines that
        reached the stopping criterion.
    pft_back_tracking_dist : float, optional
        Back tracking distance.
    pft_front_tracking_dist : float, optional
        Front tracking distance.
    pft_max_trial : int, optional
        Maximum number of trials.
    particle_count : int, optional
        Number of particles.
    save_seeds: bool, optional
        True to return the seeds with the associated streamline.
    min_wm_pve_before_stopping : float, optional
        Minimum white matter partial volume estimation before stopping.
    unidirectional : bool, optional
        True to use unidirectional tracking.
    randomize_forward_direction : bool, optional
        True to randomize forward direction

    Returns
    -------
    Tractogram

    """
    sphere = sphere if sphere is not None else default_sphere

    dg = None
    if sh is not None:
        dg = ProbabilisticDirectionGetter.from_shcoeff(
            sh,
            max_angle=max_angle,
            sphere=sphere,
            sh_to_pmf=True,
            pmf_threshold=pmf_threshold,
            basis_type=basis_type,
            legacy=legacy,
        )
    elif sf is not None:
        dg = ProbabilisticDirectionGetter.from_pmf(
            sf, max_angle=max_angle, sphere=sphere, pmf_threshold=pmf_threshold
        )
    elif pam is not None and sh is None:
        sh = pam.shm_coeff
    else:
        msg = "SH, SF or PAM should be defined. Not implemented yet for peaks."
        raise ValueError(msg)

    # convert length in mm to number of points
    min_len = int(min_len / step_size)
    max_len = int(max_len / step_size)

    return ParticleFilteringTracking(
        dg,
        sc,
        seed_positions,
        affine,
        max_cross=max_cross,
        step_size=step_size,
        minlen=min_len,
        maxlen=max_len,
        pft_back_tracking_dist=pft_back_tracking_dist,
        pft_front_tracking_dist=pft_front_tracking_dist,
        particle_count=particle_count,
        pft_max_trial=pft_max_trial,
        return_all=return_all,
        random_seed=random_seed,
        initial_directions=seed_directions,
        save_seeds=save_seeds,
        min_wm_pve_before_stopping=min_wm_pve_before_stopping,
        unidirectional=unidirectional,
        randomize_forward_direction=randomize_forward_direction,
    )


def _extract_seed_directions_from_peaks(
    seed_positions, peak_directions, peak_values, affine, strength_threshold
):
    """Extract initial tracking directions from peaks at seed positions.

    Parameters
    ----------
    seed_positions : ndarray, shape (N, 3)
        Seed positions in world coordinates.
    peak_directions : ndarray, shape (X, Y, Z, npeaks, 3)
        Peak directions as 3D unit vectors.
    peak_values : ndarray, shape (X, Y, Z, npeaks)
        Peak strength values.
    affine : ndarray, shape (4, 4)
        Affine transformation from world to voxel coordinates.
    strength_threshold : float
        Minimum peak strength to consider.

    Returns
    -------
    seed_directions : ndarray, shape (N, 3)
        Initial directions for each seed.
    """
    import numpy as np
    from nibabel.affines import apply_affine

    # Transform seeds to voxel coordinates
    inv_affine = np.linalg.inv(affine)
    seed_voxels = apply_affine(inv_affine, seed_positions)

    N = len(seed_positions)
    npeaks = peak_directions.shape[3]
    seed_directions = np.zeros((N, 3), dtype=float)

    for i in range(N):
        # Get voxel indices (round to nearest voxel)
        x = int(np.round(seed_voxels[i, 0]))
        y = int(np.round(seed_voxels[i, 1]))
        z = int(np.round(seed_voxels[i, 2]))

        # Check bounds
        if (
            x < 0
            or x >= peak_directions.shape[0]
            or y < 0
            or y >= peak_directions.shape[1]
            or z < 0
            or z >= peak_directions.shape[2]
        ):
            # Outside volume - use default direction
            seed_directions[i] = [1.0, 0.0, 0.0]
            continue

        # Find strongest peak above threshold
        max_val = -1
        max_idx = -1
        for p in range(npeaks):
            if peak_values[x, y, z, p] > max_val and peak_values[x, y, z, p] >= strength_threshold:
                max_val = peak_values[x, y, z, p]
                max_idx = p

        if max_idx >= 0:
            # Use strongest peak direction (normalize it!)
            peak_dir = peak_directions[x, y, z, max_idx]
            norm = np.linalg.norm(peak_dir)
            if norm > 1e-10:
                seed_directions[i] = peak_dir / norm
            else:
                seed_directions[i] = [1.0, 0.0, 0.0]
        else:
            # No valid peak - use default direction
            seed_directions[i] = [1.0, 0.0, 0.0]

    return seed_directions


def glide_tracking(
    seed_positions,
    sc,
    affine,
    *,
    seed_directions=None,
    peak_directions,
    peak_values,
    min_len=2,
    max_len=500,
    step_size=0.5,
    voxel_size=None,
    max_angle=60,
    strength_threshold=0.02,
    interpolation_threshold=0.5,
    nbr_threads=0,
    random_seed=0,
    buffer_frac=1.0,
    return_all=True,
    save_seeds=False,
):
    """Glide tracking - parallelized peak-based deterministic tracking.

    Glide tracking uses the EUDX trilinear interpolation principle with
    direct storage of 3D peak directions for efficient parallelized tracking.
    It follows the same deterministic propagation strategy as EUDX but with
    optimizations for modern multi-core processors.

    Parameters
    ----------
    seed_positions : ndarray, shape (N, 3)
        Seed positions for streamlines in world coordinates.
    sc : StoppingCriterion
        Stopping criterion object that determines when tracking should stop.
    affine : ndarray, shape (4, 4)
        Affine transformation from world to voxel coordinates.
    seed_directions : ndarray, shape (N, 3), optional
        Initial tracking directions for each seed. If None, directions are
        extracted from the strongest peaks at seed locations.
    peak_directions : ndarray, shape (X, Y, Z, npeaks, 3)
        Peak directions as 3D unit vectors at each voxel.
    peak_values : ndarray, shape (X, Y, Z, npeaks)
        Peak strength values (e.g., QA, GFA) at each voxel.
    min_len : float, optional
        Minimum streamline length in mm. Default: 2.
    max_len : float, optional
        Maximum streamline length in mm. Default: 500.
    step_size : float, optional
        Step size for tracking in mm. Default: 0.5.
    voxel_size : ndarray, shape (3,), optional
        Voxel size in mm. If None, extracted from affine.
    max_angle : float, optional
        Maximum angle in degrees between successive tracking steps. Default: 60.
    strength_threshold : float, optional
        Minimum peak strength to consider during tracking. Peaks below this
        value are ignored. Default: 0.02.
    interpolation_threshold : float, optional
        Minimum interpolation weight fraction to continue tracking. Tracking
        stops if less than this fraction of neighboring voxels contribute
        valid directions. Default: 0.5.
    nbr_threads : int, optional
        Number of threads for parallel tracking. 0 uses all available cores.
        Default: 0.
    random_seed : int, optional
        Random seed for reproducible tracking. Default: 0.
    buffer_frac : float, optional
        Fraction of seeds to process in each batch (0 < buffer_frac <= 1).
        Lower values use less memory. Default: 1.0.
    return_all : bool, optional
        If True, return all streamlines including invalid ones. Default: True.
    save_seeds : bool, optional
        If True, yield (streamline, seed) pairs. If False, yield only
        streamlines. Default: False.

    Yields
    ------
    streamlines : ndarray, shape (N_points, 3)
        Generated streamlines in world coordinates.
    seeds : ndarray, shape (3,), optional
        Seed position for each streamline (only if save_seeds=True).

    Examples
    --------
    >>> from dipy.tracking import tracker
    >>> from dipy.tracking.stopping_criterion import ThresholdStoppingCriterion
    >>> # Assume we have peak_dirs and peak_vals from peak detection
    >>> seeds = np.array([[30, 30, 30], [31, 31, 31]])  # in voxel coords
    >>> sc = ThresholdStoppingCriterion(fa_map, 0.2)
    >>> streamlines = tracker.glide_tracking(
    ...     seeds, sc, affine,
    ...     peak_directions=peak_dirs,
    ...     peak_values=peak_vals
    ... )
    >>> streamlines_list = list(streamlines)

    Notes
    -----
    Glide tracking implements the same trilinear interpolation principle as
    EUDX [1]_, but with several optimizations:

    - Direct storage of 3D peak directions (no sphere index lookup)
    - Modern OpenMP-based parallelization via tractogen framework
    - New parameter names for clarity (strength_threshold, interpolation_threshold)

    The tracking stops when:
    - Peak strength < strength_threshold
    - Angle between steps > max_angle
    - Interpolation weight < interpolation_threshold
    - Stopping criterion is met

    References
    ----------
    .. [1] Garyfallidis E., "Towards an accurate brain tractography", PhD
           thesis, University of Cambridge, 2012.
    """
    # Validate inputs
    peak_directions = np.asarray(peak_directions, dtype=float, order="C")
    peak_values = np.asarray(peak_values, dtype=float, order="C")

    if peak_directions.ndim != 5 or peak_directions.shape[-1] != 3:
        raise ValueError("peak_directions must be shape (X,Y,Z,npeaks,3)")
    if peak_values.ndim != 4:
        raise ValueError("peak_values must be shape (X,Y,Z,npeaks)")
    if peak_directions.shape[:4] != peak_values.shape:
        raise ValueError("peak_directions and peak_values must have matching spatial and peak dimensions")

    # Get voxel size from affine if not provided
    if voxel_size is None:
        voxel_size = voxel_sizes(affine)

    # Create tracking parameters with glide propagator
    params = generate_tracking_parameters(
        "glide",
        min_len=min_len,
        max_len=max_len,
        step_size=step_size,
        voxel_size=voxel_size,
        max_angle=max_angle,
        strength_threshold=strength_threshold,
        interpolation_threshold=interpolation_threshold,
        random_seed=random_seed,
        return_all=return_all,
    )

    # Create PeakDirectionGen data container
    peak_gen = PeakDirectionGen(peak_directions, peak_values)

    # Handle seed directions
    if seed_directions is None:
        # Extract initial directions from peaks at seed locations
        seed_directions = _extract_seed_directions_from_peaks(
            seed_positions, peak_directions, peak_values, affine, strength_threshold
        )

    # Ensure seed_directions is properly formatted
    seed_directions = np.asarray(seed_directions, dtype=float, order="C")

    # Call glide-specific tractogen function
    return generate_tractogram_glide(
        seed_positions,
        seed_directions,
        sc,
        params,
        peak_gen,
        affine=affine,
        nbr_threads=nbr_threads,
        buffer_frac=buffer_frac,
        save_seeds=save_seeds,
    )
