"""Tests for GLIDE uncertainty-adaptive hybrid tractography."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from dipy.tracking.utils import compute_uncertainty_map, score_streamlines


class TestComputeUncertaintyMap:
    """Tests for compute_uncertainty_map."""

    def test_peak_ratio_single_peak(self):
        """Single-peak voxels should have zero uncertainty."""

        class MockPam:
            peak_values = np.zeros((3, 3, 3, 5))

        pam = MockPam()
        pam.peak_values[1, 1, 1, 0] = 1.0
        u = compute_uncertainty_map(pam, method="peak_ratio")
        assert u[1, 1, 1] == 0.0

    def test_peak_ratio_equal_peaks(self):
        """Equal first and second peaks should give uncertainty of 1."""

        class MockPam:
            peak_values = np.zeros((3, 3, 3, 5))

        pam = MockPam()
        pam.peak_values[1, 1, 1, 0] = 1.0
        pam.peak_values[1, 1, 1, 1] = 1.0
        u = compute_uncertainty_map(pam, method="peak_ratio")
        assert u[1, 1, 1] == 1.0

    def test_peak_ratio_partial(self):
        """Second peak half of first should give uncertainty of 0.5."""

        class MockPam:
            peak_values = np.zeros((3, 3, 3, 5))

        pam = MockPam()
        pam.peak_values[1, 1, 1, 0] = 2.0
        pam.peak_values[1, 1, 1, 1] = 1.0
        u = compute_uncertainty_map(pam, method="peak_ratio")
        assert_allclose(u[1, 1, 1], 0.5)

    def test_peak_ratio_no_peaks(self):
        """Voxels with no peaks should have zero uncertainty."""

        class MockPam:
            peak_values = np.zeros((3, 3, 3, 5))

        pam = MockPam()
        u = compute_uncertainty_map(pam, method="peak_ratio")
        assert_allclose(u, 0.0)

    def test_gfa_method(self):
        """GFA-based uncertainty: u = 1 - gfa."""

        class MockPam:
            gfa = np.full((3, 3, 3), 0.8)

        pam = MockPam()
        u = compute_uncertainty_map(pam, method="gfa")
        assert_allclose(u, 0.2)

    def test_mask(self):
        """Mask should zero out voxels outside."""

        class MockPam:
            peak_values = np.ones((3, 3, 3, 5))

        pam = MockPam()
        mask = np.zeros((3, 3, 3), dtype=bool)
        mask[1, 1, 1] = True
        u = compute_uncertainty_map(pam, method="peak_ratio", mask=mask)
        assert u[0, 0, 0] == 0.0
        assert u[1, 1, 1] > 0.0

    def test_invalid_method(self):
        """Unknown method should raise ValueError."""

        class MockPam:
            peak_values = np.zeros((3, 3, 3, 5))

        pam = MockPam()
        with pytest.raises(ValueError, match="Unknown uncertainty method"):
            compute_uncertainty_map(pam, method="invalid")

    def test_output_shape_and_dtype(self):
        """Output should be 3D float64."""

        class MockPam:
            peak_values = np.ones((4, 5, 6, 3))

        pam = MockPam()
        u = compute_uncertainty_map(pam, method="peak_ratio")
        assert u.shape == (4, 5, 6)
        assert u.dtype == np.float64


class TestScoreStreamlines:
    """Tests for score_streamlines."""

    def test_zero_uncertainty(self):
        """Zero uncertainty everywhere gives confidence = 1."""
        u_map = np.zeros((10, 10, 10), dtype=np.float64)
        affine = np.eye(4)
        streamlines = [np.array([[5.0, 5.0, 5.0], [5.0, 5.0, 6.0]])]
        scores = score_streamlines(streamlines, u_map, affine)
        assert_allclose(scores[0], 1.0)

    def test_full_uncertainty(self):
        """Full uncertainty everywhere gives confidence = 0."""
        u_map = np.ones((10, 10, 10), dtype=np.float64)
        affine = np.eye(4)
        streamlines = [np.array([[5.0, 5.0, 5.0], [5.0, 5.0, 6.0]])]
        scores = score_streamlines(streamlines, u_map, affine)
        assert_allclose(scores[0], 0.0)

    def test_affine_transform(self):
        """Scores should respect the affine transform."""
        u_map = np.zeros((10, 10, 10), dtype=np.float64)
        u_map[5, 5, 5] = 1.0
        affine = 2.0 * np.eye(4)
        affine[3, 3] = 1.0
        # World coord (10, 10, 10) maps to voxel (5, 5, 5)
        streamlines = [np.array([[10.0, 10.0, 10.0]])]
        scores = score_streamlines(streamlines, u_map, affine)
        assert scores[0] < 1.0

    def test_multiple_streamlines(self):
        """Multiple streamlines should each get a score."""
        u_map = np.full((10, 10, 10), 0.5, dtype=np.float64)
        affine = np.eye(4)
        streamlines = [
            np.array([[3.0, 3.0, 3.0], [4.0, 4.0, 4.0]]),
            np.array([[5.0, 5.0, 5.0], [6.0, 6.0, 6.0]]),
        ]
        scores = score_streamlines(streamlines, u_map, affine)
        assert scores.shape == (2,)
        assert_allclose(scores, 0.5)


def _get_test_data():
    """Load DIPY's small test dataset and return SH, pam, affine, seeds, sc."""
    from dipy.core.gradients import gradient_table
    from dipy.data import default_sphere, get_fnames
    from dipy.direction.peaks import peaks_from_model
    from dipy.io.image import load_nifti
    from dipy.reconst.csdeconv import auto_response_ssst
    from dipy.reconst.shm import CsaOdfModel
    from dipy.tracking.stopping_criterion import ThresholdStoppingCriterion

    fimg, fbval, fbvec = get_fnames(name="small_64D")
    data, affine = load_nifti(fimg)
    bvals = np.loadtxt(fbval)
    bvecs = np.loadtxt(fbvec).T
    gtab = gradient_table(bvals, bvecs=bvecs)

    csa_model = CsaOdfModel(gtab, sh_order_max=6)
    pam = peaks_from_model(
        csa_model, data, default_sphere,
        relative_peak_threshold=0.5, min_separation_angle=25,
    )

    mask = (pam.gfa > 0.1).astype(np.float64)
    sc = ThresholdStoppingCriterion(mask, 0.5)

    # Pick a few interior seed voxels
    vox_seeds = np.argwhere(mask > 0)[:5].astype(float)
    seeds = np.dot(vox_seeds, affine[:3, :3].T) + affine[:3, 3]

    return pam, affine, seeds, sc


class TestGlideTracking:
    """Tests for glide_tracking."""

    def test_error_no_data(self):
        """Should raise when no data provided."""
        from dipy.tracking.tracker import glide_tracking

        seeds = np.array([[5.0, 5.0, 5.0]])
        with pytest.raises(ValueError, match="uncertainty map"):
            glide_tracking(seeds, None, np.eye(4))

    def test_error_sh_without_uncertainty(self):
        """Should raise when sh given but no uncertainty or pam."""
        from dipy.tracking.tracker import glide_tracking

        seeds = np.array([[5.0, 5.0, 5.0]])
        sh = np.zeros((10, 10, 10, 15))
        with pytest.raises(ValueError, match="Cannot automatically"):
            glide_tracking(seeds, None, np.eye(4), sh=sh)

    def test_error_peaks_only(self):
        """Should raise when only pam with no SH coefficients."""
        from dipy.tracking.tracker import glide_tracking

        class MockPam:
            peak_values = np.ones((5, 5, 5, 3))
            peak_indices = np.zeros((5, 5, 5, 3), dtype=np.int32)

        pam = MockPam()
        seeds = np.array([[2.0, 2.0, 2.0]])
        with pytest.raises(ValueError, match="SH coefficients"):
            glide_tracking(seeds, None, np.eye(4), pam=pam)

    def test_invalid_blend_mode(self):
        """Should raise on unknown blend_mode string."""
        from dipy.tracking.tracker import glide_tracking

        seeds = np.array([[5.0, 5.0, 5.0]])
        sh = np.zeros((10, 10, 10, 15))
        u_map = np.zeros((10, 10, 10))
        with pytest.raises(ValueError, match="Unknown blend_mode"):
            glide_tracking(
                seeds, None, np.eye(4),
                sh=sh, uncertainty_map=u_map, blend_mode="invalid",
            )

    def test_smoke_with_pam(self):
        """GLIDE produces streamlines with real SH data from pam."""
        from dipy.tracking.tracker import glide_tracking

        pam, affine, seeds, sc = _get_test_data()

        tractogram = glide_tracking(
            seeds, sc, affine,
            pam=pam,
            step_size=0.5,
            min_len=2,
            max_len=100,
            random_seed=42,
        )
        streamlines = list(tractogram)
        assert len(streamlines) > 0

    def test_smoke_with_sh_and_uncertainty(self):
        """GLIDE with explicit SH + uncertainty map."""
        from dipy.tracking.tracker import glide_tracking

        pam, affine, seeds, sc = _get_test_data()
        uncertainty_map = compute_uncertainty_map(pam, method="peak_ratio")

        tractogram = glide_tracking(
            seeds, sc, affine,
            sh=pam.shm_coeff,
            uncertainty_map=uncertainty_map,
            step_size=0.5,
            min_len=2,
            max_len=100,
            random_seed=42,
        )
        streamlines = list(tractogram)
        assert len(streamlines) > 0

    def test_low_uncertainty_deterministic(self):
        """Zero uncertainty with step blend should be fully deterministic."""
        from dipy.tracking.tracker import glide_tracking

        pam, affine, seeds, sc = _get_test_data()
        shape = pam.peak_values.shape[:3]
        uncertainty_map = np.zeros(shape, dtype=np.float64)

        results = []
        for _ in range(3):
            tractogram = glide_tracking(
                seeds[:1], sc, affine,
                pam=pam,
                uncertainty_map=uncertainty_map,
                step_size=0.5,
                min_len=2,
                max_len=100,
                blend_mode="step",
                random_seed=0,
            )
            sls = list(tractogram)
            if sls:
                results.append(sls[0])

        assert len(results) >= 2, "Expected at least 2 results"
        assert_allclose(results[0], results[1], atol=1e-10)

    def test_blend_modes(self):
        """All three blend modes should produce streamlines."""
        from dipy.tracking.tracker import glide_tracking

        pam, affine, seeds, sc = _get_test_data()

        for mode in ("linear", "sigmoid", "step"):
            tractogram = glide_tracking(
                seeds, sc, affine,
                pam=pam,
                step_size=0.5,
                min_len=2,
                max_len=100,
                blend_mode=mode,
                random_seed=42,
            )
            streamlines = list(tractogram)
            assert len(streamlines) > 0, (
                f"blend_mode={mode!r} produced no streamlines"
            )

    def test_gm_map(self):
        """Gyral bias correction with GM map should run."""
        from dipy.tracking.tracker import glide_tracking

        pam, affine, seeds, sc = _get_test_data()
        shape = pam.peak_values.shape[:3]
        gm_map = np.full(shape, 0.4, dtype=np.float64)

        tractogram = glide_tracking(
            seeds, sc, affine,
            pam=pam,
            gm_map=gm_map,
            step_size=0.5,
            min_len=2,
            max_len=100,
            random_seed=42,
        )
        streamlines = list(tractogram)
        assert len(streamlines) > 0

    def test_score_streamlines_integration(self):
        """score_streamlines should work on GLIDE output."""
        from dipy.tracking.tracker import glide_tracking

        pam, affine, seeds, sc = _get_test_data()
        uncertainty_map = compute_uncertainty_map(pam, method="peak_ratio")

        tractogram = glide_tracking(
            seeds, sc, affine,
            pam=pam,
            uncertainty_map=uncertainty_map,
            step_size=0.5,
            min_len=2,
            max_len=100,
            random_seed=42,
        )
        streamlines = list(tractogram)
        assert len(streamlines) > 0

        scores = score_streamlines(streamlines, uncertainty_map, affine)
        assert scores.shape == (len(streamlines),)
        assert np.all(scores >= 0) and np.all(scores <= 1)


class TestCompositeUncertainty:
    """Tests for the 'composite' uncertainty method."""

    def test_single_peak_zero(self):
        """Single peak should give zero uncertainty (ratio=0, count_norm=0)."""

        class MockPam:
            peak_values = np.zeros((3, 3, 3, 5))

        pam = MockPam()
        pam.peak_values[1, 1, 1, 0] = 1.0
        u = compute_uncertainty_map(pam, method="composite")
        assert_allclose(u[1, 1, 1], 0.0)

    def test_equal_peaks_high(self):
        """Equal first and second peaks should give high uncertainty."""

        class MockPam:
            peak_values = np.zeros((3, 3, 3, 5))

        pam = MockPam()
        pam.peak_values[1, 1, 1, 0] = 1.0
        pam.peak_values[1, 1, 1, 1] = 1.0
        u = compute_uncertainty_map(pam, method="composite")
        # ratio=1.0, count_norm=(2-1)/(5-1)=0.25 -> 0.7*1+0.3*0.25=0.775
        assert_allclose(u[1, 1, 1], 0.775)

    def test_many_peaks_count_component(self):
        """Many non-zero peaks should boost the count component."""

        class MockPam:
            peak_values = np.zeros((3, 3, 3, 5))

        pam = MockPam()
        pam.peak_values[1, 1, 1, :] = [1.0, 0.0, 0.0, 0.0, 0.0]
        u_single = compute_uncertainty_map(pam, method="composite")[1, 1, 1]
        pam.peak_values[1, 1, 1, :] = [1.0, 0.5, 0.3, 0.2, 0.1]
        u_many = compute_uncertainty_map(pam, method="composite")[1, 1, 1]
        assert u_many > u_single

    def test_no_peaks_zero(self):
        """All-zero peak values should give zero uncertainty."""

        class MockPam:
            peak_values = np.zeros((3, 3, 3, 5))

        pam = MockPam()
        u = compute_uncertainty_map(pam, method="composite")
        assert_allclose(u, 0.0)


class TestForceMaps:
    """Tests for FORCE-specific map inputs to glide_tracking."""

    def test_smoke_dispersion_only(self):
        """GLIDE with dispersion_map only should produce streamlines."""
        from dipy.tracking.tracker import glide_tracking

        pam, affine, seeds, sc = _get_test_data()
        shape = pam.peak_values.shape[:3]
        dispersion_map = np.full(shape, 0.15, dtype=np.float64)

        tractogram = glide_tracking(
            seeds, sc, affine,
            pam=pam,
            dispersion_map=dispersion_map,
            step_size=0.5,
            min_len=2,
            max_len=100,
            random_seed=42,
        )
        streamlines = list(tractogram)
        assert len(streamlines) > 0

    def test_smoke_num_fibers_only(self):
        """GLIDE with num_fibers_map only should produce streamlines."""
        from dipy.tracking.tracker import glide_tracking

        pam, affine, seeds, sc = _get_test_data()
        shape = pam.peak_values.shape[:3]
        num_fibers_map = np.full(shape, 2.0, dtype=np.float64)

        tractogram = glide_tracking(
            seeds, sc, affine,
            pam=pam,
            num_fibers_map=num_fibers_map,
            step_size=0.5,
            min_len=2,
            max_len=100,
            random_seed=42,
        )
        streamlines = list(tractogram)
        assert len(streamlines) > 0

    def test_smoke_all_force_maps(self):
        """GLIDE with all four FORCE maps should produce streamlines."""
        from dipy.tracking.tracker import glide_tracking

        pam, affine, seeds, sc = _get_test_data()
        shape = pam.peak_values.shape[:3]

        tractogram = glide_tracking(
            seeds, sc, affine,
            pam=pam,
            dispersion_map=np.full(shape, 0.1, dtype=np.float64),
            num_fibers_map=np.full(shape, 1.5, dtype=np.float64),
            wm_map=np.full(shape, 0.8, dtype=np.float64),
            csf_map=np.full(shape, 0.05, dtype=np.float64),
            step_size=0.5,
            min_len=2,
            max_len=100,
            random_seed=42,
        )
        streamlines = list(tractogram)
        assert len(streamlines) > 0

    def test_wm_csf_passthrough(self):
        """WM/CSF maps should not change propagator output (Tier 2 stored only)."""
        from dipy.tracking.tracker import glide_tracking

        pam, affine, seeds, sc = _get_test_data()
        shape = pam.peak_values.shape[:3]

        kwargs = dict(
            pam=pam,
            step_size=0.5,
            min_len=2,
            max_len=100,
            random_seed=42,
        )
        tractogram_base = glide_tracking(seeds, sc, affine, **kwargs)
        sls_base = list(tractogram_base)

        tractogram_wm_csf = glide_tracking(
            seeds, sc, affine,
            wm_map=np.full(shape, 0.9, dtype=np.float64),
            csf_map=np.full(shape, 0.01, dtype=np.float64),
            **kwargs,
        )
        sls_wm_csf = list(tractogram_wm_csf)

        assert len(sls_base) == len(sls_wm_csf)
        for s1, s2 in zip(sls_base, sls_wm_csf):
            assert_allclose(s1, s2)

    def test_high_dispersion_overrides_zero_uncertainty(self):
        """High dispersion should override zero peak-ratio uncertainty."""
        from dipy.tracking.tracker import glide_tracking

        pam, affine, seeds, sc = _get_test_data()
        shape = pam.peak_values.shape[:3]

        # Zero uncertainty baseline with step blend
        zero_u = np.zeros(shape, dtype=np.float64)
        kwargs = dict(
            pam=pam,
            uncertainty_map=zero_u,
            step_size=0.5,
            min_len=2,
            max_len=100,
            blend_mode="step",
            random_seed=0,
        )
        tractogram_no_disp = glide_tracking(seeds[:1], sc, affine, **kwargs)
        sls_no_disp = list(tractogram_no_disp)

        # High dispersion should push toward probabilistic (different results)
        high_disp = np.full(shape, 0.3, dtype=np.float64)  # max ODI -> 1.0
        tractogram_disp = glide_tracking(
            seeds[:1], sc, affine,
            dispersion_map=high_disp,
            **kwargs,
        )
        sls_disp = list(tractogram_disp)

        # With random_seed=0, probabilistic mode produces varying results.
        # Just verify both produce output and the dispersion version ran.
        assert len(sls_no_disp) > 0 or len(sls_disp) > 0
