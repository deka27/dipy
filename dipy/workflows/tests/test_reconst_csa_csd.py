import logging
from pathlib import Path
from tempfile import TemporaryDirectory
import warnings

import numpy as np
import numpy.testing as npt

from dipy.core.gradients import generate_bvecs
from dipy.data import get_fnames
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti, load_nifti_data, save_nifti
from dipy.io.peaks import load_pam
from dipy.reconst.shm import descoteaux07_legacy_msg, sph_harm_ind_list
from dipy.workflows.reconst import ReconstCSDFlow, ReconstQBallBaseFlow, ReconstSDTFlow

logging.getLogger().setLevel(logging.INFO)


def test_reconst_csa():
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=descoteaux07_legacy_msg,
            category=PendingDeprecationWarning,
        )
        reconst_flow_core(ReconstQBallBaseFlow)


def test_reconst_opdt():
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=descoteaux07_legacy_msg,
            category=PendingDeprecationWarning,
        )
        reconst_flow_core(ReconstQBallBaseFlow, method="opdt")


def test_reconst_qball():
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=descoteaux07_legacy_msg,
            category=PendingDeprecationWarning,
        )
        reconst_flow_core(ReconstQBallBaseFlow, method="qball")


def test_reconst_csd():
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=descoteaux07_legacy_msg,
            category=PendingDeprecationWarning,
        )
        reconst_flow_core(ReconstCSDFlow)


def test_reconst_sdt():
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=descoteaux07_legacy_msg,
            category=PendingDeprecationWarning,
        )
        reconst_flow_core(ReconstSDTFlow)


def reconst_flow_core(flow, **kwargs):
    with TemporaryDirectory() as out_dir:
        data_path, bval_path, bvec_path = get_fnames(name="small_64D")
        volume, affine = load_nifti(data_path)
        mask = np.ones_like(volume[:, :, :, 0])
        mask_path = Path(out_dir) / "tmp_mask.nii.gz"
        save_nifti(mask_path, mask.astype(np.uint8), affine)

        reconst_flow = flow()
        for sh_order in [4, 6, 8]:
            reconst_flow.run(
                data_path,
                bval_path,
                bvec_path,
                mask_path,
                sh_order_max=sh_order,
                out_dir=out_dir,
                extract_pam_values=True,
                **kwargs,
            )

            gfa_path = reconst_flow.last_generated_outputs["out_gfa"]
            gfa_data = load_nifti_data(gfa_path)
            npt.assert_equal(gfa_data.shape, volume.shape[:-1])

            peaks_dir_path = reconst_flow.last_generated_outputs["out_peaks_dir"]
            peaks_dir_data = load_nifti_data(peaks_dir_path)
            npt.assert_equal(peaks_dir_data.shape[-1], 15)
            npt.assert_equal(peaks_dir_data.shape[:-1], volume.shape[:-1])

            peaks_idx_path = reconst_flow.last_generated_outputs["out_peaks_indices"]
            peaks_idx_data = load_nifti_data(peaks_idx_path)
            npt.assert_equal(peaks_idx_data.shape[-1], 5)
            npt.assert_equal(peaks_idx_data.shape[:-1], volume.shape[:-1])

            peaks_vals_path = reconst_flow.last_generated_outputs["out_peaks_values"]
            peaks_vals_data = load_nifti_data(peaks_vals_path)
            npt.assert_equal(peaks_vals_data.shape[-1], 5)
            npt.assert_equal(peaks_vals_data.shape[:-1], volume.shape[:-1])

            shm_path = reconst_flow.last_generated_outputs["out_shm"]
            shm_data = load_nifti_data(shm_path)
            # Test that the number of coefficients is what you would expect
            # given the order of the sh basis:
            npt.assert_equal(
                shm_data.shape[-1], sph_harm_ind_list(sh_order)[0].shape[0]
            )
            npt.assert_equal(shm_data.shape[:-1], volume.shape[:-1])

            pam = load_pam(reconst_flow.last_generated_outputs["out_pam"])
            npt.assert_allclose(
                pam.peak_dirs.reshape(peaks_dir_data.shape), peaks_dir_data
            )
            npt.assert_allclose(pam.peak_values, peaks_vals_data)
            npt.assert_allclose(pam.peak_indices, peaks_idx_data)
            npt.assert_allclose(pam.shm_coeff, shm_data)
            npt.assert_allclose(pam.gfa, gfa_data)

            bvals, bvecs = read_bvals_bvecs(bval_path, bvec_path)
            bvals[0] = 5.0
            bvecs = generate_bvecs(len(bvals))

            tmp_bval_path = Path(out_dir) / "tmp.bval"
            tmp_bvec_path = Path(out_dir) / "tmp.bvec"
            np.savetxt(tmp_bval_path, bvals)
            np.savetxt(tmp_bvec_path, bvecs.T)
            reconst_flow._force_overwrite = True

            if flow.get_short_name() == "csd":
                reconst_flow = flow()
                reconst_flow._force_overwrite = True
                reconst_flow.run(
                    data_path,
                    bval_path,
                    bvec_path,
                    mask_path,
                    out_dir=out_dir,
                    frf=[15, 5, 5],
                    **kwargs,
                )
                reconst_flow = flow()
                reconst_flow._force_overwrite = True
                reconst_flow.run(
                    data_path,
                    bval_path,
                    bvec_path,
                    mask_path,
                    out_dir=out_dir,
                    frf="15, 5, 5",
                    **kwargs,
                )
                reconst_flow = flow()
                reconst_flow._force_overwrite = True
                reconst_flow.run(
                    data_path,
                    bval_path,
                    bvec_path,
                    mask_path,
                    out_dir=out_dir,
                    frf=None,
                    **kwargs,
                )
                reconst_flow2 = flow()
                reconst_flow2._force_overwrite = True
                reconst_flow2.run(
                    data_path,
                    bval_path,
                    bvec_path,
                    mask_path,
                    out_dir=out_dir,
                    frf=None,
                    roi_center=[5, 5, 5],
                    **kwargs,
                )
            else:
                with npt.assert_raises(BaseException):
                    npt.assert_warns(
                        UserWarning,
                        reconst_flow.run,
                        data_path,
                        tmp_bval_path,
                        tmp_bvec_path,
                        mask_path,
                        out_dir=out_dir,
                        extract_pam_values=True,
                        **kwargs,
                    )

            # test parallel implementation
            # Avoid SDT for now, as it is quite slow, something to introspect
            if flow.get_short_name() != "sdt":
                reconst_flow = flow()
                reconst_flow._force_overwrite = True
                reconst_flow.run(
                    data_path,
                    bval_path,
                    bvec_path,
                    mask_path,
                    out_dir=out_dir,
                    parallel=True,
                    num_processes=2,
                    **kwargs,
                )
