from umrs.vision_service.output_specs import describe_output_artifacts


def test_sam3_outputs_are_deterministic():
    assert describe_output_artifacts("sam3", "images/input.png", {}) == {
        "images/input_sam3_seg.png": "image/png",
    }


def test_hmae_reconstruction_outputs_are_deterministic():
    assert describe_output_artifacts("hmae", "ct/sample.mhd", {"task": "reconstruction"}) == {
        "ct/sample_recon.png": "image/png",
    }


def test_hmae_mode_aliases_resolve_to_expected_outputs():
    assert describe_output_artifacts("hmae", "ct/sample.mhd", {"mode": "mae"}) == {
        "ct/sample_mae.png": "image/png",
    }
    assert describe_output_artifacts("hmae", "ct/sample.mhd", {"mode": "features"}) == {
        "ct/sample_features.pt": "application/octet-stream",
    }


def test_unknown_model_type_has_no_declared_outputs():
    assert describe_output_artifacts("custom", "sample.bin", {}) == {}