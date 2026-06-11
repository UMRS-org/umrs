from umrs.vision_service.output_specs import describe_output_artifacts


def test_sam3_outputs_are_deterministic():
    result = describe_output_artifacts("sam3", "images/input.png", {})
    assert "images/input_sam3_seg.png" in result
    assert result["images/input_sam3_seg.png"] == "image/png"
    assert "images/input_sam3_results.pkl" in result
    assert result["images/input_sam3_results.pkl"] == "application/octet-stream"


def test_hmae_reconstruction_outputs_are_deterministic():
    result = describe_output_artifacts("hmae", "ct/sample.mhd", {"task": "reconstruction"})
    assert "ct/sample_recon.png" in result
    assert result["ct/sample_recon.png"] == "image/png"
    assert "ct/sample_recon_results.pkl" in result


def test_hmae_mode_aliases_resolve_to_expected_outputs():
    mae_result = describe_output_artifacts("hmae", "ct/sample.mhd", {"mode": "mae"})
    assert "ct/sample_mae.png" in mae_result
    assert mae_result["ct/sample_mae.png"] == "image/png"

    feat_result = describe_output_artifacts("hmae", "ct/sample.mhd", {"mode": "features"})
    assert "ct/sample_features.pt" in feat_result
    assert feat_result["ct/sample_features.pt"] == "application/octet-stream"


def test_huggingface_classification_outputs():
    result = describe_output_artifacts(
        "huggingface", "img/test.jpg", {"task": "image-classification"}
    )
    assert "img/test_hf_results.pkl" in result
    assert result["img/test_hf_results.pkl"] == "application/octet-stream"


def test_huggingface_segmentation_outputs():
    result = describe_output_artifacts(
        "huggingface", "img/test.jpg", {"task": "image-segmentation"}
    )
    assert "img/test_seg.png" in result
    assert result["img/test_seg.png"] == "image/png"
    assert "img/test_hf_results.pkl" in result


def test_huggingface_depth_outputs():
    result = describe_output_artifacts("huggingface", "img/scene.png", {"task": "depth-estimation"})
    assert "img/scene_depth.png" in result
    assert result["img/scene_depth.png"] == "image/png"


def test_huggingface_feature_outputs():
    result = describe_output_artifacts(
        "huggingface", "img/photo.jpg", {"task": "image-feature-extraction"}
    )
    assert "img/photo_features.pt" in result
    assert result["img/photo_features.pt"] == "application/octet-stream"


def test_unknown_model_type_has_no_declared_outputs():
    assert describe_output_artifacts("custom", "sample.bin", {}) == {}
