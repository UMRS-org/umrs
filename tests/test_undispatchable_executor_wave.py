"""WP-09 Phase FINE: fine-tune spec validator pins (gate-held until training wrapper lands."""

import pytest


class TestFineTuneSpecChain:
    def test_rank_whitelist_and_epochs_and_dataset_ref(self):
        from umrs.jobs.fine_tune import _validate_spec

        ctx_shape = _validate_spec(
            {
                "target_model": "mdl-1",
                "dataset_ref": "s3://umrs-files/datasets/x.jsonl",
                "adapter_name": "lora1",
                "rank": 16,
                "epochs": 1,
                "max_seq_len": 2048,
            }
        )
        assert ctx_shape["target_model"] == "mdl-1"

    def test_inline_dataset_rejected_s3_only(self):
        from umrs.jobs.fine_tune import _validate_spec

        with pytest.raises(ValueError, match="dataset_ref"):
            _validate_spec(
                {
                    "target_model": "mdl-1",
                    "dataset_ref": "inline-text-choice",
                    "adapter_name": "x",
                }
            )

    def test_rank_out_of_whitelist_rejected(self):
        from umrs.jobs.fine_tune import _validate_spec

        with pytest.raises(ValueError, match="rank"):
            _validate_spec(
                {
                    "target_model": "mdl-1",
                    "dataset_ref": "s3://umrs-files/datasets/x.jsonl",
                    "adapter_name": "x",
                    "rank": 12,
                }
            )

    def test_epochs_bound(self):
        from umrs.jobs.fine_tune import _validate_spec

        with pytest.raises(ValueError, match="epochs"):
            _validate_spec(
                {
                    "target_model": "mdl-1",
                    "dataset_ref": "s3://umrs-files/datasets/x.jsonl",
                    "adapter_name": "x",
                    "epochs": 4,
                }
            )

    def test_other_site_rejected_phase_f(self):
        from umrs.jobs.fine_tune import _validate_spec

        with pytest.raises(ValueError, match="pin"):
            _validate_spec(
                {
                    "target_model": "mdl-1",
                    "dataset_ref": "s3://umrs-files/datasets/x.jsonl",
                    "adapter_name": "x",
                    "site": "rccs",
                }
            )

    def test_missing_target_model_rejected(self):
        from umrs.jobs.fine_tune import _validate_spec

        with pytest.raises(ValueError, match="target_model"):
            _validate_spec({"dataset_ref": "s3://b/k", "adapter_name": "x"})
