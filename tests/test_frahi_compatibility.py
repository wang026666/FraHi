"""FraHi configuration alias and legacy checkpoint compatibility checks."""

from copy import deepcopy
from pathlib import Path
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config_utils import validate_config, validate_training_runtime


def valid_config():
    """Minimal valid full-model configuration for validation tests."""
    return (
        {
            "proc_dp": "data/proc",
            "mz_bin_res": 0.1,
            "mz_max": 500.0,
            "nist_version": 20,
            "merge_collision_energies": False,
            "peak_target_max": 256,
            "require_fragment_candidates": True,
            "fragment_hydrogen_shifts": [-2, -1, 0, 1, 2],
            "fragment_max_candidates": 512,
        },
        {
            "model_seed": 999,
            "embed_types": ["hdse"],
            "prediction_head": "frahi",
            "protect_baseline": True,
            "local_residual_log_ratio_max": 0.5,
            "hierarchical_residual_log_ratio_max": 0.15,
            "hierarchical_adapter_ratio": 0.25,
        },
        {
            "device": "cpu",
            "batch_size": 2,
            "grad_acc_interval": 1,
            "num_epochs": 1,
            "do_casmi22": False,
            "pretrained": None,
            "baseline_loss_weight": 1.0,
            "baseline_clip_grad_norm": 100000.0,
            "validation_fallback": True,
            "scheduler": "plateau",
            "scheduler_monitor": "baseline",
            "fragment_match_tolerance": 0.2,
        },
    )


class FraHiConfigCompatibilityTests(unittest.TestCase):
    def test_new_and_legacy_names_validate_without_mutation(self):
        for head in ("frahi", "hipeak"):
            with self.subTest(head=head):
                data, model, run = valid_config()
                model["prediction_head"] = head
                original = deepcopy((data, model, run))
                validate_config(data, model, run)
                validate_training_runtime(model, run)
                self.assertEqual((data, model, run), original)

    def test_unknown_head_is_rejected(self):
        data, model, run = valid_config()
        model["prediction_head"] = "unknown"
        with self.assertRaisesRegex(ValueError, "prediction_head must be"):
            validate_config(data, model, run)

    def test_alias_requires_fragment_candidates(self):
        data, model, run = valid_config()
        data["require_fragment_candidates"] = False
        with self.assertRaisesRegex(ValueError, "require_fragment_candidates"):
            validate_config(data, model, run)

    def test_alias_requires_collision_energy_resolved_data(self):
        data, model, run = valid_config()
        data["merge_collision_energies"] = True
        with self.assertRaisesRegex(ValueError, "merge_collision_energies=false"):
            validate_config(data, model, run)

    def test_alias_retains_hierarchy_bounds(self):
        for field, invalid_value in (
            ("hierarchical_residual_log_ratio_max", 0.6),
            ("hierarchical_adapter_ratio", 0.0),
        ):
            with self.subTest(field=field):
                data, model, run = valid_config()
                model[field] = invalid_value
                with self.assertRaisesRegex(ValueError, field):
                    validate_config(data, model, run)

    def test_alias_rejects_pretrained_protected_training(self):
        _, model, run = valid_config()
        run["pretrained"] = "checkpoint.pth"
        with self.assertRaisesRegex(ValueError, "pretrained must be empty"):
            validate_training_runtime(model, run)

    def test_alias_allows_checkpoint_evaluation(self):
        data, model, run = valid_config()
        run.update(num_epochs=0, pretrained="checkpoint.pth")
        validate_config(data, model, run)
        validate_training_runtime(model, run)


class FraHiModelCompatibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
            from model import Prediction
        except ImportError as exc:
            raise unittest.SkipTest(
                f"model dependency stack unavailable: {exc}"
            ) from exc
        cls.torch = torch
        cls.Prediction = Prediction

    def make_model(self, head):
        # Real model modules with smaller layer counts; no dataset or mocks.
        return self.Prediction(
            {"g_dim": 4, "o_dim": 50},
            prediction_head=head,
            embed_types=["hdse"],
            embed_dim=-1,
            embed_linear=False,
            ff_h_dim=32,
            ff_num_layers=1,
            layers=1,
            layer_type="GINE+Transformer",
            dropout=0.0,
            output_normalization="l1",
            prec_mass_offset=0,
            mz_bin_res=0.1,
            mz_max=5.0,
            peak_decoder_layers=1,
            peak_decoder_dropout=0.0,
        )

    def test_legacy_state_loads_strictly_into_frahi(self):
        legacy = self.make_model("hipeak")
        legacy.set_inference_prediction("baseline")
        renamed = self.make_model("frahi")
        renamed.load_checkpoint(legacy.state_dict())
        self.assertEqual(renamed.prediction_head, "hipeak")
        self.assertEqual(renamed.inference_prediction_source(), "baseline")
        self.assertEqual(legacy.state_dict().keys(), renamed.state_dict().keys())
        for name, original in legacy.state_dict().items():
            with self.subTest(parameter=name):
                self.assertTrue(
                    self.torch.equal(original, renamed.state_dict()[name])
                )

    def test_frahi_state_retains_legacy_architecture_names(self):
        renamed = self.make_model("frahi")
        state = renamed.state_dict()
        self.assertTrue(any(key.startswith("hipeak_decoder.") for key in state))
        self.assertFalse(any(key.startswith("frahi_decoder.") for key in state))
        legacy = self.make_model("hipeak")
        legacy.load_checkpoint(state)
        self.assertEqual(legacy.inference_prediction_source(), "full")

    def test_dense_checkpoint_remains_incompatible(self):
        renamed = self.make_model("frahi")
        dense = self.make_model("dense")
        with self.assertRaisesRegex(ValueError, "same architecture"):
            renamed.load_checkpoint(dense.state_dict())


if __name__ == "__main__":
    unittest.main()
