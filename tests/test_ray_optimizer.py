import unittest

import torch
from omegaconf import OmegaConf

from trainers.pi3_trainer import Pi3Trainer
from utils.scheduler import build_scheduler


class DummyRayModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Linear(3, 4)
        self.ray_embed = torch.nn.Conv2d(2, 4, kernel_size=1)
        self.head = torch.nn.Linear(4, 3)


class DummyVisibilityMaskModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Linear(3, 4)
        self.visibility_mask_embed = torch.nn.Conv2d(2, 4, kernel_size=1)
        self.head = torch.nn.Linear(4, 3)


class DummyBaselineModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Linear(3, 4)
        self.head = torch.nn.Linear(4, 3)


def optimizer_config(*, include_ray=False, include_visibility_mask=False):
    values = {
        "type": "AdamW",
        "lr": 5e-6,
        "encoder_lr": 0.0,
        "weight_decay": 0.05,
        "betas": [0.9, 0.95],
        "filter_bias_and_bn": True,
    }
    if include_ray:
        values["ray_lr"] = 1e-5
    if include_visibility_mask:
        values["visibility_mask_lr"] = 5e-5
    return OmegaConf.create(values)


class RayOptimizerTest(unittest.TestCase):
    def build_optimizer(self, model, *, include_ray):
        trainer = object.__new__(Pi3Trainer)
        return trainer.build_optimizer(
            optimizer_config(include_ray=include_ray),
            model,
        )

    def test_ray_parameters_are_disjoint_and_use_their_own_lr(self):
        model = DummyRayModel()
        optimizer = self.build_optimizer(model, include_ray=True)

        parameter_ids = [
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        ]
        self.assertEqual(len(parameter_ids), len(set(parameter_ids)))
        self.assertEqual(len(parameter_ids), len(list(model.parameters())))

        ray_groups = [
            group for group in optimizer.param_groups
            if group.get("group_name") == "ray"
        ]
        self.assertTrue(ray_groups)
        self.assertTrue(all(group["lr"] == 1e-5 for group in ray_groups))
        self.assertTrue(all(group["max_lr"] == 1e-5 for group in ray_groups))

    def test_one_cycle_scheduler_preserves_ray_peak_lr(self):
        model = DummyRayModel()
        optimizer = self.build_optimizer(model, include_ray=True)
        scheduler = build_scheduler(
            OmegaConf.create(
                {
                    "type": "OneCycleLR",
                    "max_lr": 5e-6,
                    "total_steps": 10,
                    "pct_start": 0.3,
                    "anneal_strategy": "cos",
                    "div_factor": 10.0,
                    "final_div_factor": 10.0,
                }
            ),
            optimizer,
        )

        self.assertIsNotNone(scheduler)
        ray_groups = [
            group for group in optimizer.param_groups
            if group.get("group_name") == "ray"
        ]
        other_groups = [
            group for group in optimizer.param_groups
            if group.get("group_name") == "other"
        ]
        self.assertTrue(all(group["max_lr"] == 1e-5 for group in ray_groups))
        self.assertTrue(all(group["max_lr"] == 5e-6 for group in other_groups))

    def test_baseline_model_does_not_create_ray_groups(self):
        model = DummyBaselineModel()
        optimizer = self.build_optimizer(model, include_ray=False)

        self.assertFalse(any(
            group.get("group_name") == "ray"
            for group in optimizer.param_groups
        ))

    def test_visibility_mask_parameters_are_disjoint_and_use_their_own_lr(self):
        model = DummyVisibilityMaskModel()
        trainer = object.__new__(Pi3Trainer)
        optimizer = trainer.build_optimizer(
            optimizer_config(include_visibility_mask=True),
            model,
        )

        parameter_ids = [
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        ]
        self.assertEqual(len(parameter_ids), len(set(parameter_ids)))
        self.assertEqual(len(parameter_ids), len(list(model.parameters())))

        mask_groups = [
            group for group in optimizer.param_groups
            if group.get("group_name") == "visibility_mask"
        ]
        self.assertTrue(mask_groups)
        self.assertTrue(all(group["lr"] == 5e-5 for group in mask_groups))
        self.assertTrue(all(group["max_lr"] == 5e-5 for group in mask_groups))


if __name__ == "__main__":
    unittest.main()
