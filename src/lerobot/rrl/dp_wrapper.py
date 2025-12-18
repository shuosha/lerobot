"""Simple wrapper around a DiffusionPolicy and its pre/post processors.

This provides a tiny inference helper used for quick local testing. It:
 - loads a pretrained DiffusionPolicy from `model_id` (a local folder or HF repo)
 - loads/creates pre- and post-processors (optionally using dataset stats from `dataset_id`)
 - exposes `act(obs)` which runs preprocessor -> model.select_action -> postprocessor

The wrapper deliberately keeps the surface area small. It assumes the input `obs` is already in
LeRobot observation format (keys that the pipeline expects, typically starting with
`observation.*`). The returned value is whatever the postprocessor returns (usually a tensor or
numpy array representing the robot action).
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.processor import PolicyAction, PolicyProcessorPipeline


class DPWrapper:
    """Minimal DiffusionPolicy + processors wrapper.

    Args:
        model_id: pretrained model folder or HF repo id containing the policy (config + weights).
        dataset_id: optional dataset repo id (used to load dataset stats for processors).
        device: torch device string (e.g. 'cpu' or 'cuda'). If None, the policy config's device is used.
    """

    def __init__(self, model_id: str, dataset_id: str, device: Optional[str] = None):
        # Load dataset metadata to provide normalization stats to the processors
        dataset_metadata = LeRobotDatasetMetadata(dataset_id)

        # Load the pretrained diffusion policy
        self.policy: DiffusionPolicy = DiffusionPolicy.from_pretrained(model_id)

        # Prefer explicit device argument if provided
        self.device = "cuda" if device is None else device

        self.policy.to(self.device)
        self.policy.eval()

        # Create / load pre and post processors. We instruct processors to use our device.
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.policy.config,
            pretrained_path=model_id,
            dataset_stats=dataset_metadata.stats,
        )

    def act(self, obs: Dict[str, Any]) -> PolicyAction:
        """Run a single observation through preprocessor, model and postprocessor.

        The `obs` argument should be in the LeRobot observation format expected by the processors
        (for example, keys like `observation.state`, `observation.images`, ...). The method returns
        the postprocessed action (usually a tensor or numpy array).
        """
        # Preprocess observation (converts to batched tensors and moves to device)
        processed = self.preprocessor(obs)

        # Run model inference
        with torch.inference_mode():
            action_tensor = self.policy.select_action(processed)

        # Postprocess (unnormalize, move to CPU / numpy as configured by the pipeline)
        postprocessed = self.postprocessor(action_tensor)

        return postprocessed
    
    def reset(self):
        """Reset any internal stateful components (e.g. normalizers)."""
        self.policy.reset()


__all__ = ["DPWrapper"]
