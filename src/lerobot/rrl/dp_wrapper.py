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

import os, json
from typing import Any, Dict, Optional

import torch
from torch import Tensor

from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.diffusion.processor_diffusion import make_action_normalizer
from lerobot.policies.factory import make_pre_post_processors
from lerobot.processor import PolicyAction, PolicyProcessorPipeline

def q_normalize(q):
    return q / (q.norm(dim=-1, keepdim=True).clamp_min(1e-12))


def q_conj(q):  # (w, x, y, z) -> (w, -x, -y, -z)
    w, x, y, z = q.unbind(-1)
    return torch.stack([w, -x, -y, -z], dim=-1)


def q_mul(q1, q2):
    # (w1,x1,y1,z1) * (w2,x2,y2,z2)
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return torch.stack([w, x, y, z], dim=-1)


def q_apply(q, v):
    # rotate vector v by quaternion q (wxyz)
    # v' = q * (0,v) * q_conj
    zeros = torch.zeros_like(v[..., :1])
    v_as_quat = torch.cat([zeros, v], dim=-1)
    return q_mul(q_mul(q, v_as_quat), q_conj(q))[..., 1:]

class DPWrapper:
    """Minimal DiffusionPolicy + processors wrapper.

    Args:
        model_id: pretrained model folder or HF repo id containing the policy (config + weights).
        dataset_id: optional dataset repo id (used to load dataset stats for processors).
        device: torch device string (e.g. 'cpu' or 'cuda'). If None, the policy config's device is used.
    """

    def __init__(self, model_id: str, device: Optional[str] = None):
        # Load training config to get policy horizon
        training_cfg = os.path.join(model_id, "train_config.json")
        with open(training_cfg, "r") as f:
            train_config = json.load(f)

        # Load the pretrained diffusion policy
        self.policy: DiffusionPolicy = DiffusionPolicy.from_pretrained(model_id)

        # Prefer explicit device argument if provided
        self.device = "cuda" if device is None else device

        self.policy.to(self.device)
        self.policy.eval()

        # Create / load pre and post processors. Normalization stats are loaded from the checkpoint.
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.policy.config,
            pretrained_path=model_id,
        )

        # Load the reference action normalizer from checkpoint (stats are already embedded)
        self.ref_action_processor = make_action_normalizer(
            config=self.policy.config,
            pretrained_path=model_id,
        )

        self.chunk_size = train_config["policy"]["horizon"]
        self.idx = 0

    def act(self, obs: Dict[str, Any], ref_action: Tensor | None = None) -> PolicyAction:
        """Run a single observation through preprocessor, model and postprocessor.

        The `obs` argument should be in the LeRobot observation format expected by the processors
        (for example, keys like `observation.state`, `observation.images`, ...). The method returns
        the postprocessed action (usually a tensor or numpy array).
        """
        # Preprocess observation (converts to batched tensors and moves to device)
        processed = self.preprocessor(obs)

        if ref_action is not None:
            ref_action = self.ref_action_processor(ref_action)

        # Run model inference
        with torch.inference_mode():
            action_tensor = self.policy.select_action(processed, ref_action=ref_action)

        # Postprocess (unnormalize, move to CPU / numpy as configured by the pipeline)
        postprocessed = self.postprocessor(action_tensor).to(self.device)

        # if self.idx % self.chunk_size == 0:
        #     self._p0 = obs["observation.state"][:,:3].clone()
        #     self._q0 = q_normalize(obs["observation.state"][:,3:7].clone())
        #     self._g0 = obs["observation.state"][:, 7:8].clone()

        # pt = self._p0 + q_apply(self._q0, postprocessed[:, :3])
        # qt = q_mul(self._q0, q_normalize(postprocessed[:, 3:7]))
        # gt = postprocessed[:, -1:]
        # postprocessed = torch.cat([pt, qt, gt], dim=-1)

        self.idx += 1

        return postprocessed
    
    def reset(self):
        """Reset any internal stateful components (e.g. normalizers)."""
        self.policy.reset()
        self.idx = 0


__all__ = ["DPWrapper"]
