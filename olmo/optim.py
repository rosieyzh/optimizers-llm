import logging
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass, replace
from math import cos, pi, sqrt
from typing import Any, Dict, List, Optional, Callable, Tuple, Union, Iterable
import warnings

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel
from torch.optim.optimizer import Optimizer as OptimizerBase
from timm import optim as timm_optim
import torch.optim
import math

from . import LayerNormBase
from .config import OptimizerType, SchedulerConfig, SchedulerType, TrainConfig
from .torch_util import get_default_device, is_distributed
from .sophia import SophiaG

import warnings

__all__ = [
    "Optimizer",
    "LionW",
    "AdamW",
    "SGDW",
    "AdafactorW",
    "AdafactorWLast",
    "AdalayerW",
    "AdalayerWLast"
    "SophiaG",
    "Scheduler",
    "CosWithWarmup",
    "FreezeCosWithWarmup",
    "LinearWithWarmup",
    "InvSqrtWithWarmup",
    "MaxScheduler",
    "ConstantScheduler",
    "BoltOnWarmupScheduler",
    "build_optimizer",
    "build_scheduler",
]


log = logging.getLogger(__name__)


class Optimizer(OptimizerBase):
    def _clean_param_name(self, name: str) -> str:
        return name.replace("_fsdp_wrapped_module.", "")

    @torch.no_grad()
    def clip_grads_and_collect_metrics(
        self, global_step: int, collect_param_metrics: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Clips gradients for every group that has the field `max_grad_norm`.
        At the same time collect metrics for each parameter and its gradient.
        """
        device = get_default_device()

        # NOTE (epwalsh): during distributed training we're making an assumption that the order of
        # the param groups and the params within each group are the same across all ranks.
        # This is justified since we initialize the parameter groups in every rank by iterating over
        # `module.parameters()` or `module.named_modules()` / `module.named_parameters()`, each of which
        # provides a consistent order.
        #  For each parameter (with a gradient) we'll collect:
        # - min, max, avg, norm of the param itself
        # - min, max, avg, norm of the param's gradient
        # - min, max, avg, norm of any additional per-parameter optimizer state metrics returned from
        #   `self.get_state_for_param()`.
        # Afterwards we'll reduce these all over all ranks.
        per_param_min_metrics: List[torch.Tensor] = []
        per_param_max_metrics: List[torch.Tensor] = []
        per_param_sum_metrics: List[torch.Tensor] = []
        per_param_norm_metrics: List[torch.Tensor] = []
        per_param_numel_metrics: List[torch.Tensor] = []

        per_param_min_metric_names: List[str] = []
        per_param_max_metric_names: List[str] = []
        per_param_avg_metric_names: List[str] = []
        per_param_norm_metric_names: List[str] = []

        # Collect metrics locally.
        for group in self.param_groups:
            if is_distributed():
                # TODO (epwalsh): handle non-sharded params. We don't have any right now but we would
                # with ReLoRa, for example.
                assert group.get("sharded", True) is True

            for name, p in zip(group["param_names"], group["params"]):
                name = self._clean_param_name(name)
                # Always need to collect the norm of gradients for clipping, even if we're not collecting
                # other metrics.
                tensors: List[Optional[torch.Tensor]] = [p.grad]
                prefixes: List[str] = [f"grad/{name}"]
                if collect_param_metrics:
                    state = self.get_state_for_param(p)
                    sorted_state_keys = sorted([k for k in state.keys()])
                    tensors.extend([p] + [state[key] for key in sorted_state_keys])
                    prefixes.extend([f"param/{name}"] + [f"{key}/{name}" for key in sorted_state_keys])
                assert len(tensors) == len(prefixes)

                # Get min, max, avg, and norm for all `tensors` associated with the parameter.
                for x, prefix in zip(tensors, prefixes):
                    # grad or state tensors could be none for params that have their shards completely on
                    # other ranks.
                    if x is not None and x.numel() > 0:
                        if collect_param_metrics:
                            x_abs = x.abs()
                            per_param_min_metrics.append(x_abs.min().unsqueeze(0).to(dtype=torch.float32))
                            per_param_max_metrics.append(x_abs.max().unsqueeze(0).to(dtype=torch.float32))
                            per_param_sum_metrics.append(x.sum().unsqueeze(0).to(dtype=torch.float32))
                            per_param_numel_metrics.append(
                                torch.tensor([x.numel()], device=device, dtype=torch.float32)
                            )
                        per_param_norm_metrics.append(
                            torch.linalg.vector_norm(x, 2.0, dtype=torch.float32).unsqueeze(0)
                        )
                    else:
                        if collect_param_metrics:
                            per_param_min_metrics.append(
                                torch.tensor([float("inf")], device=device, dtype=torch.float32)
                            )
                            per_param_max_metrics.append(torch.tensor([0.0], device=device, dtype=torch.float32))
                            per_param_sum_metrics.append(torch.tensor([0.0], device=device, dtype=torch.float32))
                            per_param_numel_metrics.append(torch.tensor([0.0], device=device, dtype=torch.float32))
                        per_param_norm_metrics.append(torch.tensor([0.0], device=device, dtype=torch.float32))
                    if collect_param_metrics:
                        per_param_min_metric_names.append(f"{prefix}.min")
                        per_param_max_metric_names.append(f"{prefix}.max")
                        per_param_avg_metric_names.append(f"{prefix}.avg")
                    per_param_norm_metric_names.append(f"{prefix}.norm")

        assert (
            len(per_param_min_metrics)
            == len(per_param_min_metric_names)
            == len(per_param_max_metrics)
            == len(per_param_max_metric_names)
            == len(per_param_sum_metrics)
            == len(per_param_numel_metrics)
            == len(per_param_avg_metric_names)
        )
        assert len(per_param_norm_metrics) == len(per_param_norm_metric_names)

        def is_grad_norm_metric(metric_name: str) -> bool:
            return metric_name.startswith("grad/") and metric_name.endswith(".norm")

        # Now reduce metrics over all ranks.
        total_grad_norm: torch.Tensor
        per_param_avg_metrics: List[torch.Tensor] = []
        if is_distributed():  # TODO (epwalsh): skip for non-sharded params
            # Reduce metrics across all ranks. Note that we can use a `reduce` for most cases
            # instead of an `all_reduce`, but we need `all_reduce` for norms so that all ranks
            # get the right value for gradient norms so they can clip correctly.
            # Reduce mins.
            if per_param_min_metrics:
                all_mins = torch.cat(per_param_min_metrics).to(device)
                dist.reduce(all_mins, 0, op=dist.ReduceOp.MIN)
                per_param_min_metrics = all_mins.split(1)
            # Reduce maxs.
            if per_param_max_metrics:
                all_maxs = torch.cat(per_param_max_metrics).to(device)
                dist.reduce(all_maxs, 0, op=dist.ReduceOp.MAX)
                per_param_max_metrics = all_maxs.split(1)
            # Reduce sums or just norms.
            all_norms = torch.cat(per_param_norm_metrics).to(device) ** 2.0
            if per_param_sum_metrics and per_param_numel_metrics:
                all_sums = torch.cat(per_param_sum_metrics).to(device)
                all_numels = torch.cat(per_param_numel_metrics).to(device)
                all_sums_norms_numels = torch.cat(
                    [all_sums.unsqueeze(0), all_norms.unsqueeze(0), all_numels.unsqueeze(0)], dim=0
                )
                dist.all_reduce(all_sums_norms_numels, op=dist.ReduceOp.SUM)
                all_sums, all_norms, all_numels = all_sums_norms_numels.split(1)
                # Get averages.
                # NOTE: could get infs for non-rank0 processes but that's okay.
                per_param_avg_metrics = (all_sums / all_numels).squeeze(0).split(1)
            else:
                dist.all_reduce(all_norms, op=dist.ReduceOp.SUM)
            grad_norm_metric_mask = torch.tensor(
                [float(is_grad_norm_metric(n)) for n in per_param_norm_metric_names], device=all_norms.device
            )
            total_grad_norm = (all_norms * grad_norm_metric_mask).sum() ** 0.5
            per_param_norm_metrics = (all_norms ** (0.5)).squeeze(0).split(1)
        else:
            total_grad_norm = (
                torch.cat(
                    [
                        m
                        for m, n in zip(per_param_norm_metrics, per_param_norm_metric_names)
                        if is_grad_norm_metric(n)
                    ]
                )
                ** 2.0
            ).sum() ** 0.5
            per_param_avg_metrics = [x / n for x, n in zip(per_param_sum_metrics, per_param_numel_metrics)]

        assert len(per_param_avg_metrics) == len(per_param_avg_metric_names)

        # Collect all metrics into a single dict.
        all_metrics: Dict[str, torch.Tensor] = {}
        for metric_name, metric in zip(per_param_min_metric_names, per_param_min_metrics):
            all_metrics[metric_name] = metric.squeeze(0)
        for metric_name, metric in zip(per_param_max_metric_names, per_param_max_metrics):
            all_metrics[metric_name] = metric.squeeze(0)
        for metric_name, metric in zip(per_param_avg_metric_names, per_param_avg_metrics):
            all_metrics[metric_name] = metric.squeeze(0)
        for metric_name, metric in zip(per_param_norm_metric_names, per_param_norm_metrics):
            all_metrics[metric_name] = metric.squeeze(0)
        all_metrics["total_grad_norm"] = total_grad_norm

        # Clip gradients.
        num_grads_clipped = 0
        num_eligible_grads = 0
        for group in self.param_groups:
            if (max_norm_ratio := group.get("max_grad_norm_ratio")) is not None:
                num_clipped = self._do_adaptive_clipping(
                    group, max_norm_ratio, global_step, all_metrics, collect_param_metrics=collect_param_metrics
                )
            elif (max_norm := group.get("max_grad_norm")) is not None:
                num_clipped = self._do_global_fixed_clipping(
                    group, max_norm, all_metrics, collect_param_metrics=collect_param_metrics
                )
            else:
                # No clipping needed.
                continue
            num_eligible_grads += len(group["params"])
            if num_clipped is not None:
                num_grads_clipped += num_clipped

        if collect_param_metrics:
            if num_eligible_grads > 0:
                clipping_rate = torch.tensor(num_grads_clipped / num_eligible_grads, device="cpu")
            else:
                clipping_rate = torch.tensor(0.0, device="cpu")
            all_metrics["clipping_rate"] = clipping_rate
            return all_metrics
        else:
            return {}

    @torch.no_grad()
    def _do_adaptive_clipping(
        self,
        group: Dict[str, Any],
        max_norm_ratio: float,
        global_step: int,
        all_metrics: Dict[str, torch.Tensor],
        collect_param_metrics: bool = True,
    ) -> Optional[int]:
        """
        Do adaptive gradient clipping on a param group.

        If ``collect_param_metrics`` is ``True`` this will return the total number of gradients clipped.
        """
        device = get_default_device()
        num_grads_clipped = 0
        # We'll use the bigger of beta1 and beta2 to update the exponential average of the norm of
        # the gradient (a scalar), not to be confused with the exponential average of the gradient.
        # TODO (epwalsh): handle optimizers that don't have betas.
        beta1, beta2 = group["betas"]
        beta = max(beta1, beta2)
        for name, p in zip(group["param_names"], group["params"]):
            name = self._clean_param_name(name)
            grad_norm = all_metrics.get(f"grad/{name}.norm")
            if grad_norm is None:
                continue

            # Get or initialize the exponential average of grad norm.
            # TODO: The way we have it right now, every rank tracks the `grad_norm_exp_avg` of every parameter,
            # even parameters for which the corresponding local shard is empty. This has the potential to
            # cause some issues with the optimizer, as we ran into with https://github.com/allenai/LLM/pull/372.
            # So we should consider changing how we do this at some point so that we don't add any state
            # to parameters for which the local shard is empty. That would probably add extra distributed
            # communication, at least on steps where we have to log (i.e. when `collect_param_metrics=True`).
            state = self.state[p]
            grad_norm_exp_avg = state.get("grad_norm_exp_avg")
            if grad_norm_exp_avg is None:
                grad_norm_exp_avg = grad_norm.clone().to(device)
                # We don't want to add anything to `state` until `state` has been initialized, otherwise
                # this will crash some optimizers which rely on checking `len(state)`. The downside here
                # is that we won't start tracking `grad_norm_exp_avg` until the 2nd training step.
                if global_step > 1:
                    state["grad_norm_exp_avg"] = grad_norm_exp_avg

            max_allowed_norm = max_norm_ratio * grad_norm_exp_avg
            clip_coef = max_allowed_norm / (grad_norm + 1e-6)

            # Clip the gradients and update the exponential average.
            # Note that multiplying by the clamped coefficient is meaningless when it is
            # equal to 1, but it avoids the host-device sync that would result from `if clip_coef_clamped < 1`.
            clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
            if p.grad is not None:
                # p.grad could be none for some ranks when using FSDP.
                p.grad.detach().mul_(clip_coef_clamped.to(p.grad.device, p.grad.dtype))

            # Update the exponential average of the norm of the gradient with the clipped norm of the gradient.
            grad_norm_exp_avg.lerp_((grad_norm * clip_coef_clamped).to(grad_norm_exp_avg.device), 1 - beta)
            # Alternative: update with the *unclipped* norm of the gradient.
            #  grad_norm_exp_avg.lerp_(grad_norm.to(grad_norm_exp_avg.device), 1 - beta)

            if collect_param_metrics:
                # Can't avoid host-device sync here.
                if clip_coef_clamped < 1.0:
                    num_grads_clipped += 1
                all_metrics[f"grad_norm_exp_avg/{name}"] = grad_norm_exp_avg
        return num_grads_clipped if collect_param_metrics else None

    @torch.no_grad()
    def _do_global_fixed_clipping(
        self,
        group: Dict[str, Any],
        max_norm: float,
        all_metrics: Dict[str, torch.Tensor],
        collect_param_metrics: bool = True,
    ) -> Optional[int]:
        """
        Do global fixed gradient clipping on a param group.

        If ``collect_param_metrics`` is ``True`` this will return the total number of gradients clipped.
        """
        device = get_default_device()
        total_grad_norm = all_metrics["total_grad_norm"]
        clip_coef = max_norm / (total_grad_norm.to(device) + 1e-6)
        clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
        num_grads_clipped: Optional[int] = None
        if collect_param_metrics:
            # Can't avoid host-device sync here.
            if clip_coef_clamped < 1.0:
                num_grads_clipped = len(group["params"])
        for p in group["params"]:
            # Clip the gradients.
            # Note that multiplying by the clamped coefficient is meaningless when it is
            # equal to 1, but it avoids the host-device sync that would result from `if clip_coef_clamped < 1`.
            if p.grad is not None:
                # p.grad could be none for some ranks when using FSDP.
                p.grad.detach().mul_(clip_coef_clamped.to(p.grad.device, p.grad.dtype))
        return num_grads_clipped

    def get_post_step_metrics(self, module: nn.Module) -> Dict[str, torch.Tensor]:
        del module
        return {}

    def get_state_for_param(self, param: nn.Parameter) -> Dict[str, Optional[torch.Tensor]]:
        del param
        return {}
    

class LionW(Optimizer):
    """
    Adapted from https://github.com/google/automl/blob/master/lion/lion_pytorch.py
    """

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
    ):
        assert lr > 0.0
        assert all([0.0 <= beta <= 1.0 for beta in betas])
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)
        for group in self.param_groups:
            group["initial_lr"] = group["lr"]
        self._update_total_dot_prod: Optional[torch.Tensor] = None
        self._update_total_norm: Optional[torch.Tensor] = None
        self._signed_update_total_norm: Optional[torch.Tensor] = None

    def get_post_step_metrics(self, module: nn.Module) -> Dict[str, torch.Tensor]:
        update_total_dot_prod = self._update_total_dot_prod
        update_total_norm = self._update_total_norm
        signed_update_total_norm = self._signed_update_total_norm
        if update_total_dot_prod is None or update_total_norm is None or signed_update_total_norm is None:
            return {}

        if is_distributed() and isinstance(module, FullyShardedDataParallel):
            # Reduce total dot prod and norms across all ranks.
            update_total_norm = update_total_norm**2.0
            signed_update_total_norm = signed_update_total_norm**2.0
            # Reduce all together to avoid multiple communication calls.
            all_together = torch.stack([update_total_dot_prod, update_total_norm, signed_update_total_norm])
            # Only need the final result on rank0, since that's where we log from.
            dist.reduce(all_together, 0)
            update_total_dot_prod, update_total_norm, signed_update_total_norm = all_together
            update_total_norm = update_total_norm**0.5
            signed_update_total_norm = signed_update_total_norm**0.5

        update_cos_sim = update_total_dot_prod / torch.max(
            update_total_norm * signed_update_total_norm, torch.tensor(1e-8, device=get_default_device())
        )
        return {"update_cos_sim": update_cos_sim}

    @torch.no_grad()
    def step(self, closure=None) -> None:
        if closure is not None:
            with torch.enable_grad():
                closure()

        update_total_dot_prod = torch.tensor(0.0, dtype=torch.float32)
        update_norms = []
        signed_update_norms = []

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                # Perform step weight decay
                p.data.mul_(1 - group["lr"] * group["weight_decay"])

                grad = p.grad
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p)

                exp_avg = state["exp_avg"]
                beta1, beta2 = group["betas"]

                # Weight update
                update = exp_avg * beta1 + grad * (1 - beta1)
                signed_update = torch.sign(update)
                p.add_(signed_update, alpha=-group["lr"])

                # Decay the momentum running average coefficient
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)

                # Track dot product and norms of update vs signed update in order to calculate
                # their cosine similarity.
                update_total_dot_prod = update_total_dot_prod.to(update.device)
                update_total_dot_prod += torch.tensordot(update, signed_update, dims=len(update.shape))
                update_norms.append(torch.linalg.vector_norm(update, 2.0, dtype=torch.float32))
                signed_update_norms.append(torch.linalg.vector_norm(signed_update, 2.0, dtype=torch.float32))

        # Compute cosine similarity between update and signed update.
        self._update_total_dot_prod = update_total_dot_prod.to(get_default_device())
        self._update_total_norm = torch.linalg.vector_norm(
            torch.stack(update_norms),
            2.0,
            dtype=torch.float32,
        ).to(get_default_device())
        self._signed_update_total_norm = torch.linalg.vector_norm(
            torch.stack(signed_update_norms),
            2.0,
            dtype=torch.float32,
        ).to(get_default_device())


class AdamW(torch.optim.AdamW, Optimizer):
    def get_state_for_param(self, param: nn.Parameter) -> Dict[str, Optional[torch.Tensor]]:
        return {key: self.state[param].get(key) for key in ("exp_avg", "exp_avg_sq")}  # type: ignore
    

class SophiaG(SophiaG, Optimizer):
    def get_state_for_param(self, param: nn.Parameter) -> Dict[str, Optional[torch.Tensor]]:
        return {key: self.state[param].get(key) for key in ("exp_avg", "hessian")}  # type: ignore

    @torch.no_grad()
    def clip_grads_and_collect_metrics(
        self, global_step: int, collect_param_metrics: bool = True
    ) -> Dict[str, torch.Tensor]:
        all_metrics = super().clip_grads_and_collect_metrics(global_step, collect_param_metrics)
        clip_ratio_num = torch.tensor(0.0, device="cpu")
        clip_ratio_den = torch.tensor(0.0, device="cpu")
        for group in self.param_groups:
            for p in group["params"]:
                state = self.get_state_for_param(p)
                exp_avg = state["exp_avg"]
                hess = state["hessian"]
                if exp_avg is None or hess is None:
                    continue
                bs = 256 * 512  # TODO: hardcoding this hack for now
                ratio = exp_avg.abs() / (group["rho"] * bs * hess + 1e-15)
                clip_ratio_num += (ratio > 1.0).sum().detach().cpu()
                clip_ratio_den += ratio.numel()
        all_metrics["clip_ratio"] = clip_ratio_num / (clip_ratio_den + 1e-15)
        return all_metrics

    # TODO: add logging here


class SGDW(timm_optim.sgdw.SGDW, Optimizer):
    def get_state_for_param(self, param: nn.Parameter) -> Dict[str, Optional[torch.Tensor]]:
        return {key: self.state[param].get(key) for key in ("momentum_buffer",)}  # type: ignore


from .config import OptimizerType, SchedulerType, TrainConfig


class AdafactorW(Optimizer):
    def __init__(
        self,
        params,
        lr: Union[float, torch.Tensor] = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        neuron_only: bool = False,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            neuron_only=neuron_only,
        )
        self._neuron_only = neuron_only
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        state_values = list(self.state.values())
        step_is_tensor = (len(state_values) != 0) and torch.is_tensor(state_values[0]["step"])
        if not step_is_tensor:
            for s in state_values:
                s["step"] = torch.tensor(float(s["step"]), dtype=torch.float32)

    def _init_group(
        self,
        group,
        params_with_grad,
        grads,
        exp_avgs,
        exp_avg_sqs,
        state_steps,
    ):
        for p in group["params"]:
            if p.grad is None:
                continue
            params_with_grad.append(p)
            grads.append(p.grad)

            state = self.state[p]

            # State initialization
            if len(state) == 0:
                # note(crcrpar): Deliberately host `step` on CPU if both capturable and fused are off.
                # This is because kernel launches are costly on CUDA and XLA.
                state["step"] = torch.tensor(0.0, dtype=torch.float32)
                # Exponential moving average of gradient values
                state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                # Exponential moving average of squared gradient values marginalized on all but the dim_i^th dimension

                state["exp_avg_sq"] = {}
                for dim_i in range(len(p.shape)):
                    state["exp_avg_sq"][dim_i] = torch.zeros((p.shape[dim_i]), device=p.device, dtype=p.dtype)

            exp_avgs.append(state["exp_avg"])
            exp_avg_sqs.append(state["exp_avg_sq"])

            state_steps.append(state["step"])
        return

    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        self._cuda_graph_capture_health_check()

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            state_steps = []
            beta1, beta2 = group["betas"]
            param_names = group["param_names"]

            self._init_group(
                group,
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                state_steps,
            )

            adafactorw(
                param_names,
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                state_steps,
                beta1=beta1,
                beta2=beta2,
                lr=group["lr"],
                weight_decay=group["weight_decay"],
                eps=group["eps"],
                grad_scale=getattr(self, "grad_scale", None),
                found_inf=getattr(self, "found_inf", None),
                neuron_only=group["neuron_only"],
            )

        return loss

    def get_state_for_param(self, param: nn.Parameter) -> Dict[str, Optional[torch.Tensor]]:
        state = {}
        state["exp_avg"] = self.state[param]["exp_avg"]
        state["exp_avg_sq"] = rank1_approximation(self.state[param]["exp_avg_sq"], self._neuron_only)

        return state  # type: ignore


def rank1_approximation(exp_avg_sq: torch.Tensor, neuron_only=False):
    exp_avg_sq_tmp = exp_avg_sq[0].clone()

    avg = torch.mean(exp_avg_sq[0])

    # Rank 1 approximation to the tensor
    for dim_i in range(1, len(exp_avg_sq)):
        if neuron_only and dim_i == len(exp_avg_sq) - 1:
            cur_shape = exp_avg_sq_tmp.shape
            a_flat = exp_avg_sq_tmp.flatten()
            b_flat = exp_avg_sq[dim_i].mean(dim=0, keepdim=True).expand(*exp_avg_sq[dim_i].shape)
            exp_avg_sq_tmp = torch.outer(a_flat, b_flat)
            # Reshape to dim_i+1 dimensional tensor
            exp_avg_sq_tmp = exp_avg_sq_tmp.view(*cur_shape, exp_avg_sq[dim_i].shape[0])
        else:
            cur_shape = exp_avg_sq_tmp.shape
            a_flat = exp_avg_sq_tmp.flatten()
            b_flat = exp_avg_sq[dim_i]
            exp_avg_sq_tmp = torch.outer(a_flat, b_flat)
            # Reshape to dim_i+1 dimensional tensor
            exp_avg_sq_tmp = exp_avg_sq_tmp.view(*cur_shape, exp_avg_sq[dim_i].shape[0])

    # Check
    if len(exp_avg_sq) > 1:
        exp_avg_sq_tmp /= avg ** (len(exp_avg_sq) - 1)
    return exp_avg_sq_tmp


def adafactorw(
    param_names: List[str],
    params: List[torch.Tensor],
    grads: List[torch.Tensor],
    exp_avgs: List[torch.Tensor],
    exp_avg_sqs: List[torch.Tensor],
    state_steps: List[torch.Tensor],
    # kwonly args with defaults are not supported by functions compiled with torchscript issue #70627
    # setting this as kwarg for now as functional API is compiled by torch/distributed/optim
    grad_scale: Optional[torch.Tensor] = None,
    found_inf: Optional[torch.Tensor] = None,
    *,
    beta1: float,
    beta2: float,
    lr: Union[float, torch.Tensor],
    weight_decay: float,
    eps: float,
    neuron_only: bool,
):
    r"""Functional API that performs AdamW algorithm computation.

    See :class:`~torch.optim.AdamW` for details.
    """
    if not torch._utils.is_compiling() and not all(isinstance(t, torch.Tensor) for t in state_steps):
        raise RuntimeError("API has changed, `state_steps` argument must contain a list of singleton tensors")

    func = _single_tensor_adafactorw

    func(
        param_names,
        params,
        grads,
        exp_avgs,
        exp_avg_sqs,
        state_steps,
        beta1=beta1,
        beta2=beta2,
        lr=lr,
        weight_decay=weight_decay,
        eps=eps,
        grad_scale=grad_scale,
        found_inf=found_inf,
        neuron_only=neuron_only,
    )


def _single_tensor_adafactorw(
    param_names: List[str],
    params: List[torch.Tensor],
    grads: List[torch.Tensor],
    exp_avgs: List[torch.Tensor],
    exp_avg_sqs: List[torch.Tensor],
    state_steps: List[torch.Tensor],
    grad_scale: Optional[torch.Tensor],
    found_inf: Optional[torch.Tensor],
    *,
    beta1: float,
    beta2: float,
    lr: Union[torch.Tensor, float],
    weight_decay: float,
    eps: float,
    neuron_only: bool,
):

    assert grad_scale is None and found_inf is None

    if torch.jit.is_scripting():
        # this assert is due to JIT being dumb and not realizing that the ops below
        # have overloads to handle both float and Tensor lrs, so we just assert it's
        # a float since most people using JIT are using floats
        assert isinstance(lr, float)

    for i, (n, param) in enumerate(zip(param_names, params)):

        grad = grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step_t = state_steps[i]

        # update step
        step_t += 1

        # Perform stepweight decay
        param.data.mul_(1 - lr * weight_decay)
        print("Multiplier: ", lr * weight_decay)

        step = step_t.item()

        # Decay the first and second moment running average coefficient
        exp_avg.lerp_(grad, 1 - beta1)

        for dim_i in range(len(param.shape)):
            contract_dim = [i for i in range(len(param.shape)) if i != dim_i]
            exp_avg_sq[dim_i].lerp_(grad.square().mean(dim=contract_dim), 1 - beta2)

        # exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

        bias_correction1 = 1 - beta1**step
        bias_correction2 = 1 - beta2**step

        step_size = lr / bias_correction1

        bias_correction2_sqrt = bias_correction2**0.5

        exp_avg_sq_tmp = rank1_approximation(exp_avg_sq, neuron_only)

        denom = (exp_avg_sq_tmp.sqrt() / bias_correction2_sqrt).add_(eps)
        param.data.addcdiv_(exp_avg, denom, value=-step_size)


class AdalayerW(Optimizer):
    def __init__(
        self,
        params,
        lr: Union[float, torch.Tensor] = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        att_correction: bool = False,
        lastlayer_correction: bool = False,
        firstlayer_correction: bool = False,
        update_last: bool = True,
        update_norm: bool = False,
        t_freeze: Optional[int] = None,
        no_norm_training: bool = False,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            att_correction=att_correction,
            lastlayer_correction=lastlayer_correction,
            firstlayer_correction=firstlayer_correction,
            update_last=update_last,
            update_norm=update_norm,
            t_freeze=t_freeze,
            no_norm_training=no_norm_training
        )
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        state_values = list(self.state.values())
        step_is_tensor = (len(state_values) != 0) and torch.is_tensor(state_values[0]["step"])
        if not step_is_tensor:
            for s in state_values:
                s["step"] = torch.tensor(float(s["step"]), dtype=torch.float32)

    def _init_group(
        self,
        group,
        params_with_grad,
        grads,
        exp_avgs,
        exp_avg_sqs,
        state_steps,
        att_correction,
        lastlayer_correction,
        firstlayer_correction,
    ):
        for p, n in zip(group["params"], group["param_names"]):
            if p.grad is None:
                continue
            params_with_grad.append(p)
            grads.append(p.grad)

            state = self.state[p]

            # State initialization
            if len(state) == 0:
                # note(crcrpar): Deliberately host `step` on CPU if both capturable and fused are off.
                # This is because kernel launches are costly on CUDA and XLA.
                state["step"] = torch.tensor(0.0, dtype=torch.float32)
                # Exponential moving average of gradient values
                state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                # Exponential moving average of squared gradient values marginalized on all but the dim_i^th dimension
                if att_correction and "att_proj" in n:
                    state["exp_avg_sq"] = []
                    for j in range(3):
                        state["exp_avg_sq"].append(torch.tensor(0.0, dtype=torch.float32, device=p.device))
                elif lastlayer_correction and "ff_out" in n and p.shape[0] > 10000:
                    log.info("Used heuristic to detect last layer")
                    state["exp_avg_sq"] = torch.zeros_like(p.mean(dim=1))
                elif firstlayer_correction and "wte" in n and p.shape[0] > 10000:
                    log.info("Used heuristic to detect first layer")
                    state["exp_avg_sq"] = torch.zeros_like(p.mean(dim=1))
                else:
                    state["exp_avg_sq"] = torch.tensor(0.0, dtype=torch.float32, device=p.device)

            exp_avgs.append(state["exp_avg"])
            exp_avg_sqs.append(state["exp_avg_sq"])

            state_steps.append(state["step"])
        return

    def step(self, closure=None):
        """Perform a single optimization step.
        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        self._cuda_graph_capture_health_check()

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            state_steps = []
            beta1, beta2 = group["betas"]
            param_names = group["param_names"]

            self._init_group(
                group,
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                state_steps,
                group["att_correction"],
                group["lastlayer_correction"],
                group["firstlayer_correction"],
            )

            Adalayerw(
                param_names,
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                state_steps,
                beta1=beta1,
                beta2=beta2,
                lr=group["lr"],
                weight_decay=group["weight_decay"],
                eps=group["eps"],
                grad_scale=getattr(self, "grad_scale", None),
                found_inf=getattr(self, "found_inf", None),
                att_correction=group["att_correction"],
                lastlayer_correction=group["lastlayer_correction"],
                firstlayer_correction=group["firstlayer_correction"],
                update_last=group["update_last"],
                update_norm=group["update_norm"],
                t_freeze=group["t_freeze"],
                no_norm_training=group["no_norm_training"]
            )

        return loss

    def get_state_for_param(self, param: nn.Parameter) -> Dict[str, Optional[torch.Tensor]]:
        state = {}
        state["exp_avg"] = self.state[param]["exp_avg"]
        # check if state["exp_avg_sq"] is a list
        if isinstance(self.state[param]["exp_avg_sq"], list):
            state["exp_avg_sq"] = torch.zeros_like(state["exp_avg"])
            shape0 = state["exp_avg"].shape[0]
            for j in range(3):
                state["exp_avg_sq"][j * (shape0 // 3) : (j + 1) * (shape0 // 3)] += self.state[param][
                    "exp_avg_sq"
                ][j]
        else:
            state["exp_avg_sq"] = torch.zeros_like(self.state[param]["exp_avg_sq"])
            state["exp_avg_sq"] += self.state[param]["exp_avg_sq"]

        return state  # type: ignore


def Adalayerw(
    param_names: List[str],
    params: List[torch.Tensor],
    grads: List[torch.Tensor],
    exp_avgs: List[torch.Tensor],
    exp_avg_sqs: List[torch.Tensor],
    state_steps: List[torch.Tensor],
    # kwonly args with defaults are not supported by functions compiled with torchscript issue #70627
    # setting this as kwarg for now as functional API is compiled by torch/distributed/optim
    grad_scale: Optional[torch.Tensor] = None,
    found_inf: Optional[torch.Tensor] = None,
    *,
    beta1: float,
    beta2: float,
    lr: Union[float, torch.Tensor],
    weight_decay: float,
    eps: float,
    att_correction: bool = False,
    lastlayer_correction: bool = False,
    firstlayer_correction: bool = False,
    update_last: bool = True,
    update_norm: bool = False,
    t_freeze: Optional[int] = None,
    no_norm_training: bool = False,
):
    r"""Functional API that performs AdamW algorithm computation.
    See :class:`~torch.optim.AdamW` for details.
    """
    if not torch._utils.is_compiling() and not all(isinstance(t, torch.Tensor) for t in state_steps):
        raise RuntimeError("API has changed, `state_steps` argument must contain a list of singleton tensors")

    func = _single_tensor_Adalayerw

    func(
        param_names,
        params,
        grads,
        exp_avgs,
        exp_avg_sqs,
        state_steps,
        beta1=beta1,
        beta2=beta2,
        lr=lr,
        weight_decay=weight_decay,
        eps=eps,
        grad_scale=grad_scale,
        found_inf=found_inf,
        att_correction=att_correction,
        lastlayer_correction=lastlayer_correction,
        firstlayer_correction=firstlayer_correction,
        update_last=update_last,
        update_norm=update_norm,
        t_freeze=t_freeze,
        no_norm_training=no_norm_training
    )


def _single_tensor_Adalayerw(
    param_names: List[str],
    params: List[torch.Tensor],
    grads: List[torch.Tensor],
    exp_avgs: List[torch.Tensor],
    exp_avg_sqs: List[torch.Tensor],
    state_steps: List[torch.Tensor],
    grad_scale: Optional[torch.Tensor],
    found_inf: Optional[torch.Tensor],
    *,
    beta1: float,
    beta2: float,
    lr: Union[torch.Tensor, float],
    weight_decay: float,
    eps: float,
    att_correction: bool = False,
    lastlayer_correction: bool = False,
    firstlayer_correction: bool = False,
    update_last: bool = True,
    update_norm: bool = False,
    t_freeze: Optional[int] = None,
    no_norm_training: bool = False
):

    assert grad_scale is None and found_inf is None

    if torch.jit.is_scripting():
        # this assert is due to JIT being dumb and not realizing that the ops below
        # have overloads to handle both float and Tensor lrs, so we just assert it's
        # a float since most people using JIT are using floats
        assert isinstance(lr, float)

    for i, (n, param) in enumerate(zip(param_names, params)):
        if no_norm_training and len(param.shape) == 1 and ('norm' in n or 'ln_f' in n):
            continue

        grad = grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step_t = state_steps[i]

        # update step
        step_t += 1

        # Perform stepweight decay
        param.data.mul_(1 - lr * weight_decay)

        step = step_t.item()

        # Decay the first and second moment running average coefficient
        exp_avg.lerp_(grad, 1 - beta1)
        
        if not t_freeze:
            if att_correction and "att_proj" in n:
                shape0 = grad.shape[0]
                # print(shape0)
                exp_avg_sq_tmp = torch.ones_like(grad)
                for j in range(3):
                    exp_avg_sq[j].lerp_(grad[j * shape0 // 3 : (j + 1) * shape0 // 3].square().mean(), 1 - beta2)
                    exp_avg_sq_tmp[j * shape0 // 3 : (j + 1) * shape0 // 3] *= exp_avg_sq[j]
            elif lastlayer_correction and "ff_out" in n and grad.shape[0] > 10000:
                # print(exp_avg_sq.shape)
                exp_avg_sq_tmp = torch.ones_like(grad)
                exp_avg_sq.lerp_(grad.square().mean(dim=1), 1 - beta2)
                exp_avg_sq_tmp *= exp_avg_sq[:, None]

            elif firstlayer_correction and "wte" in n:
                exp_avg_sq_tmp = torch.ones_like(grad)
                exp_avg_sq.lerp_(grad.square().mean(dim=1), 1 - beta2)
                exp_avg_sq_tmp *= exp_avg_sq[:, None]
            else:
                exp_avg_sq.lerp_(grad.square().mean(), 1 - beta2)
                exp_avg_sq_tmp = exp_avg_sq
        elif step <= t_freeze:
            tmp_beta2 = 1 - 1/ (step + 1)
            if att_correction and "att_proj" in n:
                shape0 = grad.shape[0]
                # print(shape0)
                exp_avg_sq_tmp = torch.ones_like(grad)
                for j in range(3):
                    exp_avg_sq[j].lerp_(grad[j * shape0 // 3 : (j + 1) * shape0 // 3].square().mean(), 1 - tmp_beta2)
                    exp_avg_sq_tmp[j * shape0 // 3 : (j + 1) * shape0 // 3] *= exp_avg_sq[j]
            elif lastlayer_correction and "ff_out" in n and grad.shape[0] > 10000:
                # print(exp_avg_sq.shape)
                exp_avg_sq_tmp = torch.ones_like(grad)
                exp_avg_sq.lerp_(grad.square().mean(dim=1), 1 - tmp_beta2)
                exp_avg_sq_tmp *= exp_avg_sq[:, None]

            elif firstlayer_correction and "wte" in n:
                exp_avg_sq_tmp = torch.ones_like(grad)
                exp_avg_sq.lerp_(grad.square().mean(dim=1), 1 - tmp_beta2)
                exp_avg_sq_tmp *= exp_avg_sq[:, None]
            else:
                exp_avg_sq.lerp_(grad.square().mean(), 1 - tmp_beta2)
                exp_avg_sq_tmp = exp_avg_sq
        else: # step > t_freeze
            if att_correction and "att_proj" in n:
                shape0 = grad.shape[0]
                # print(shape0)
                exp_avg_sq_tmp = torch.ones_like(grad)
                for j in range(3):
                    exp_avg_sq_tmp[j * shape0 // 3 : (j + 1) * shape0 // 3] *= exp_avg_sq[j]
            elif lastlayer_correction and "ff_out" in n and grad.shape[0] > 10000:
                # print(exp_avg_sq.shape)
                exp_avg_sq_tmp = torch.ones_like(grad)
                if update_last:
                    exp_avg_sq.lerp_(grad.square().mean(dim=1), 1 - beta2)
                exp_avg_sq_tmp *= exp_avg_sq[:, None]
            elif update_norm and ('norm' in n or 'ln_f' in n):
                exp_avg_sq.lerp_(grad.square().mean(), 1 - beta2)
                exp_avg_sq_tmp = exp_avg_sq
            elif firstlayer_correction and "wte" in n:
                exp_avg_sq_tmp = torch.ones_like(grad)
                exp_avg_sq.lerp_(grad.square().mean(dim=1), 1 - beta2)
                exp_avg_sq_tmp *= exp_avg_sq[:, None]
            else:
                exp_avg_sq_tmp = torch.ones_like(grad)
                exp_avg_sq_tmp *= exp_avg_sq # after t_freeze, don't update exp_avg_sq

        # exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

        bias_correction1 = 1 - beta1**step

        if t_freeze:
            bias_correction2 = 1
        else:
            bias_correction2 = 1 - beta2**step

        step_size = lr / bias_correction1

        bias_correction2_sqrt = bias_correction2**0.5
        denom = (exp_avg_sq_tmp.sqrt() / bias_correction2_sqrt).add_(eps)
        param.data.addcdiv_(exp_avg, denom, value=-step_size)

class AdafactorWLast(Optimizer):
    def __init__(
        self,
        params,
        lr: Union[float, torch.Tensor] = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        neuron_only: bool = False,
        update_last: bool = True,
        update_norm: bool = True,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            neuron_only=neuron_only,
            update_last=update_last,
            update_norm=update_norm,
        )
        self._neuron_only = neuron_only
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        state_values = list(self.state.values())
        step_is_tensor = (len(state_values) != 0) and torch.is_tensor(state_values[0]["step"])
        if not step_is_tensor:
            for s in state_values:
                s["step"] = torch.tensor(float(s["step"]), dtype=torch.float32)

    def _init_group(
        self,
        group,
        params_with_grad,
        grads,
        exp_avgs,
        exp_avg_sqs,
        state_steps,
        update_last,
        update_norm,
    ):
        for p, n in zip(group["params"], group["param_names"]):
            if p.grad is None:
                continue
            params_with_grad.append(p)
            grads.append(p.grad)

            state = self.state[p]

            # State initialization
            if len(state) == 0:
                # note(crcrpar): Deliberately host `step` on CPU if both capturable and fused are off.
                # This is because kernel launches are costly on CUDA and XLA.
                state["step"] = torch.tensor(0.0, dtype=torch.float32)
                # Exponential moving average of gradient values
                state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                # Exponential moving average of squared gradient values marginalized on all but the dim_i^th dimension

                state["exp_avg_sq"] = {}
                if update_last and "ff_out_last" in n:
                    for dim_i in range(len(p.shape)):
                        state["exp_avg_sq"][dim_i] = torch.zeros((p.shape[dim_i]), device=p.device, dtype=p.dtype)
                elif update_norm and ("norm" in n or "ln_f" in n):
                    for dim_i in range(len(p.shape)):
                        state["exp_avg_sq"][dim_i] = torch.zeros((p.shape[dim_i]), device=p.device, dtype=p.dtype)
                else:
                    state["exp_avg_sq"] = torch.ones_like(p)
                    # for dim_i in range(len(p.shape)):
                    #     state["exp_avg_sq"][dim_i] = torch.ones((p.shape[dim_i]), device=p.device, dtype=p.dtype)

            exp_avgs.append(state["exp_avg"])
            exp_avg_sqs.append(state["exp_avg_sq"])

            state_steps.append(state["step"])
        return

    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        self._cuda_graph_capture_health_check()

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            state_steps = []
            beta1, beta2 = group["betas"]
            param_names = group["param_names"]

            self._init_group(
                group,
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                state_steps,
                group["update_last"],
                group["update_norm"],
            )

            adafactorw_last(
                param_names,
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                state_steps,
                beta1=beta1,
                beta2=beta2,
                lr=group["lr"],
                weight_decay=group["weight_decay"],
                eps=group["eps"],
                grad_scale=getattr(self, "grad_scale", None),
                found_inf=getattr(self, "found_inf", None),
                neuron_only=group["neuron_only"],
                update_last=group["update_last"],
                update_norm=group["update_norm"],
            )

        return loss

    def get_state_for_param(self, param: nn.Parameter) -> Dict[str, Optional[torch.Tensor]]:
        state = {}
        state["exp_avg"] = self.state[param]["exp_avg"]
        state["exp_avg_sq"] = rank1_approximation(self.state[param]["exp_avg_sq"], self._neuron_only)

        return state  # type: ignore


def rank1_approximation(exp_avg_sq: torch.Tensor, neuron_only):
    if isinstance(exp_avg_sq, torch.Tensor) and len(exp_avg_sq.shape) >=1: return exp_avg_sq

    exp_avg_sq_tmp = exp_avg_sq[0].clone()

    avg = torch.mean(exp_avg_sq[0])

    # Rank 1 approximation to the tensor
    for dim_i in range(1, len(exp_avg_sq)):
        if neuron_only and dim_i == len(exp_avg_sq) - 1:
            cur_shape = exp_avg_sq_tmp.shape
            a_flat = exp_avg_sq_tmp.flatten()
            b_flat = exp_avg_sq[dim_i].mean(dim=0, keepdim=True).expand(*exp_avg_sq[dim_i].shape)
            exp_avg_sq_tmp = torch.outer(a_flat, b_flat)
            # Reshape to dim_i+1 dimensional tensor
            exp_avg_sq_tmp = exp_avg_sq_tmp.view(*cur_shape, exp_avg_sq[dim_i].shape[0])
        else:
            cur_shape = exp_avg_sq_tmp.shape
            a_flat = exp_avg_sq_tmp.flatten()
            b_flat = exp_avg_sq[dim_i]
            exp_avg_sq_tmp = torch.outer(a_flat, b_flat)
            # Reshape to dim_i+1 dimensional tensor
            exp_avg_sq_tmp = exp_avg_sq_tmp.view(*cur_shape, exp_avg_sq[dim_i].shape[0])

    # Check
    if len(exp_avg_sq) > 1:
        exp_avg_sq_tmp /= avg ** (len(exp_avg_sq) - 1)
    return exp_avg_sq_tmp


def adafactorw_last(
    param_names: List[str],
    params: List[torch.Tensor],
    grads: List[torch.Tensor],
    exp_avgs: List[torch.Tensor],
    exp_avg_sqs: List[torch.Tensor],
    state_steps: List[torch.Tensor],
    # kwonly args with defaults are not supported by functions compiled with torchscript issue #70627
    # setting this as kwarg for now as functional API is compiled by torch/distributed/optim
    grad_scale: Optional[torch.Tensor] = None,
    found_inf: Optional[torch.Tensor] = None,
    *,
    beta1: float,
    beta2: float,
    lr: Union[float, torch.Tensor],
    weight_decay: float,
    eps: float,
    neuron_only: bool,
    update_last: bool,
    update_norm: bool,
):
    r"""Functional API that performs AdamW algorithm computation.

    See :class:`~torch.optim.AdamW` for details.
    """
    if not torch._utils.is_compiling() and not all(isinstance(t, torch.Tensor) for t in state_steps):
        raise RuntimeError("API has changed, `state_steps` argument must contain a list of singleton tensors")

    func = _single_tensor_adafactorw_last

    func(
        param_names,
        params,
        grads,
        exp_avgs,
        exp_avg_sqs,
        state_steps,
        beta1=beta1,
        beta2=beta2,
        lr=lr,
        weight_decay=weight_decay,
        eps=eps,
        grad_scale=grad_scale,
        found_inf=found_inf,
        neuron_only=neuron_only,
        update_last=update_last,
        update_norm=update_norm,
    )


def _single_tensor_adafactorw_last(
    param_names: List[str],
    params: List[torch.Tensor],
    grads: List[torch.Tensor],
    exp_avgs: List[torch.Tensor],
    exp_avg_sqs: List[torch.Tensor],
    state_steps: List[torch.Tensor],
    grad_scale: Optional[torch.Tensor],
    found_inf: Optional[torch.Tensor],
    *,
    beta1: float,
    beta2: float,
    lr: Union[torch.Tensor, float],
    weight_decay: float,
    eps: float,
    neuron_only: bool,
    update_last: bool,
    update_norm: bool,
):

    assert grad_scale is None and found_inf is None

    if torch.jit.is_scripting():
        # this assert is due to JIT being dumb and not realizing that the ops below
        # have overloads to handle both float and Tensor lrs, so we just assert it's
        # a float since most people using JIT are using floats
        assert isinstance(lr, float)

    for i, (n, param) in enumerate(zip(param_names, params)):

        grad = grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step_t = state_steps[i]

        # update step
        step_t += 1

        # Perform stepweight decay
        param.data.mul_(1 - lr * weight_decay)

        step = step_t.item()

        # Decay the first and second moment running average coefficient
        exp_avg.lerp_(grad, 1 - beta1)

        if update_last and "ff_out_last" in n:
            for dim_i in range(len(param.shape)):
                contract_dim = [i for i in range(len(param.shape)) if i != dim_i]
                exp_avg_sq[dim_i].lerp_(grad.square().mean(dim=contract_dim), 1 - beta2)
        elif update_norm and ("norm" in n or "ln_f" in n):
            for dim_i in range(len(param.shape)):
                contract_dim = [i for i in range(len(param.shape)) if i != dim_i]
                exp_avg_sq[dim_i].lerp_(grad.square().mean(dim=contract_dim), 1 - beta2)


        # exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

        bias_correction1 = 1 - beta1**step
        step_size = lr / bias_correction1

        if (update_last and "ff_out_last" in n) or (update_norm and ("norm" in n or "ln_f" in n)):
            bias_correction2 = 1 - beta2**step
            bias_correction2_sqrt = bias_correction2**0.5

            exp_avg_sq_tmp = rank1_approximation(exp_avg_sq, neuron_only)

            denom = (exp_avg_sq_tmp.sqrt() / bias_correction2_sqrt).add_(eps)
        else:
            exp_avg_sq_tmp = exp_avg_sq
            denom = exp_avg_sq_tmp.sqrt()

        param.data.addcdiv_(exp_avg, denom, value=-step_size)


class AdalayerWLast(Optimizer):
    def __init__(
        self,
        params,
        lr: Union[float, torch.Tensor] = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        att_correction: bool = False,
        lastlayer_correction: bool = False,
        firstlayer_correction: bool = False,
        no_norm_training: bool = False,
        update_norm: bool = False,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            att_correction=att_correction,
            lastlayer_correction=lastlayer_correction,
            firstlayer_correction=firstlayer_correction,
            no_norm_training=no_norm_training,
            update_norm=update_norm,
        )
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        state_values = list(self.state.values())
        step_is_tensor = (len(state_values) != 0) and torch.is_tensor(state_values[0]["step"])
        if not step_is_tensor:
            for s in state_values:
                s["step"] = torch.tensor(float(s["step"]), dtype=torch.float32)

    def _init_group(
        self,
        group,
        params_with_grad,
        grads,
        exp_avgs,
        exp_avg_sqs,
        state_steps,
        att_correction,
        lastlayer_correction,
        firstlayer_correction,
        update_norm,
    ):
        for p, n in zip(group["params"], group["param_names"]):
            if p.grad is None:
                continue
            params_with_grad.append(p)
            grads.append(p.grad)

            state = self.state[p]

            # State initialization
            if len(state) == 0:
                # note(crcrpar): Deliberately host `step` on CPU if both capturable and fused are off.
                # This is because kernel launches are costly on CUDA and XLA.
                state["step"] = torch.tensor(0.0, dtype=torch.float32)
                # Exponential moving average of gradient values
                state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                # Exponential moving average of squared gradient values marginalized on all but the dim_i^th dimension
                if att_correction and "att_proj" in n:
                    state["exp_avg_sq"] = []
                    for j in range(3):
                        state["exp_avg_sq"].append(torch.tensor(1.0, dtype=torch.float32, device=p.device))
                elif lastlayer_correction and "ff_out" in n and p.shape[0] > 10000:
                    print("Used heuristic to detect last layer")
                    state["exp_avg_sq"] = torch.zeros_like(p.mean(dim=1))
                elif firstlayer_correction and "wte" in n:
                    state["exp_avg_sq"] = torch.zeros_like(p.mean(dim=1))
                elif update_norm and ("norm" in n or "ln_f" in n):
                    state["exp_avg_sq"] = torch.tensor(0.0, dtype=torch.float32, device=p.device)
                else:
                    state["exp_avg_sq"] = torch.tensor(1.0, dtype=torch.float32, device=p.device)

            exp_avgs.append(state["exp_avg"])
            exp_avg_sqs.append(state["exp_avg_sq"])

            state_steps.append(state["step"])
        return

    def step(self, closure=None):
        """Perform a single optimization step.
        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        self._cuda_graph_capture_health_check()

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            state_steps = []
            beta1, beta2 = group["betas"]
            param_names = group["param_names"]

            self._init_group(
                group,
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                state_steps,
                group["att_correction"],
                group["lastlayer_correction"],
                group["firstlayer_correction"],
                group["update_norm"]
            )

            Adalayerw_last(
                param_names,
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                state_steps,
                beta1=beta1,
                beta2=beta2,
                lr=group["lr"],
                weight_decay=group["weight_decay"],
                eps=group["eps"],
                grad_scale=getattr(self, "grad_scale", None),
                found_inf=getattr(self, "found_inf", None),
                att_correction=group["att_correction"],
                lastlayer_correction=group["lastlayer_correction"],
                firstlayer_correction=group["firstlayer_correction"],
                no_norm_training=group["no_norm_training"],
                update_norm=group["update_norm"],
            )

        return loss

    def get_state_for_param(self, param: nn.Parameter) -> Dict[str, Optional[torch.Tensor]]:
        state = {}
        state["exp_avg"] = self.state[param]["exp_avg"]
        # check if state["exp_avg_sq"] is a list
        if isinstance(self.state[param]["exp_avg_sq"], list):
            state["exp_avg_sq"] = torch.zeros_like(state["exp_avg"])
            shape0 = state["exp_avg"].shape[0]
            for j in range(3):
                state["exp_avg_sq"][j * (shape0 // 3) : (j + 1) * (shape0 // 3)] += self.state[param][
                    "exp_avg_sq"
                ][j]
        else:
            state["exp_avg_sq"] = torch.zeros_like(self.state[param]["exp_avg_sq"])
            state["exp_avg_sq"] += self.state[param]["exp_avg_sq"]

        return state  # type: ignore


def Adalayerw_last(
    param_names: List[str],
    params: List[torch.Tensor],
    grads: List[torch.Tensor],
    exp_avgs: List[torch.Tensor],
    exp_avg_sqs: List[torch.Tensor],
    state_steps: List[torch.Tensor],
    # kwonly args with defaults are not supported by functions compiled with torchscript issue #70627
    # setting this as kwarg for now as functional API is compiled by torch/distributed/optim
    grad_scale: Optional[torch.Tensor] = None,
    found_inf: Optional[torch.Tensor] = None,
    *,
    beta1: float,
    beta2: float,
    lr: Union[float, torch.Tensor],
    weight_decay: float,
    eps: float,
    att_correction: bool = False,
    lastlayer_correction: bool = False,
    firstlayer_correction: bool = False,
    no_norm_training: bool = False,
    update_norm: bool = False,
):
    r"""Functional API that performs AdamW algorithm computation.
    See :class:`~torch.optim.AdamW` for details.
    """
    if not torch._utils.is_compiling() and not all(isinstance(t, torch.Tensor) for t in state_steps):
        raise RuntimeError("API has changed, `state_steps` argument must contain a list of singleton tensors")

    func = _single_tensor_Adalayerw_last

    func(
        param_names,
        params,
        grads,
        exp_avgs,
        exp_avg_sqs,
        state_steps,
        beta1=beta1,
        beta2=beta2,
        lr=lr,
        weight_decay=weight_decay,
        eps=eps,
        grad_scale=grad_scale,
        found_inf=found_inf,
        att_correction=att_correction,
        lastlayer_correction=lastlayer_correction,
        firstlayer_correction=firstlayer_correction,
        no_norm_training=no_norm_training,
        update_norm=update_norm,
    )


def _single_tensor_Adalayerw_last(
    param_names: List[str],
    params: List[torch.Tensor],
    grads: List[torch.Tensor],
    exp_avgs: List[torch.Tensor],
    exp_avg_sqs: List[torch.Tensor],
    state_steps: List[torch.Tensor],
    grad_scale: Optional[torch.Tensor],
    found_inf: Optional[torch.Tensor],
    *,
    beta1: float,
    beta2: float,
    lr: Union[torch.Tensor, float],
    weight_decay: float,
    eps: float,
    att_correction: bool = False,
    lastlayer_correction: bool = False,
    firstlayer_correction: bool = False,
    no_norm_training: bool = False,
    update_norm: bool = False,
):

    assert grad_scale is None and found_inf is None

    if torch.jit.is_scripting():
        # this assert is due to JIT being dumb and not realizing that the ops below
        # have overloads to handle both float and Tensor lrs, so we just assert it's
        # a float since most people using JIT are using floats
        assert isinstance(lr, float)

    for i, (n, param) in enumerate(zip(param_names, params)):
        if no_norm_training and len(param.shape) == 1 and ('norm' in n or 'ln_f' in n):
            continue

        grad = grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step_t = state_steps[i]

        # update step
        step_t += 1

        # Perform stepweight decay
        param.data.mul_(1 - lr * weight_decay)

        step = step_t.item()

        # Decay the first and second moment running average coefficient
        exp_avg.lerp_(grad, 1 - beta1)
        if att_correction and "att_proj" in n:
            shape0 = grad.shape[0]
            # print(shape0)
            exp_avg_sq_tmp = torch.ones_like(grad)
            for j in range(3):
                exp_avg_sq_tmp[j * shape0 // 3 : (j + 1) * shape0 // 3] *= exp_avg_sq[j]
        elif lastlayer_correction and "ff_out" in n and grad.shape[0] > 10000:
            # print(exp_avg_sq.shape)
            exp_avg_sq_tmp = torch.ones_like(grad)
            exp_avg_sq.lerp_(grad.square().mean(dim=1), 1 - beta2)
            exp_avg_sq_tmp *= exp_avg_sq[:, None]
        elif firstlayer_correction and "wte" in n:
            exp_avg_sq_tmp = torch.ones_like(grad)
            exp_avg_sq.lerp_(grad.square().mean(dim=1), 1 - beta2)
            exp_avg_sq_tmp *= exp_avg_sq[:, None]
        elif update_norm and ("norm" in n or "ln_f" in n):
            exp_avg_sq.lerp_(grad.square().mean(), 1 - beta2)
            exp_avg_sq_tmp = exp_avg_sq
        else:
            exp_avg_sq_tmp = torch.ones_like(grad)
            exp_avg_sq_tmp *= exp_avg_sq # after t_freeze, don't update exp_avg_sq
        
        bias_correction1 = 1 - beta1**step
        step_size = lr / bias_correction1

        if (lastlayer_correction and "ff_out" in n and grad.shape[0] > 10000) or (firstlayer_correction and "wte" in n) or (update_norm and ("norm" in n or "ln_f" in n)):
            bias_correction2 = 1 - beta2**step
            bias_correction2_sqrt = bias_correction2**0.5

            denom = (exp_avg_sq_tmp.sqrt() / bias_correction2_sqrt).add_(eps)
        else:
            denom = exp_avg_sq_tmp.sqrt()

        param.data.addcdiv_(exp_avg, denom, value=-step_size)


@dataclass
class Scheduler(metaclass=ABCMeta):
    # NOTE: these fields are not given default values because otherwise dataclasses complains
    # about how the scheduler subclasses are defined.
    grad_clip_warmup_steps: Optional[int]
    grad_clip_warmup_factor: Optional[float]

    @abstractmethod
    def get_lr(self, initial_lr: float, step: int, max_steps: int) -> float:
        raise NotImplementedError

    def _get_max_grad_norm_coeff(
        self, initial_value: Optional[float], step: int, max_steps: int
    ) -> Optional[float]:
        del max_steps  # might need this in the future, but for now I just wanted to match the API of `get_lr()`.
        if initial_value is None:
            return None
        elif (
            self.grad_clip_warmup_steps is None
            or self.grad_clip_warmup_factor is None
            or step > self.grad_clip_warmup_steps
        ):
            return initial_value
        else:
            return self.grad_clip_warmup_factor * initial_value

    def get_max_grad_norm(
        self, initial_max_grad_norm: Optional[float], step: int, max_steps: int
    ) -> Optional[float]:
        return self._get_max_grad_norm_coeff(initial_max_grad_norm, step, max_steps)

    def get_max_grad_norm_ratio(
        self, initial_max_grad_norm_ratio: Optional[float], step: int, max_steps: int
    ) -> Optional[float]:
        return self._get_max_grad_norm_coeff(initial_max_grad_norm_ratio, step, max_steps)

    def _linear_warmup(self, initial_lr: float, step: int, warmup_steps: int = 2000, alpha=0.1) -> float:
        return initial_lr * (alpha + (1.0 - alpha) * min(step, warmup_steps) / warmup_steps)


@dataclass
class CosWithWarmup(Scheduler):
    warmup_steps: int
    alpha_0: float = 0.1
    alpha_f: float = 0.1
    t_max: Optional[int] = None

    def get_lr(self, initial_lr: float, step: int, max_steps: int) -> float:
        max_steps = max_steps if self.t_max is None else self.t_max
        eta_min = initial_lr * self.alpha_f
        if step < self.warmup_steps:
            return self._linear_warmup(initial_lr, step, self.warmup_steps, alpha=self.alpha_0)
        elif step >= max_steps:
            return eta_min
        else:
            step = step - self.warmup_steps
            max_steps = max_steps - self.warmup_steps
            return eta_min + (initial_lr - eta_min) * (1 + cos(pi * step / max_steps)) / 2

@dataclass
class FreezeCosWithWarmup(Scheduler):
    t_freeze: int
    warmup_steps: int
    alpha_0: float = 0.1
    alpha_f: float = 0.1
    t_max: Optional[int] = None

    def get_lr(self, initial_lr: float, step: int, max_steps: int) -> float:
        max_steps = max_steps if self.t_max is None else self.t_max
        eta_min = initial_lr * self.alpha_f
        if step < self.t_freeze:
            return 0
        elif step < self.t_freeze + self.warmup_steps:
            return self._linear_warmup(initial_lr, step - self.t_freeze, self.warmup_steps, alpha=self.alpha_0)
        elif step >= max_steps:
            return eta_min
        else:
            step = step - self.warmup_steps - self.t_freeze
            max_steps = max_steps - self.warmup_steps - self.t_freeze
            return eta_min + (initial_lr - eta_min) * (1 + cos(pi * step / max_steps)) / 2


@dataclass
class LinearWithWarmup(Scheduler):
    warmup_steps: int
    alpha_f: float = 0.1
    t_max: Optional[int] = None

    def get_lr(self, initial_lr: float, step: int, max_steps: int) -> float:
        max_steps = max_steps if self.t_max is None else self.t_max
        eta_min = initial_lr * self.alpha_f
        if step < self.warmup_steps:
            return self._linear_warmup(initial_lr, step, self.warmup_steps)
        elif step >= max_steps:
            return eta_min
        else:
            step = step - self.warmup_steps
            max_steps = max_steps - self.warmup_steps
            return initial_lr - (initial_lr - eta_min) * (step / max_steps)


@dataclass
class InvSqrtWithWarmup(Scheduler):
    warmup_steps: int

    def get_lr(self, initial_lr: float, step: int, max_steps: int) -> float:
        if step < self.warmup_steps:
            return self._linear_warmup(initial_lr, step, self.warmup_steps)
        del max_steps
        return initial_lr * sqrt(self.warmup_steps / max(self.warmup_steps, step))


@dataclass
class MaxScheduler(Scheduler):
    sched1: Scheduler
    sched2: Scheduler

    def get_lr(self, initial_lr: float, step: int, max_steps: int) -> float:
        return max(
            self.sched1.get_lr(initial_lr, step, max_steps), self.sched2.get_lr(initial_lr, step, max_steps)
        )


@dataclass
class BoltOnWarmupScheduler(Scheduler):
    inner: Scheduler
    warmup_start: int
    warmup_end: int

    @classmethod
    def wrap(cls, scheduler: Scheduler, warmup_start: int, warmup_end: int) -> "BoltOnWarmupScheduler":
        return cls(
            grad_clip_warmup_steps=None,
            grad_clip_warmup_factor=None,
            inner=scheduler,
            warmup_start=warmup_start,
            warmup_end=warmup_end,
        )

    def get_lr(self, initial_lr: float, step: int, max_steps: int) -> float:
        if step < self.warmup_start:
            return 0.0
        if step < self.warmup_end:
            lr_at_intercept = self.inner.get_lr(initial_lr, self.warmup_end, max_steps)
            return lr_at_intercept * (step - self.warmup_start) / (self.warmup_end - self.warmup_start)
        else:
            return self.inner.get_lr(initial_lr, step, max_steps)

    def _get_max_grad_norm_coeff(
        self, initial_value: Optional[float], step: int, max_steps: int
    ) -> Optional[float]:
        return self.inner._get_max_grad_norm_coeff(initial_value, step, max_steps)


@dataclass
class ConstantScheduler(Scheduler):
    def get_lr(self, initial_lr: float, step: int, max_steps: int) -> float:
        del step, max_steps
        return initial_lr


PARAM_GROUP_FIELDS = ("sharded", "max_grad_norm", "max_grad_norm_ratio", "param_names")


def get_param_groups(cfg: TrainConfig, model: nn.Module) -> List[Dict[str, Any]]:
    """
    Separate parameters into weight decay and non weight decay groups, and separate norm layers and last layer (if applicable).
    """
    param_groups: List[Dict[str, Any]]
    param_group_defaults = {
        "sharded": isinstance(model, FullyShardedDataParallel),
        "max_grad_norm": cfg.max_grad_norm,
        "max_grad_norm_ratio": cfg.max_grad_norm_ratio,
    }

    # Separate out parameters that we don't want to apply weight decay to, like norms and biases.
    decay = set()
    no_decay = set()
    last_layer = None
    norm_layers = None
    all_params = {}
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            # NOTE: because named_modules and named_parameters are recursive
            # we will see the same tensors p many many times, but doing it this way
            # allows us to know which parent module any tensor p belongs to...
            if not p.requires_grad:
                continue

            fpn = f"{mn}.{pn}" if mn else pn
            all_params[fpn] = p

            if pn.endswith("bias"):
                if cfg.optimizer.decay_norm_and_bias:
                    decay.add(fpn)
                else:
                    no_decay.add(fpn)
            elif (cfg.optimizer.name == OptimizerType.adalayerw_last or (cfg.optimizer.name == OptimizerType.adalayerw and cfg.optimizer.t_freeze) or cfg.optimizer.name == OptimizerType.adafactorw_last) and cfg.optimizer.update_norm and ('norm' in fpn or 'ln_f' in fpn):
                if norm_layers is None:
                    norm_layers = set()
                norm_layers.add(fpn)
            elif (cfg.optimizer.name == OptimizerType.adalayerw_last or (cfg.optimizer.name == OptimizerType.adalayerw and cfg.optimizer.t_freeze) or cfg.optimizer.name == OptimizerType.adafactorw_last) and cfg.optimizer.update_last and "ff_out_last" in fpn: #heuristic for last layer
                log.info("Adding last layer to separate param group")
                last_layer = set()
                last_layer.add(fpn)
            elif pn.endswith("weight") and isinstance(m, nn.Linear):
                decay.add(fpn)
            elif pn.endswith("weight") and isinstance(m, (LayerNormBase, nn.LayerNorm)):
                if cfg.optimizer.decay_norm_and_bias:
                    decay.add(fpn)
                else:
                    no_decay.add(fpn)
            elif pn.endswith("weight") and isinstance(m, nn.Embedding):
                if cfg.optimizer.decay_embeddings:
                    decay.add(fpn)
                else:
                    no_decay.add(fpn)

    # Validate that we've considered every parameter
    if last_layer and norm_layers:
        inter_params = decay & no_decay & last_layer & norm_layers
        union_params = decay | no_decay | last_layer | norm_layers
    elif last_layer:
        inter_params = decay & no_decay & last_layer
        union_params = decay | no_decay | last_layer
    elif norm_layers:
        inter_params = decay & no_decay & norm_layers
        union_params = decay | no_decay | norm_layers
    else:
        inter_params = decay & no_decay
        union_params = decay | no_decay

    assert len(inter_params) == 0, f"parameters {inter_params} made it into both decay/no_decay sets!"
    assert (
        len(all_params.keys() - union_params) == 0
    ), f"parameters {all_params.keys() - union_params} were not separated into either decay/no_decay set!"

    # Create the pytorch optimizer groups.
    decay_sorted = sorted(list(decay))
    no_decay_sorted = sorted(list(no_decay))
    param_groups = []
    if len(decay_sorted) > 0:
        param_groups.append(
            {
                "params": [all_params[pn] for pn in decay_sorted],
                "param_names": decay_sorted,
                **param_group_defaults,
            }
        )
    if len(no_decay_sorted) > 0:
        param_groups.append(
            {
                "params": [all_params[pn] for pn in no_decay_sorted],
                "param_names": no_decay_sorted,
                "weight_decay": 0.0,
                **param_group_defaults,
            }
        )
    if last_layer:
        log.info("Adding last layer group to param group")
        last_layer_sorted = sorted(list(last_layer))
        if cfg.optimizer.lr_last_multiplier is not None:
            lr_last = cfg.optimizer.learning_rate * cfg.optimizer.lr_last_multiplier
        else:
            lr_last = cfg.optimizer.lr_last
        param_groups.append(
            {
                "params": [all_params[pn] for pn in last_layer_sorted],
                "param_names": last_layer_sorted,
                "lr": lr_last, #should be set if optimizer is adalayerw_last
                "initial_lr": lr_last,
                **param_group_defaults,
            }
        )
    
    if norm_layers:
        log.info("Adding norm layer group to param group")
        norm_layers_sorted = sorted(list(norm_layers))
        
        if cfg.optimizer.lr_last_multiplier is not None:
            lr_last = cfg.optimizer.learning_rate * cfg.optimizer.lr_last_multiplier
        else:
            lr_last = cfg.optimizer.lr_last
        param_groups.append(
            {
                "params": [all_params[pn] for pn in norm_layers_sorted],
                "param_names": norm_layers_sorted,
                "lr": lr_last,
                "initial_lr": lr_last,
                **param_group_defaults,
            }
        )

    # Validate fields.
    for group in param_groups:
        for key in PARAM_GROUP_FIELDS:
            assert key in group

    return param_groups


def fix_optim_state_dict(optimizer: Optimizer, state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Make sure old optim state dicts are compatible with new versions.
    """
    if len(state_dict["param_groups"]) == 1 and len(optimizer.param_groups) == 2:
        assert optimizer.param_groups[1]["weight_decay"] == 0.0

        # Decay
        decay_param_group = {k: v for k, v in state_dict["param_groups"][0].items() if k != "params"}
        decay_param_group["params"] = optimizer.state_dict()["param_groups"][0]["params"]

        # No decay.
        no_decay_param_group = {k: v for k, v in state_dict["param_groups"][0].items() if k != "params"}
        no_decay_param_group["weight_decay"] = 0.0
        no_decay_param_group["params"] = optimizer.state_dict()["param_groups"][1]["params"]

        state_dict["param_groups"] = [decay_param_group, no_decay_param_group]

    assert len(optimizer.param_groups) == len(state_dict["param_groups"])

    # Make sure:
    #  - All required fields are included in the state dict,
    #  - And that the values of those fields doesn't change from what's currently set in the optimizer,
    #    since we might have changed those fields on purpose after a restart.
    for group, sd_group in zip(optimizer.param_groups, state_dict["param_groups"]):
        for key in PARAM_GROUP_FIELDS:
            sd_group[key] = group[key]

    return state_dict


def build_optimizer(cfg: TrainConfig, model: nn.Module) -> Optimizer:
    param_groups = get_param_groups(cfg, model)
    log.info(f"Param groups: {param_groups}")
    log.info(f"Constructing optimizer with {len(param_groups)} param groups")
    if cfg.optimizer.decouple_weight_decay:
        wd = cfg.optimizer.weight_decay / cfg.optimizer.learning_rate
    else:
        wd = cfg.optimizer.weight_decay
    if cfg.optimizer.tie_betas:
        cfg.optimizer.beta_1 = cfg.optimizer.beta_0

    if cfg.optimizer.name == OptimizerType.lionw:
        return LionW(
            param_groups,
            lr=cfg.optimizer.learning_rate,
            betas=(cfg.optimizer.beta_0, cfg.optimizer.beta_1),
            weight_decay=wd,
        )
    elif cfg.optimizer.name == OptimizerType.signsgd:
        return LionW(
            param_groups,
            lr=cfg.optimizer.learning_rate,
            betas=(cfg.optimizer.beta_0, cfg.optimizer.beta_0),
            weight_decay=wd,
        )
    elif cfg.optimizer.name == OptimizerType.adamw:
        return AdamW(
            param_groups,
            lr=cfg.optimizer.learning_rate,
            betas=(cfg.optimizer.beta_0, cfg.optimizer.beta_1),
            weight_decay=wd,
            eps=cfg.optimizer.eps,
        )
    elif cfg.optimizer.name == OptimizerType.adafactorw:
        return AdafactorW(
            param_groups,
            lr=cfg.optimizer.learning_rate,
            betas=(cfg.optimizer.beta_0, cfg.optimizer.beta_1),
            weight_decay=wd,
            eps=cfg.optimizer.eps,
            neuron_only=cfg.optimizer.neuron_only,
        )
    elif cfg.optimizer.name == OptimizerType.adalayerw:
        return AdalayerW(
            param_groups,
            lr=cfg.optimizer.learning_rate,
            betas=(cfg.optimizer.beta_0, cfg.optimizer.beta_1),
            weight_decay=wd,
            eps=cfg.optimizer.eps,
            att_correction=cfg.optimizer.att_correction,
            lastlayer_correction=cfg.optimizer.lastlayer_correction,
            firstlayer_correction = cfg.optimizer.firstlayer_correction,
            update_last=cfg.optimizer.update_last,
            update_norm=cfg.optimizer.update_norm,
            t_freeze=cfg.optimizer.t_freeze,
            no_norm_training=cfg.optimizer.no_norm_training
        )
    elif cfg.optimizer.name == OptimizerType.adafactorw_last:
        return AdafactorWLast(
            param_groups,
            lr=cfg.optimizer.learning_rate,
            betas=(cfg.optimizer.beta_0, cfg.optimizer.beta_1),
            weight_decay=wd,
            eps=cfg.optimizer.eps,
            neuron_only=cfg.optimizer.neuron_only,
            update_last=cfg.optimizer.update_last,
            update_norm=cfg.optimizer.update_norm,
        )
    elif cfg.optimizer.name == OptimizerType.adalayerw_last:
        return AdalayerWLast(
            param_groups,
            lr=cfg.optimizer.learning_rate,
            betas=(cfg.optimizer.beta_0, cfg.optimizer.beta_1),
            weight_decay=wd,
            eps=cfg.optimizer.eps,
            att_correction=cfg.optimizer.att_correction,
            lastlayer_correction=cfg.optimizer.lastlayer_correction,
            firstlayer_correction = cfg.optimizer.firstlayer_correction,
            no_norm_training=cfg.optimizer.no_norm_training,
            update_norm=cfg.optimizer.update_norm,
        )
    elif cfg.optimizer.name == OptimizerType.sgdw:
        return SGDW(
            param_groups,
            lr=cfg.optimizer.learning_rate,
            momentum=cfg.optimizer.beta_0,
            dampening=cfg.optimizer.beta_0 if cfg.optimizer.use_dampening else 0.0,
            weight_decay=wd,
            nesterov=cfg.optimizer.nesterov,
        )
    elif cfg.optimizer.name == OptimizerType.sophiag:
        return SophiaG(
            param_groups,
            lr=cfg.optimizer.learning_rate,
            betas=(cfg.optimizer.beta_0, cfg.optimizer.beta_1),
            weight_decay=wd,
            rho=cfg.optimizer.eps,
        )
    else:
        raise NotImplementedError


def build_scheduler(cfg: TrainConfig, sched_cfg: Optional[SchedulerConfig] = None) -> Scheduler:
    sched_cfg = sched_cfg if sched_cfg is not None else cfg.scheduler
    if sched_cfg.name == SchedulerType.cosine_with_warmup:
        return CosWithWarmup(
            grad_clip_warmup_steps=(
                None if sched_cfg.grad_clip_warmup_steps is None else int(sched_cfg.grad_clip_warmup_steps)
            ),
            grad_clip_warmup_factor=sched_cfg.grad_clip_warmup_factor,
            warmup_steps=int(sched_cfg.t_warmup),
            alpha_f=sched_cfg.alpha_f,
            alpha_0=sched_cfg.alpha_0,
            t_max=None if sched_cfg.t_max is None else int(sched_cfg.t_max),
        )
    elif sched_cfg.name == SchedulerType.freeze_cosine_with_warmup:
        assert cfg.optimizer.t_freeze is not None, "t_freeze must be set to an integer to use this scheduler."
        return FreezeCosWithWarmup(
            t_freeze=cfg.optimizer.t_freeze,
            grad_clip_warmup_steps=(
                None if sched_cfg.grad_clip_warmup_steps is None else int(sched_cfg.grad_clip_warmup_steps)
            ),
            grad_clip_warmup_factor=sched_cfg.grad_clip_warmup_factor,
            warmup_steps=int(sched_cfg.t_warmup),
            alpha_f=sched_cfg.alpha_f,
            alpha_0=sched_cfg.alpha_0,
            t_max=None if sched_cfg.t_max is None else int(sched_cfg.t_max),
        )
    elif sched_cfg.name == SchedulerType.linear_with_warmup:
        return LinearWithWarmup(
            grad_clip_warmup_steps=(
                None if sched_cfg.grad_clip_warmup_steps is None else int(sched_cfg.grad_clip_warmup_steps)
            ),
            grad_clip_warmup_factor=sched_cfg.grad_clip_warmup_factor,
            warmup_steps=int(sched_cfg.t_warmup),
            alpha_f=sched_cfg.alpha_f,
            t_max=None if sched_cfg.t_max is None else int(sched_cfg.t_max),
        )
    elif sched_cfg.name == SchedulerType.inverse_sqrt_with_warmup:
        return InvSqrtWithWarmup(
            grad_clip_warmup_steps=(
                None if sched_cfg.grad_clip_warmup_steps is None else int(sched_cfg.grad_clip_warmup_steps)
            ),
            grad_clip_warmup_factor=sched_cfg.grad_clip_warmup_factor,
            warmup_steps=int(sched_cfg.t_warmup),
        )
    elif sched_cfg.name == SchedulerType.max_scheduler:
        return MaxScheduler(
            grad_clip_warmup_steps=(
                None if sched_cfg.grad_clip_warmup_steps is None else int(sched_cfg.grad_clip_warmup_steps)
            ),
            grad_clip_warmup_factor=sched_cfg.grad_clip_warmup_factor,
            sched1=build_scheduler(cfg, replace(sched_cfg, name=SchedulerType.cosine_with_warmup)),
            sched2=build_scheduler(cfg, replace(sched_cfg, name=SchedulerType.inverse_sqrt_with_warmup)),
        )
    elif sched_cfg.name == SchedulerType.constant:
        return ConstantScheduler(
            grad_clip_warmup_steps=(
                None if sched_cfg.grad_clip_warmup_steps is None else int(sched_cfg.grad_clip_warmup_steps)
            ),
            grad_clip_warmup_factor=sched_cfg.grad_clip_warmup_factor,
        )
    else:
        raise NotImplementedError
