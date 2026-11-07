from __future__ import annotations


import torch


import torch.nn as nn


from mimic_baseline.tasks.supermimic.learning.models import WM_OUT


from mimic_baseline.tasks.supermimic.learning.perception import (
    CLEAN_DEPTH_CLIP_M,
    height_scan_orientation,
)


from mimic_baseline.tasks.supermimic.learning.perception_loss import (
    cap_decoder_physics_gradients,
)


from .common import _valid_depth


class WorldModelUpdateMixin:
    """Original Trainer methods; state is owned by Trainer."""

    def _encode_clean_perception(self, state, scan, hit, depth):
        if getattr(self.perception, "world_frame", False):
            return self.perception.encode(
                scan, hit, depth, height_scan_orientation(state[1])
            )
        return self.perception.encode(scan, hit, depth)

    @torch.no_grad()
    def _burnin_hidden(self, history: dict[str, torch.Tensor]) -> torch.Tensor:
        """Recompute recurrent context with current weights, within one episode.

        The finite prefix is truncated BPTT context, not a cached hidden state
        from older model weights. Padded rows leave the zero reset state intact.
        """
        batch, length = history["mask"].shape
        hidden = self.perception.initial_hidden(batch, self.device)
        if length == 0:
            return hidden
        # Recorded frames are independent inputs to the encoders and WM. Batch
        # those computations; only the GRU recurrence must run in time order.
        historical_state = self._as_state(history["s"].flatten(0, 1))
        features = self._encode_clean_perception(
            historical_state,
            history["scan"].flatten(0, 1),
            history["hit"].flatten(0, 1),
            history["clean_depth"].flatten(0, 1).float(),
        )
        _, driver = self.wm.step(
            historical_state,
            history["a"].flatten(0, 1),
            history["a_prev"].flatten(0, 1),
            self.normalize_xi(history["xi"].flatten(0, 1)),
            features,
        )
        features = features.reshape(batch, length, -1)
        driver = driver.reshape(batch, length, -1)
        for step in range(length):
            next_hidden = self.perception.advance_hidden(
                features[:, step], driver[:, step], hidden
            )
            hidden = torch.where(history["mask"][:, step, None], next_hidden, hidden)
        return hidden

    def _clean_prediction(self, state, action, previous, xi, scan, hit, depth, hidden):
        features = self._encode_clean_perception(state, scan, hit, depth)
        state, driver = self.wm.step(state, action, previous, xi, features)
        hidden, scan, logits, depth = self.perception.step(
            features, driver, hidden, _valid_depth(depth)
        )
        return state, hidden, scan, logits, depth

    @staticmethod
    def _perception_errors(scan, logits, depth, target_scan, target_hit, target_depth):
        # Pixel/cell means prevent resolution from scaling the model objective.
        height_error = torch.where(target_hit, scan - target_scan.float(), 0.0)
        height_count = target_hit.sum(-1).clamp_min(1)
        depth_valid = _valid_depth(target_depth)
        depth_error = torch.where(depth_valid, depth - target_depth.float(), 0.0)
        depth_count = depth_valid.flatten(1).sum(-1).clamp_min(1)
        height_mse = height_error.square().sum(-1) / height_count
        depth_mse = depth_error.square().flatten(1).sum(-1) / depth_count
        hit_bce = nn.functional.binary_cross_entropy_with_logits(
            logits, target_hit.to(logits.dtype), reduction="none"
        ).mean(-1)
        return height_mse, depth_mse, hit_bce

    def _clean_world_model_rollout(self, sample, window):
        state = self._as_state(sample["s"][:, 0])
        hidden = self._burnin_hidden(sample["burnin"])
        scan, hit = sample["scan"][:, 0].float(), sample["hit"][:, 0]
        depth = sample["clean_depth"][:, 0].float()
        losses, physical, normalized, sensory = [], [], [], []
        reconstruction_losses = []
        # Prediction-only runners predate the optional loss configuration.
        config = getattr(self, "cfg", None)
        height_weight = getattr(config, "perception_height_weight", 1.0)
        depth_weight = getattr(config, "perception_depth_weight", 1.0)
        cap_ratio = getattr(config, "perception_decoder_physics_ratio", None)
        std = torch.maximum(self._norm_std(self.wm.out_norm), self._floor)
        for step in range(window):
            state, hidden, scan, logits, depth = self._clean_prediction(
                state,
                sample["a"][:, step],
                sample["a_prev"][:, step],
                self.normalize_xi(sample["xi"][:, step]),
                scan,
                hit,
                depth,
                hidden,
            )
            hit = logits.sigmoid()
            mask = sample["mask"][:, step]
            error = self.wm.state_error(state, self._as_state(sample["s1"][:, step]))
            height_mse, depth_mse, hit_bce = self._perception_errors(
                scan,
                logits,
                depth,
                sample["scan1"][:, step],
                sample["hit1"][:, step],
                sample["clean_depth1"][:, step],
            )
            # Physical SSE is unchanged. Sensor losses have fixed metric scales:
            # height / 1 m, ERP / 8 m, and hit BCE; all averaged per image.
            physics = error.square().sum(-1)
            losses.append(
                (
                    (
                        physics
                        + height_weight * height_mse
                        + depth_weight * (depth_mse / CLEAN_DEPTH_CLIP_M**2)
                        + hit_bce
                    )
                    * mask
                ).sum()
            )
            if cap_ratio is not None:
                reconstruction_losses.append(
                    (
                        (
                            height_weight * height_mse
                            + depth_weight * (depth_mse / CLEAN_DEPTH_CLIP_M**2)
                            + hit_bce
                        )
                        * mask
                    ).sum()
                )
            physical.append((error.square().mean(-1) * mask).sum())
            normalized.append(((error / std).square().mean(-1) * mask).sum())
            sensory.append(
                torch.stack(
                    (
                        (height_mse * mask).sum(),
                        (depth_mse * mask).sum(),
                        (hit_bce * mask).sum(),
                    )
                )
            )
        means = torch.stack(sensory).sum(0) / sample["mask"].sum().clamp_min(1)
        return {
            "state": state,
            "latent": None,
            "losses": losses,
            "reconstruction_loss": torch.stack(reconstruction_losses).sum()
            if reconstruction_losses
            else None,
            "physical_errors": physical,
            "normalized_errors": normalized,
            "latent_errors": [],
            "scan_metrics": {
                "Perception/clean_height_rmse_m": means[0].sqrt().detach(),
                "Perception/clean_depth_rmse_m": means[1].sqrt().detach(),
                "Perception/height_hit_bce": means[2].detach(),
                "Loss/height_prediction": means[0].detach(),
                "Loss/depth_prediction": (means[1] / CLEAN_DEPTH_CLIP_M**2).detach(),
                "Loss/height_hit": means[2].detach(),
                **self._perception_loss_metrics,
            },
        }

    def _world_model_rollout(self, sample: dict[str, torch.Tensor], window: int):
        if self.recurrent:
            return self._clean_world_model_rollout(sample, window)
        state = self._as_state(sample["s"][:, 0])
        latent = self._latent_at(sample, 0) if self.perception is not None else None
        losses: list[torch.Tensor] = []
        normalized_errors: list[torch.Tensor] = []
        physical_errors: list[torch.Tensor] = []
        latent_errors: list[torch.Tensor] = []
        scan_metrics: dict[str, torch.Tensor] = {}
        std = torch.maximum(self._norm_std(self.wm.out_norm), self._floor)
        latent_std = std[..., WM_OUT:] if self.perception is not None else None
        for step in range(window):
            mask = sample["mask"][:, step]
            if self.perception is None:
                state = self.wm.step(
                    state,
                    sample["a"][:, step],
                    sample["a_prev"][:, step],
                    self.normalize_xi(sample["xi"][:, step]),
                )
                error = self.wm.state_error(
                    state, self._as_state(sample["s1"][:, step])
                )
            else:
                state, latent = self.wm.step(
                    state,
                    sample["a"][:, step],
                    sample["a_prev"][:, step],
                    self.normalize_xi(sample["xi"][:, step]),
                    latent,
                )
                target = self._as_state(sample["s1"][:, step])
                error = self.wm.state_error(state, target)
                # Encode the pre-reset endpoint with current weights. The
                # detached target does not pull the encoder toward predictions.
                with torch.no_grad():
                    target_latent = self._latent_at(sample, step, next_state=True)
                latent_error = latent - target_latent
                losses.append(
                    latent_error.square()
                    .mean(-1)
                    .mul(self.cfg.perception_latent_weight)
                    .mul(mask)
                    .sum()
                )
                latent_errors.append(
                    (latent_error / latent_std).square().mean(-1).mul(mask).sum()
                )
            normalized = (error[..., :WM_OUT] / std[..., :WM_OUT]) ** 2
            # Local raw-state SSE choice. Eq. (1) leaves state scaling unspecified;
            # the reference implementation instead uses inverse-variance weights.
            losses.append((error[..., :WM_OUT].square().sum(-1) * mask).sum())
            normalized_errors.append((normalized.mean(-1) * mask).sum())
            physical_errors.append((error[..., :WM_OUT].square().mean(-1) * mask).sum())
        if self.perception is not None:
            # Two reconstruction terms, each evaluated once per window:
            #
            # * the recorded first frame pins the encoder's latent to terrain
            #   geometry, which is what stops a latent-only objective from
            #   collapsing toward a constant;
            # * the rolled-out last latent is decoded to the scan at t+k, so the
            #   predictor is also scored on the metric quantity the actor will
            #   effectively see during imagination.
            first_mask = sample["mask"][:, 0]
            losses.append(
                self._reconstruction_loss(
                    self._latent_at(sample, 0), sample, 0, first_mask
                )
            )
            scan_metrics = self._scan_metrics(
                latent,
                sample,
                window - 1,
                sample["mask"][:, window - 1],
                next_state=True,
            )
            losses.append(
                scan_metrics.pop("Perception/predicted_sse")
                * self.cfg.perception_predicted_scan_weight
            )
        return {
            "state": state,
            "latent": latent,
            "losses": losses,
            "normalized_errors": normalized_errors,
            "physical_errors": physical_errors,
            "latent_errors": latent_errors,
            "scan_metrics": scan_metrics,
        }

    def _latent_at(
        self, sample: dict[str, torch.Tensor], index: int, *, next_state: bool = False
    ) -> torch.Tensor:
        frame = sample["depth1" if next_state else "depth"][:, index]
        return self.perception.encode(frame, _valid_depth(frame))

    def _reconstruction_loss(
        self,
        latent: torch.Tensor,
        sample: dict[str, torch.Tensor],
        index: int,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Reconstruct the recorded height scan from a latent.

        The scan is the encoder's supervision target in metric space; the
        decoder is scored with the same alive mask as the dynamics so a window
        never trains on frames past a reset.
        """
        scan = sample["scan"][:, index].float()
        hit = sample["hit"][:, index]
        predicted_scan, hit_logit = self.perception.decode(latent)
        observed = hit & mask[:, None].bool()
        height_error = torch.where(observed, predicted_scan - scan, 0.0)
        reconstruction = height_error.square().sum(-1)
        bce = nn.functional.binary_cross_entropy_with_logits(
            hit_logit, hit.to(scan.dtype), reduction="none"
        ).mean(-1)
        return (
            ((reconstruction + bce) * self.cfg.perception_reconstruction_weight)
            .mul(mask)
            .sum()
        )

    def _scan_metrics(
        self,
        latent: torch.Tensor,
        sample: dict[str, torch.Tensor],
        index: int,
        mask: torch.Tensor,
        *,
        next_state: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Report the predicted latent's decoded scan error in metres.

        Two numbers, kept separate on purpose: the latent L2 says whether the
        rollout state stayed close, and the decoded MAE says whether the latent
        still carries the terrain. A shrinking latent looks healthy under the
        first and fails the second.
        """
        scan = sample["scan1" if next_state else "scan"][:, index].float()
        hit = sample["hit1" if next_state else "hit"][:, index]
        predicted_scan, hit_logit = self.perception.decode(latent)
        denominator = mask.sum().clamp_min(1.0)
        observed = hit & mask[:, None].bool()
        height_error = torch.where(observed, predicted_scan - scan, 0.0)
        # Losses are batch sums here; update_world_model divides once by B.
        sse = (height_error.square().sum(-1) * mask).sum()
        hit_count = (hit * mask[:, None]).sum().clamp_min(1.0)
        mae = (height_error.abs() * mask[:, None]).sum() / hit_count
        bce = (
            nn.functional.binary_cross_entropy_with_logits(
                hit_logit, hit.to(scan.dtype), reduction="none"
            ).mean(-1)
            * mask
        ).sum() / denominator
        return {
            "Perception/predicted_scan_mae_m": mae.detach(),
            "Perception/predicted_hit_bce": bce.detach(),
            "Perception/predicted_sse": sse,
        }

    def update_world_model(self) -> dict[str, torch.Tensor]:
        """Run one of the ten independently sampled world-model Adam updates."""
        self._check_physics()
        with self.profiler.record("model"):
            sample = self.buffer.sample_windows(
                self.cfg.wm_batch,
                self.cfg.wm_window,
                **(
                    {
                        "motion": self.motion,
                        "bin_probabilities": self.sampler.wm_probabilities,
                    }
                    if self.sampler is not None
                    else {}
                ),
            )
            if self.sampler is not None:
                # Replay caps the actual rank-local proposal. Importance
                # weights retain the global bin objective under DDP averaging.
                sample = {
                    **sample,
                    "mask": sample["mask"] * sample["sampling_weight"][:, None],
                }
            rollout = self._world_model_rollout(sample, self.cfg.wm_window)
            losses = rollout["losses"]
            normalized_errors = rollout["normalized_errors"]
            physical_errors = rollout["physical_errors"]
            batch = self.cfg.wm_batch
            first_mask = sample["mask"][:, 0]
            last_mask = sample["mask"][:, -1]
            # Eq. 1 is feature-SUM across the window, MEAN over sampled windows.
            # The replay window length is not part of the loss denominator.
            loss = torch.stack(losses).sum() / batch
            predicted = self._wm_modules()
            self.opt_wm.zero_grad(set_to_none=True)
            cap_ratio = (
                self.cfg.perception_decoder_physics_ratio if self.recurrent else None
            )
            cap_metrics = {}
            if cap_ratio is not None:
                decoders = {
                    "height": self.perception.height_decoder,
                    "depth": self.perception.depth_decoder,
                }
                # Directly compute the smaller reconstruction contribution;
                # total-minus-physics would erase it when physics dominates.
                reconstruction_gradients = torch.autograd.grad(
                    rollout["reconstruction_loss"] / batch,
                    tuple(
                        p for module in decoders.values() for p in module.parameters()
                    ),
                    retain_graph=True,
                    allow_unused=True,
                )
            loss.backward()
            self._average_gradients(predicted)
            if cap_ratio is not None:
                cap_metrics = cap_decoder_physics_gradients(
                    decoders,
                    reconstruction_gradients,
                    cap_ratio,
                    distributed=self.distributed,
                )
            grad_norm = self._clip_gradient_norm("world model update", predicted)
            self.opt_wm.step()
            self.optimizer_steps += 1
            self.windows += batch
            self.global_windows += batch * self.world_size
            rmse_1 = torch.sqrt(physical_errors[0] / first_mask.sum().clamp_min(1))
            rmse_last = torch.sqrt(physical_errors[-1] / last_mask.sum().clamp_min(1))
            normalized_1 = torch.sqrt(
                normalized_errors[0] / first_mask.sum().clamp_min(1)
            )
            normalized_last = torch.sqrt(
                normalized_errors[-1] / last_mask.sum().clamp_min(1)
            )
            return {
                "Model/loss": loss.detach(),
                "Model/physical_rmse_1": rmse_1.detach(),
                f"Model/physical_rmse_{self.cfg.wm_window}": rmse_last.detach(),
                "Model/running_std_normalized_rmse_1": normalized_1.detach(),
                f"Model/running_std_normalized_rmse_{self.cfg.wm_window}": normalized_last.detach(),
                "Gradient/world_model": grad_norm.detach(),
                "Model/mean_running_std_normalized_window_error": (
                    torch.stack(normalized_errors).sum()
                    / sample["mask"].sum().clamp_min(1)
                ).detach(),
                "Model/mean_physical_mse": (
                    torch.stack(physical_errors).sum()
                    / sample["mask"].sum().clamp_min(1)
                ).detach(),
                **rollout["scan_metrics"],
                **cap_metrics,
                **(
                    {
                        "Perception/latent_normalized_rmse": torch.sqrt(
                            torch.stack(rollout["latent_errors"]).sum()
                            / sample["mask"].sum().clamp_min(1)
                        ).detach()
                    }
                    if rollout["latent_errors"]
                    else {}
                ),
            }

    def world_model_diagnostics(
        self, horizons: tuple[int, ...] = (1, 8, 32)
    ) -> dict[str, torch.Tensor]:
        available = int(getattr(self.buffer, "size", self.cfg.buffer_steps))
        wanted = min(max(horizons), available)
        if wanted < min(horizons):
            return {}
        sample = self.buffer.sample_windows(self.cfg.wm_batch, wanted)
        metrics: dict[str, torch.Tensor] = {}
        if self.recurrent:
            with torch.no_grad():
                rollout = self._clean_world_model_rollout(sample, wanted)
            for horizon in horizons:
                if horizon <= wanted:
                    count = sample["mask"][:, horizon - 1].sum().clamp_min(1)
                    metrics[f"Model/physical_rmse_{horizon}"] = (
                        rollout["physical_errors"][horizon - 1] / count
                    ).sqrt()
                    metrics[f"Model/running_std_normalized_rmse_{horizon}"] = (
                        rollout["normalized_errors"][horizon - 1] / count
                    ).sqrt()
            metrics.update(
                {
                    f"{name}_horizon_{wanted}": value
                    for name, value in rollout["scan_metrics"].items()
                    if name.startswith("Perception/")
                }
            )
            return metrics
        std = torch.maximum(self._norm_std(self.wm.out_norm), self._floor)
        with torch.no_grad():
            state = self._as_state(sample["s"][:, 0])
            latent = self._latent_at(sample, 0) if self.perception is not None else None
            for step in range(wanted):
                if self.perception is None:
                    state = self.wm.step(
                        state,
                        sample["a"][:, step],
                        sample["a_prev"][:, step],
                        self.normalize_xi(sample["xi"][:, step]),
                    )
                else:
                    state, latent = self.wm.step(
                        state,
                        sample["a"][:, step],
                        sample["a_prev"][:, step],
                        self.normalize_xi(sample["xi"][:, step]),
                        latent,
                    )
                target = self._as_state(sample["s1"][:, step])
                error = self.wm.state_error(state, target)
                normalized = (error[..., :WM_OUT] / std[..., :WM_OUT]).square()
                mask = sample["mask"][:, step]
                denominator = mask.sum().clamp_min(1)
                for horizon in horizons:
                    if step + 1 == horizon:
                        metrics[f"Model/physical_rmse_{horizon}"] = torch.sqrt(
                            (error[..., :WM_OUT].square().mean(-1) * mask).sum()
                            / denominator
                        )
                        metrics[f"Model/running_std_normalized_rmse_{horizon}"] = (
                            torch.sqrt((normalized.mean(-1) * mask).sum() / denominator)
                        )
                if step + 1 in horizons:
                    groups = {
                        "root_pos": slice(0, 3),
                        "root_rot": slice(3, 9),
                        "root_linvel": slice(9, 12),
                        "root_angvel": slice(12, 15),
                        "joint_pos": slice(15, 44),
                        "joint_vel": slice(44, 73),
                    }
                    for name, group in groups.items():
                        metrics[f"Model/physical_{name}_rmse_{step + 1}"] = torch.sqrt(
                            (error[..., group].square().mean(-1) * mask).sum()
                            / denominator
                        )
                        metrics[
                            f"Model/running_std_normalized_{name}_rmse_{step + 1}"
                        ] = torch.sqrt(
                            (normalized[..., group].mean(-1) * mask).sum() / denominator
                        )
        return metrics
