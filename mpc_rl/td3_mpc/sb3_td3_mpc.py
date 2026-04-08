"""
SB3 (PyTorch) TD3 with MPC trajectory injection support.

Used for quadruped environments that require Dict observations and
asymmetric actor-critic policies (not supported by SBX/JAX TD3_MPC).
"""

import numpy as np
import torch as th
from torch.nn import functional as F
from stable_baselines3 import TD3
from stable_baselines3.common.utils import polyak_update


class SB3_TD3_MPC(TD3):
    """
    TD3 with MPC-augmented replay buffer (PyTorch / SB3).

    Subclasses SB3's TD3 to check the replay buffer's MPC percentage
    before each training update and inject more MPC trajectories when
    the percentage falls below the target.

    Attributes set externally by the training script:
        target_mpc_percentage (float): Target MPC data percentage (0-100).
        mpc_inject_callback: PercentMPCInjectCallback with _inject_mpc_trajectories().
    """

    def _excluded_save_params(self):
        return super()._excluded_save_params() + [
            "mpc_inject_callback",
            "target_mpc_percentage",
        ]
    
    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        """
        TD3-MPC training step.

        This keeps the standard SB3 TD3 update logic, but adds:
        1. MPC-injection percentage enforcement before training
        2. extra TensorBoard logging for:
        - train/q_mean
        - train/target_q_mean
        - train/td_error_abs_mean
        """

        # ------------------------------------------------------------------
        # MPC-injection check before gradient updates
        # ------------------------------------------------------------------
        if hasattr(self.replay_buffer, "get_mpc_percentage"):
            actual_mpc_pct = self.replay_buffer.get_mpc_percentage()
            self.logger.record("replay_buffer/mpc_percentage_actual", actual_mpc_pct)

            if hasattr(self, "target_mpc_percentage") and hasattr(self, "mpc_inject_callback"):
                buffer_full = self.replay_buffer.size() >= self.replay_buffer.buffer_size
                target_reached = (
                    actual_mpc_pct >= self.target_mpc_percentage
                    or (
                        self.target_mpc_percentage >= 100
                        and actual_mpc_pct >= 99.0
                        and buffer_full
                    )
                )

                if not target_reached:
                    if self.verbose > 1:
                        print(
                            f"\n[Train Update {self._n_updates}] MPC percentage low: "
                            f"{actual_mpc_pct:.2f}% < {self.target_mpc_percentage}%"
                        )
                        print("Injecting MPC trajectories before sampling...")

                    self.mpc_inject_callback._inject_mpc_trajectories()

                    # Re-log after injection so TensorBoard shows the new value
                    new_mpc_pct = self.replay_buffer.get_mpc_percentage()
                    self.logger.record("replay_buffer/mpc_percentage_actual", new_mpc_pct)

        # ------------------------------------------------------------------
        # Standard SB3 TD3 train() logic
        # ------------------------------------------------------------------
        self.policy.set_training_mode(True)

        self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])

        actor_losses = []
        critic_losses = []

        # Extra diagnostics
        q_means = []
        target_q_means = []
        td_error_abs_means = []

        for _ in range(gradient_steps):
            self._n_updates += 1

            replay_data = self.replay_buffer.sample(
                batch_size,
                env=self._vec_normalize_env,
            )

            discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma

            with th.no_grad():
                noise = replay_data.actions.clone().data.normal_(0, self.target_policy_noise)
                noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)

                next_actions = (
                    self.actor_target(replay_data.next_observations) + noise
                ).clamp(-1, 1)

                next_q_values = th.cat(
                    self.critic_target(replay_data.next_observations, next_actions),
                    dim=1,
                )
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)

                # Form the Bellman target
                target_q_values = replay_data.rewards + (
                    1 - replay_data.dones
                ) * discounts * next_q_values

            current_q_values = self.critic(
                replay_data.observations,
                replay_data.actions,
            )

            critic_loss = sum(
                F.mse_loss(current_q, target_q_values)
                for current_q in current_q_values
            )
            assert isinstance(critic_loss, th.Tensor)
            critic_losses.append(critic_loss.item())

            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # --------------------------------------------------------------
            # Extra logging diagnostics
            # --------------------------------------------------------------
            with th.no_grad():
                q_cat = th.cat(current_q_values, dim=1)
                td_abs = th.cat(
                    [
                        (current_q - target_q_values).abs()
                        for current_q in current_q_values
                    ],
                    dim=1,
                )

                q_means.append(q_cat.mean().item())
                target_q_means.append(target_q_values.mean().item())
                td_error_abs_means.append(td_abs.mean().item())

            if self._n_updates % self.policy_delay == 0:
                actor_loss = -self.critic.q1_forward(
                    replay_data.observations,
                    self.actor(replay_data.observations),
                ).mean()
                actor_losses.append(actor_loss.item())

                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                self.actor.optimizer.step()

                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.actor.parameters(), self.actor_target.parameters(), self.tau)

                # Copy running stats, same as official SB3 implementation
                polyak_update(self.critic_batch_norm_stats, self.critic_batch_norm_stats_target, 1.0)
                polyak_update(self.actor_batch_norm_stats, self.actor_batch_norm_stats_target, 1.0)

        # ------------------------------------------------------------------
        # Standard SB3 logging + new custom logging
        # ------------------------------------------------------------------
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")

        if len(actor_losses) > 0:
            self.logger.record("train/actor_loss", np.mean(actor_losses))

        self.logger.record("train/critic_loss", np.mean(critic_losses))

        if len(q_means) > 0:
            self.logger.record("train/q_mean", np.mean(q_means))

        if len(target_q_means) > 0:
            self.logger.record("train/target_q_mean", np.mean(target_q_means))

        if len(td_error_abs_means) > 0:
            self.logger.record("train/td_error_abs_mean", np.mean(td_error_abs_means))
