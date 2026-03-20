"""
SB3 (PyTorch) SAC with MPC trajectory injection support.

Used for quadruped environments that require Dict observations and
asymmetric actor-critic policies (not supported by SBX/JAX SAC_MPC).
"""

from stable_baselines3 import SAC


class SB3_SAC_MPC(SAC):
    """
    SAC with MPC-augmented replay buffer (PyTorch / SB3).

    Subclasses SB3's SAC to check the replay buffer's MPC percentage
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

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        # Check MPC percentage before training (if using tagged replay buffer)
        if hasattr(self.replay_buffer, "get_mpc_percentage"):
            actual_mpc_pct = self.replay_buffer.get_mpc_percentage()
            self.logger.record("replay_buffer/mpc_percentage_actual", actual_mpc_pct)

            if hasattr(self, "target_mpc_percentage") and hasattr(self, "mpc_inject_callback"):
                buffer_full = self.replay_buffer.size() >= self.replay_buffer.buffer_size
                target_reached = (
                    actual_mpc_pct >= self.target_mpc_percentage
                    or (self.target_mpc_percentage >= 100
                        and actual_mpc_pct >= 99.0
                        and buffer_full)
                )

                if not target_reached:
                    if self.verbose > 0:
                        print(
                            f"\n[Train Update {self._n_updates}] MPC percentage low: "
                            f"{actual_mpc_pct:.2f}% < {self.target_mpc_percentage}%"
                        )
                        print("Injecting MPC trajectories before sampling...")

                    self.mpc_inject_callback._inject_mpc_trajectories()

        super().train(gradient_steps, batch_size)
