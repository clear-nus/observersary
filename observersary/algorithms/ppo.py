import gym.spaces
import numpy as np
import stable_baselines3 as sb3
import stable_baselines3.common.utils
import torch


class PPO(sb3.PPO):

    def compute_loss(self, policy_loss, entropy_loss, value_loss):
        """Compute PPO training loss.

        Args:
            policy_loss: Policy loss.
            entropy_loss: Entropy loss.
            value_loss: Value loss.
        """
        return policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

    def train(self):
        """
        Update policy using the currently gathered rollout buffer.
        This code is copied and modified from sb3 PPO train method.
        """
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []

        continue_training = True

        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, gym.spaces.Discrete):
                    actions = rollout_data.actions.long().flatten()

                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()
                advantages = rollout_data.advantages
                if self.normalize_advantage:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                ratio = torch.exp(log_prob - rollout_data.old_log_prob)

                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                pg_losses.append(policy_loss.item())
                clip_fraction = torch.mean((torch.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + torch.clamp(values - rollout_data.old_values, -clip_range_vf, clip_range_vf)
                value_loss = torch.nn.functional.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                if entropy is None:
                    entropy_loss = -torch.mean(-log_prob)
                else:
                    entropy_loss = -torch.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                loss = self.compute_loss(policy_loss, entropy_loss, value_loss)

                with torch.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                self.policy.optimizer.zero_grad()
                loss.backward()
                actor_parameters = self.get_parameters()["policy"]["mlp_extractor.policy_net.0.weight"]
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            if not continue_training:
                break

        self._n_updates += self.n_epochs
        explained_var = stable_baselines3.common.utils.explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", torch.exp(self.policy.log_std).mean().item())
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

        with torch.no_grad():
            actor_parameters = self.policy.get_parameter("mlp_extractor.policy_net.0.weight")
            critic_parameters = self.policy.get_parameter("mlp_extractor.value_net.0.weight")
            for i in range(actor_parameters.shape[1]):
                self.logger.record(f"norms/actornorm_{i:02d}", torch.norm(actor_parameters[:, i]).item())
                self.logger.record(f"norms/criticnorm_{i:02d}", torch.norm(critic_parameters[:, i]).item())


class PPOWithInputL1Regularizer(PPO):

    def compute_loss(self, policy_loss, entropy_loss, value_loss):
        """Compute PPO training loss with L1 regularization on input weights.

        Args:
            policy_loss: Policy loss.
            entropy_loss: Entropy loss.
            value_loss: Value loss.
        """
        base_loss = super().compute_loss(policy_loss, entropy_loss, value_loss)
        actor_parameters = self.policy.get_parameter("mlp_extractor.policy_net.0.weight")
        critic_parameters = self.policy.get_parameter("mlp_extractor.value_net.0.weight")
        actor_regularizer = torch.linalg.norm(actor_parameters, 1)
        critic_regularizer = torch.linalg.norm(critic_parameters, 1)
        return base_loss + 0.001 * actor_regularizer + 0.002 * critic_regularizer
