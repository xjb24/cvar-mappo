import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from onpolicy.utils.util import get_gard_norm, huber_loss, mse_loss
from onpolicy.utils.valuenorm import ValueNorm
from onpolicy.algorithms.utils.util import check

class R_MAPPO():
    """
    Trainer class for MAPPO to update policies.
    :param args: (argparse.Namespace) arguments containing relevant model, policy, and env information.
    :param policy: (R_MAPPO_Policy) policy to update.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self,
                 args,
                 policy,
                 device=torch.device("cpu")):

        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.policy = policy

        self.clip_param = args.clip_param
        self.ppo_epoch = args.ppo_epoch
        self.num_mini_batch = args.num_mini_batch
        self.data_chunk_length = args.data_chunk_length
        self.value_loss_coef = args.value_loss_coef
        self.entropy_coef = args.entropy_coef
        self.max_grad_norm = args.max_grad_norm       
        self.huber_delta = args.huber_delta

        self._use_recurrent_policy = args.use_recurrent_policy
        self._use_naive_recurrent = args.use_naive_recurrent_policy
        self._use_max_grad_norm = args.use_max_grad_norm
        self._use_clipped_value_loss = args.use_clipped_value_loss
        self._use_huber_loss = args.use_huber_loss
        self._use_popart = args.use_popart
        self._use_valuenorm = args.use_valuenorm
        self._use_value_active_masks = args.use_value_active_masks
        self._use_policy_active_masks = args.use_policy_active_masks
        self.cvar_alpha = getattr(args, "cvar_alpha", 0.95)
        self.cvar_beta = getattr(args, "cvar_beta", 50.0)
        self.num_users = getattr(args, "num_users", None)
        
        assert (self._use_popart and self._use_valuenorm) == False, ("self._use_popart and self._use_valuenorm can not be set True simultaneously")
        
        if self._use_popart:
            self.value_normalizer = self.policy.critic.v_out
        elif self._use_valuenorm:
            self.value_normalizer = ValueNorm(1, device=self.device)
        else:
            self.value_normalizer = None

    def cal_value_loss(self, values, value_preds_batch, return_batch, active_masks_batch):
        """
        Calculate value function loss.
        :param values: (torch.Tensor) value function predictions.
        :param value_preds_batch: (torch.Tensor) "old" value  predictions from data batch (used for value clip loss)
        :param return_batch: (torch.Tensor) reward to go returns.
        :param active_masks_batch: (torch.Tensor) denotes if agent is active or dead at a given timesep.

        :return value_loss: (torch.Tensor) value function loss.
        """
        value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param,
                                                                                        self.clip_param)
        if self._use_popart or self._use_valuenorm:
            self.value_normalizer.update(return_batch)
            error_clipped = self.value_normalizer.normalize(return_batch) - value_pred_clipped
            error_original = self.value_normalizer.normalize(return_batch) - values
        else:
            error_clipped = return_batch - value_pred_clipped
            error_original = return_batch - values

        if self._use_huber_loss:
            value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
            value_loss_original = huber_loss(error_original, self.huber_delta)
        else:
            value_loss_clipped = mse_loss(error_clipped)
            value_loss_original = mse_loss(error_original)

        if self._use_clipped_value_loss:
            value_loss = torch.max(value_loss_original, value_loss_clipped)
        else:
            value_loss = value_loss_original

        if self._use_value_active_masks:
            value_loss = (value_loss * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            value_loss = value_loss.mean()

        return value_loss

    def ppo_update(self, sample, update_actor=True):
        """
        Update actor and critic networks.
        :param sample: (Tuple) contains data batch with which to update networks.
        :update_actor: (bool) whether to update actor network.

        :return value_loss: (torch.Tensor) value function loss.
        :return critic_grad_norm: (torch.Tensor) gradient norm from critic up9date.
        ;return policy_loss: (torch.Tensor) actor(policy) loss value.
        :return dist_entropy: (torch.Tensor) action entropies.
        :return actor_grad_norm: (torch.Tensor) gradient norm from actor update.
        :return imp_weights: (torch.Tensor) importance sampling weights.
        """
        user_rates_batch = None
        if len(sample) == 12:
            share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, \
            value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
            adv_targ, available_actions_batch = sample
        elif len(sample) == 13:
            share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, \
            value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
            adv_targ, available_actions_batch, user_rates_batch = sample
        else:
            raise ValueError(f"unexpected sample length {len(sample)}")

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        value_preds_batch = check(value_preds_batch).to(**self.tpdv)
        return_batch = check(return_batch).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)
        if user_rates_batch is not None:
            user_rates_batch = check(user_rates_batch).to(**self.tpdv)

        # Reshape to do in a single forward pass for all steps
        values, action_log_probs, dist_entropy = self.policy.evaluate_actions(share_obs_batch,
                                                                              obs_batch, 
                                                                              rnn_states_batch, 
                                                                              rnn_states_critic_batch, 
                                                                              actions_batch, 
                                                                              masks_batch, 
                                                                              available_actions_batch,
                                                                              active_masks_batch)
        # actor update
        imp_weights = torch.exp(torch.clamp(action_log_probs - old_action_log_probs_batch, max=5.0))
        print('action_log_probs',action_log_probs[0])
        print('old_action_log_probs_batch',old_action_log_probs_batch[0])
        # imp_weights = torch.exp(action_log_probs - old_action_log_probs_batch)

        lambda_param = F.softplus(self.policy.cvar_lambda_raw)
        eta_param = self.policy.cvar_eta
        user_rates_sorted, _ = torch.sort(user_rates_batch, dim=1)  # shape: (7680, 20) → (7680, 20) 排序后
        user_rates_aggregated = user_rates_sorted.mean(dim=0)  # shape: (20,)
        # print('user_rates_aggregated',user_rates_aggregated[0])
        cvar_constraint = (F.relu(eta_param - user_rates_aggregated).mean() / max(1 - self.cvar_alpha, 1e-6)) - eta_param + self.cvar_beta

        if cvar_constraint.item() > 0:
            # 计算 CVaR 约束并融入优势函数

            
            # 对每行计算 CVaR 惩罚（形状：[7680, 20] → [7680, 1]）
            # CVaR = E[relu(η - user_rate)] / (1 - α)
            # min_rate = torch.min(user_rates_batch, dim=-1, keepdim=True)[0]
            # tail = F.relu(self.cvar_beta - min_rate)  # [7680, 20]
            tail = (F.relu(eta_param - user_rates_batch).mean(dim=1, keepdim=True) / max(1 - self.cvar_alpha, 1e-6)) - eta_param.detach() + self.cvar_beta

            # cvar_per_sample = tail.mean(dim=1, keepdim=True) / max(1 - self.cvar_alpha, 1e-6)  # [7680, 1]

            # 惩罚项：λ * (CVaR - η + β)
            # penalty_per_sample = lambda_param * (cvar_per_sample - eta_param + self.cvar_beta)  # [7680, 1]
            penalty_per_sample = lambda_param.detach() * tail

            # 中心化：减去均值，使惩罚项均值为 0
            # centralized_penalty = penalty_per_sample - penalty_per_sample.mean()  # [7680, 1]

            # Clip 限制参数（可根据需要调整）
            delta = 1  # 惩罚强度系数
            k = 1.0      # Clip 范围系数（相对于 adv_targ 的比例）
        
            # 应用 Clip
            # clipped_penalty = torch.clamp(delta * penalty_per_sample, -k * adv_targ.abs(), k * adv_targ.abs())
            # 优势函数塑形：减去 CVaR 惩罚
            penalty_per_sample = penalty_per_sample / 5.0
            # print('adv_targ',adv_targ)
            # print('penalty_per_sample',penalty_per_sample)
            adv_targ = adv_targ - penalty_per_sample  # [7680, 1]

            
            # 优势函数标准化
            adv_targ = (adv_targ - adv_targ.mean()) / (adv_targ.std() + 1e-5)
        
        surr1 = imp_weights * adv_targ
        surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ

        if self._use_policy_active_masks:
            policy_action_loss = (-torch.sum(torch.min(surr1, surr2),
                                             dim=-1,
                                             keepdim=True) * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            policy_action_loss = -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True).mean()

        # η/λ 更新用的全局约束
        # 对每一行升序排序，然后每列求均值，得到 20 维数组
        user_rates_sorted, _ = torch.sort(user_rates_batch, dim=1)  # shape: (7680, 20) → (7680, 20) 排序后
        user_rates_aggregated = user_rates_sorted.mean(dim=0)  # shape: (20,)
        # print('user_rates_aggregated',user_rates_aggregated[0])
        cvar_constraint = (F.relu(eta_param - user_rates_aggregated).mean() / max(1 - self.cvar_alpha, 1e-6)) - eta_param + self.cvar_beta

        # PPO 损失（纯 PPO，不再加 cvar_penalty）
        policy_loss = policy_action_loss

        self.policy.actor_optimizer.zero_grad()

        if update_actor:
            # (policy_loss - dist_entropy * self.entropy_coef).backward()
            (policy_loss).backward()

        if self._use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.max_grad_norm)
        else:
            actor_grad_norm = get_gard_norm(self.policy.actor.parameters())

        self.policy.actor_optimizer.step()

        # 更新 η：最小化 CVaR 约束
        self.policy.cvar_eta_optimizer.zero_grad()
        eta_loss = cvar_constraint
        eta_loss.backward(retain_graph=True)
        self.policy.cvar_eta_optimizer.step()

        # lambda_constraint = 
        # 更新 λ：最大化约束违反（对偶上升）
        self.policy.lambda_optimizer.zero_grad()
        lambda_loss = -F.softplus(self.policy.cvar_lambda_raw) * cvar_constraint.detach()
        lambda_loss.backward()
        self.policy.lambda_optimizer.step()

        # critic update
        value_loss = self.cal_value_loss(values, value_preds_batch, return_batch, active_masks_batch)

        self.policy.critic_optimizer.zero_grad()

        (value_loss * self.value_loss_coef).backward()

        if self._use_max_grad_norm:
            critic_grad_norm = nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.max_grad_norm)
        else:
            critic_grad_norm = get_gard_norm(self.policy.critic.parameters())

        self.policy.critic_optimizer.step()

        # alpha update
        log_alpha_loss = -torch.mean(self.policy.actor.log_alpha * ( - dist_entropy + self.policy.actor.target_entropy))

        self.policy.alpha_optimizer.zero_grad()

        log_alpha_loss.backward()

        if self._use_max_grad_norm:
            alpha_grad_norm = nn.utils.clip_grad_norm_([self.policy.log_alpha], self.max_grad_norm)
        else:
            alpha_grad_norm = get_gard_norm([self.policy.log_alpha])
        
        self.policy.alpha_optimizer.step()

        # 返回 CVaR 相关指标用于 wandb 记录
        lambda_value = F.softplus(self.policy.cvar_lambda_raw).item()
        eta_value = self.policy.cvar_eta.item()
        cvar_term_value = cvar_constraint.item()
        return value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights, \
               cvar_term_value, lambda_value, eta_value

    def train(self, buffer, update_actor=True):
        """
        Perform a training update using minibatch GD.
        :param buffer: (SharedReplayBuffer) buffer containing training data.
        :param update_actor: (bool) whether to update actor network.

        :return train_info: (dict) contains information regarding training update (e.g. loss, grad norms, etc).
        """
        if self._use_popart or self._use_valuenorm:
            advantages = buffer.returns[:-1] - self.value_normalizer.denormalize(buffer.value_preds[:-1])
        else:
            advantages = buffer.returns[:-1] - buffer.value_preds[:-1]
        advantages_copy = advantages.copy()
        advantages_copy[buffer.active_masks[:-1] == 0.0] = np.nan
        mean_advantages = np.nanmean(advantages_copy)
        std_advantages = np.nanstd(advantages_copy)
        advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)
        

        train_info = {}

        train_info['value_loss'] = 0
        train_info['policy_loss'] = 0
        train_info['dist_entropy'] = 0
        train_info['actor_grad_norm'] = 0
        train_info['critic_grad_norm'] = 0
        train_info['ratio'] = 0
        train_info['cvar_constraint'] = 0
        train_info['cvar_lambda'] = 0
        train_info['cvar_eta'] = 0

        for _ in range(self.ppo_epoch):
            if self._use_recurrent_policy:
                data_generator = buffer.recurrent_generator(advantages, self.num_mini_batch, self.data_chunk_length)
            elif self._use_naive_recurrent:
                data_generator = buffer.naive_recurrent_generator(advantages, self.num_mini_batch)
            else:
                data_generator = buffer.feed_forward_generator(advantages, self.num_mini_batch)

            for sample in data_generator:
                results = self.ppo_update(sample, update_actor)
                value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights, \
                    cvar_term, lambda_value, eta_value = results
                train_info['value_loss'] += value_loss.item()
                train_info['policy_loss'] += policy_loss.item()
                train_info['dist_entropy'] += dist_entropy.item()
                train_info['actor_grad_norm'] += actor_grad_norm
                train_info['critic_grad_norm'] += critic_grad_norm
                train_info['ratio'] += imp_weights.mean()
                train_info['cvar_constraint'] += cvar_term
                train_info['cvar_lambda'] += lambda_value
                train_info['cvar_eta'] += eta_value

        num_updates = self.ppo_epoch * self.num_mini_batch

        for k in train_info.keys():
            train_info[k] /= num_updates
 
        return train_info

    def prep_training(self):
        self.policy.actor.train()
        self.policy.critic.train()

    def prep_rollout(self):
        self.policy.actor.eval()
        self.policy.critic.eval()
