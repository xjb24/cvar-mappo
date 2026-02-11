import torch
import torch.nn as nn
from onpolicy.algorithms.utils.util import init, check
from onpolicy.algorithms.utils.cnn import CNNBase
from onpolicy.algorithms.utils.mlp import MLPBase
from onpolicy.algorithms.utils.rnn import RNNLayer
from onpolicy.algorithms.utils.act import ACTLayer
from onpolicy.algorithms.utils.popart import PopArt
from onpolicy.algorithms.utils.DACERPolicyNet import DACERPolicyNet
from onpolicy.algorithms.utils.GaussianDiffusion import GaussianDiffusion
from onpolicy.utils.util import get_shape_from_obs_space


class R_Actor(nn.Module):
    """
    Actor network class for MAPPO. Outputs actions given observations.
    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param obs_space: (gym.Space) observation space.
    :param action_space: (gym.Space) action space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self, args, obs_space, action_space, device=torch.device("cpu"), log_alpha=None):
        super(R_Actor, self).__init__()
        self.hidden_size = args.hidden_size
        self.device = device
        self.act_dim = action_space.shape[0]
        self.target_entropy = - self.act_dim * 0.9

        self._gain = args.gain
        self._use_orthogonal = args.use_orthogonal
        self._use_policy_active_masks = args.use_policy_active_masks
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self.tpdv = dict(dtype=torch.float32, device=device)

        obs_shape = get_shape_from_obs_space(obs_space)
        base = CNNBase if len(obs_shape) == 3 else MLPBase
        self.base = base(args, obs_shape)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)

        self.DACERPolicyNet = DACERPolicyNet(args, self.act_dim, device=device) # 用于生成noise 需要结合diffusion算法来用
        self.diffusion_step_T = 5
        self.sample_N = 50
        self.diffusion = GaussianDiffusion(num_timesteps=self.diffusion_step_T, device=device)
        self.log_alpha = log_alpha
        self.entropy = 0
        self.evaluate_action_number = 0

        self.to(device)
        self.algo = args.algorithm_name

    def forward(self, obs, rnn_states, masks, available_actions=None, deterministic=False):
        """
        Compute actions from the given inputs.
        :param obs: (np.ndarray / torch.Tensor) observation inputs into network.
        :param rnn_states: (np.ndarray / torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (np.ndarray / torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.
        :param available_actions: (np.ndarray / torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param deterministic: (bool) whether to sample from action distribution or return the mode.

        :return actions: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of taken actions.
        :return rnn_states: (torch.Tensor) updated RNN hidden states.
        """
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        actor_features = self.base(obs)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        # actions, action_log_probs = self.act(actor_features, available_actions, deterministic)
        actions = self.get_actions(actor_features, self.log_alpha)
        action_log_probs = self.get_logp(actions, actor_features)
        # 修改动作范围用的
        actions_tanh = torch.tanh(actions)
        actions = (actions_tanh + 1.0) * 0.5  # scale to [0, 1]
        return actions, action_log_probs, rnn_states

    def evaluate_actions(self, obs, rnn_states, action, masks, available_actions=None, active_masks=None):
        """
        Compute log probability and entropy of given actions.
        :param obs: (torch.Tensor) observation inputs into network.
        :param action: (torch.Tensor) actions whose entropy and log probability to evaluate.
        :param rnn_states: (torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.
        :param available_actions: (torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param active_masks: (torch.Tensor) denotes whether an agent is active or dead.

        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)

        actor_features = self.base(obs)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        if self.algo == "hatrpo":
            action_log_probs, dist_entropy ,action_mu, action_std, all_probs= self.act.evaluate_actions_trpo(actor_features,
                                                                    action, available_actions,
                                                                    active_masks=
                                                                    active_masks if self._use_policy_active_masks
                                                                    else None)

            return action_log_probs, dist_entropy, action_mu, action_std, all_probs
        else:
            action_log_probs = self.get_logp(action, actor_features)
            if self.evaluate_action_number % 1000 == 0:
                actions = self.get_sample_actions(actor_features)  # (batch, sample, act_dim)
                self.entropy = self.estimate_entropy(actions=actions)
                dist_entropy = self.entropy
            else:
                dist_entropy = self.entropy
            self.evaluate_action_number += 1
            # print(self.evaluate_action_number)

        return action_log_probs, dist_entropy

    def get_actions(self, obs, log_alpha):
        def model_fn(t, x):
            return self.DACERPolicyNet(obs, x, t)
        actions = self.diffusion.p_sample(model_fn, (*obs.shape[:-1], self.act_dim), device=obs.device)
        noise = torch.randn_like(actions)
        actions = actions + noise * torch.exp(log_alpha) * 0.1
        return actions # (batch, act_dim)
    
    def get_sample_actions(self, obs, n_samples=200):
        actions = []
        for _ in range(n_samples):
            action = self.get_actions(obs, self.log_alpha)  # 输出 (batch_size, act_dim)
            actions.append(action.unsqueeze(1))  # 添加采样维度
        actions = torch.cat(actions, dim=1)  # (batch_size, n_samples, act_dim)
        return actions
    
    # action是一个和一批应该都能处理
    def get_logp(self, actions, obs):
        def model_fn(t, x):
            return self.DACERPolicyNet(obs, x, t)
        shape = actions.shape
        batch_size = shape[0]
        action_dim = shape[1]
        noise_pred_matrix = torch.zeros((self.diffusion_step_T, self.sample_N, batch_size, action_dim), device=self.device)
        noise = torch.zeros((self.diffusion_step_T, self.sample_N, batch_size, action_dim), device=self.device)
        for t in range(self.diffusion_step_T):
            t = torch.tensor(t, dtype=torch.long)
            for n in range(self.sample_N):
                x = torch.randn(shape, device=self.device)
                a_t = self.diffusion.q_sample(t=t, x_start=actions, noise=x)
                noise_pred = model_fn(t, a_t)
                noise[t, n] = x
                noise_pred_matrix[t, n] = noise_pred
        
        l2_norm_pow2 = torch.norm(noise - noise_pred_matrix, p=2, dim=-1).pow(2) # shape:(20,50,10) --> (diffusion_T, sample_N, agent_num)
        noise_pred_error_estimation = torch.mean(l2_norm_pow2, dim=1) # shape:(20,10) --> (diffusion_T, agent_num)

        c = (-self.act_dim / 2) * torch.log(2 * torch.pi * torch.exp(torch.tensor(1.0)))
        alpha_cumprod = self.diffusion.B.alphas_cumprod
        alpha_cumprod_prev = self.diffusion.B.alphas_cumprod_prev
        d_alpha = (alpha_cumprod * self.act_dim).unsqueeze(dim=-1).repeat(1, batch_size).to(self.device)
        # multiplier1 = d_alpha - noise_pred_error_estimation # -->shape:(diffusion_T, agent_num) = (20, 10)
        weight = (alpha_cumprod_prev - alpha_cumprod) / (2 * alpha_cumprod * (1 - alpha_cumprod))
        weight = weight.unsqueeze(dim=-1).repeat(1, batch_size).to(self.device) # --> shape:(diffusion_T, agent_num) = (20, 10)
        sum_dim0 = torch.sum(weight * (d_alpha - noise_pred_error_estimation), dim=0)
        log_p = c + sum_dim0
        log_p = log_p.unsqueeze(1)
        print('log_p',log_p,log_p.shape)
        # log_p = log_p.unsqueeze(dim=-1).repeat(1,1,2)
        return log_p

    def estimate_entropy(self, actions, num_components=3):  # (batch, sample, dim)
        import numpy as np
        from sklearn.mixture import GaussianMixture
        actions = actions.detach().cpu().numpy()
        total_entropy = []
        for action in actions:
            gmm = GaussianMixture(n_components=num_components, covariance_type='full', random_state=42)
            gmm.fit(action)
            weights = gmm.weights_
            entropies = []
            for i in range(gmm.n_components):
                cov_matrix = gmm.covariances_[i]
                d = cov_matrix.shape[0]
                entropy = 0.5 * d * (1 + np.log(2 * np.pi)) + 0.5 * np.linalg.slogdet(cov_matrix)[1]
                entropies.append(entropy)
            entropy = -np.sum(weights * np.log(weights)) + np.sum(weights * np.array(entropies))
            total_entropy.append(entropy)
        final_entropy = sum(total_entropy) / len(total_entropy)
        return final_entropy
    

class R_Critic(nn.Module):
    """
    Critic network class for MAPPO. Outputs value function predictions given centralized input (MAPPO) or
                            local observations (IPPO).
    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param cent_obs_space: (gym.Space) (centralized) observation space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self, args, cent_obs_space, device=torch.device("cpu")):
        super(R_Critic, self).__init__()
        self.hidden_size = args.hidden_size
        self._use_orthogonal = args.use_orthogonal
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self._use_popart = args.use_popart
        self.tpdv = dict(dtype=torch.float32, device=device)
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]

        cent_obs_shape = get_shape_from_obs_space(cent_obs_space)
        base = CNNBase if len(cent_obs_shape) == 3 else MLPBase
        self.base = base(args, cent_obs_shape)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        if self._use_popart:
            self.v_out = init_(PopArt(self.hidden_size, 1, device=device))
        else:
            self.v_out = init_(nn.Linear(self.hidden_size, 1))

        self.to(device)

    def forward(self, cent_obs, rnn_states, masks):
        """
        Compute actions from the given inputs.
        :param cent_obs: (np.ndarray / torch.Tensor) observation inputs into network.
        :param rnn_states: (np.ndarray / torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (np.ndarray / torch.Tensor) mask tensor denoting if RNN states should be reinitialized to zeros.

        :return values: (torch.Tensor) value function predictions.
        :return rnn_states: (torch.Tensor) updated RNN hidden states.
        """
        cent_obs = check(cent_obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        critic_features = self.base(cent_obs)
        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            critic_features, rnn_states = self.rnn(critic_features, rnn_states, masks)
        values = self.v_out(critic_features)

        return values, rnn_states
