import time
import numpy as np
import torch
from onpolicy.runner.shared.base_runner import Runner
import wandb
import imageio
import torch.nn.functional as F
from scipy.special import softmax
from tqdm import trange

def _t2n(x):
    return x.detach().cpu().numpy()

class MPERunner(Runner):
    """Runner class to perform training, evaluation. and data collection for the MPEs. See parent class for details."""
    def __init__(self, config):
        super(MPERunner, self).__init__(config)

    def run(self):
        self.warmup()   

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        for episode in trange(episodes, desc="Training", leave=True):
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            min_rates_episode = []  # 记录每个时间步（跨 env 平均后的）最低用户速率
            mean_rates_episode = []  # 记录每个时间步（跨 env 平均后的）平均用户速率
            for step in range(self.episode_length):
                # Sample actions
                values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env = self.collect(step)

                # Obser reward and next obs
                # env_start = time.time()
                obs, rewards, dones, infos = self.envs.step(actions_env)
                # env_end = time.time()
                # print(f"[Perf] env.step took {(env_end - env_start):.4f}s (episode {episode})")

                # 记录当前 step 的最低用户速率和平均用户速率
                if infos:
                    step_min_rates = []
                    step_mean_rates = []
                    for env_info in infos:
                        if env_info and isinstance(env_info, list):
                            for agent_info in env_info:
                                if isinstance(agent_info, dict):
                                    if "min_user_rate_kbps" in agent_info:
                                        step_min_rates.append(agent_info["min_user_rate_kbps"])
                                    if "mean_user_rate_kbps" in agent_info:
                                        step_mean_rates.append(agent_info["mean_user_rate_kbps"])
                                    break  # 同一 env 的 agent info 相同，取一个即可
                    if step_min_rates:
                        min_rates_episode.append(float(np.mean(step_min_rates)))
                    if step_mean_rates:
                        mean_rates_episode.append(float(np.mean(step_mean_rates)))

                data = obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic

                # insert data into buffer
                self.insert(data)

            # compute return and update network
            self.compute()
            train_infos = self.train()
            if min_rates_episode:
                # 整个 episode 内"各步环境平均最小速率"的平均值
                train_infos["episode_min_user_rate_kbps"] = float(np.mean(min_rates_episode))
            if mean_rates_episode:
                # 整个 episode 内"各步环境平均速率"的平均值
                train_infos["episode_mean_user_rate_kbps"] = float(np.mean(mean_rates_episode))
            
            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads
            
            # save model (disabled to reduce I/O overhead)
            # if (episode % self.save_interval == 0 or episode == episodes - 1):
            #     self.save()

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                print("\n Scenario {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                        .format(self.all_args.scenario_name,
                                self.algorithm_name,
                                self.experiment_name,
                                episode,
                                episodes,
                                total_num_steps,
                                self.num_env_steps,
                                int(total_num_steps / (end - start))))

                env_infos = {}
                if self.env_name == "MPE":
                    for agent_id in range(self.num_agents):
                        idv_rews = []
                        for info in infos:
                            if 'individual_reward' in info[agent_id].keys():
                                idv_rews.append(info[agent_id]['individual_reward'])
                        agent_k = 'agent%i/individual_rewards' % agent_id
                        env_infos[agent_k] = idv_rews

                # train_infos["average_episode_rewards"] = np.mean(self.buffer.rewards) * self.episode_length
                train_infos["average_episode_rewards"] = np.mean(self.buffer.rewards)
                print("average episode rewards is {}".format(train_infos["average_episode_rewards"]))
                self.log_train(train_infos, total_num_steps)
                self.log_env(env_infos, total_num_steps)

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

    def warmup(self):
        # reset env
        obs = self.envs.reset()

        # replay buffer
        if self.use_centralized_V:
            share_obs = obs.reshape(self.n_rollout_threads, -1)
            share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
        else:
            share_obs = obs

        self.buffer.share_obs[0] = share_obs.copy()
        self.buffer.obs[0] = obs.copy()

    @torch.no_grad()
    def collect(self, step):
        self.trainer.prep_rollout()
        value, action, action_log_prob, rnn_states, rnn_states_critic \
            = self.trainer.policy.get_actions(np.concatenate(self.buffer.share_obs[step]),
                            np.concatenate(self.buffer.obs[step]),
                            np.concatenate(self.buffer.rnn_states[step]),
                            np.concatenate(self.buffer.rnn_states_critic[step]),
                            np.concatenate(self.buffer.masks[step]))
        # [self.envs, agents, dim]
        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads))
        rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_states_critic), self.n_rollout_threads))
        # rearrange action
        if self.envs.action_space[0].__class__.__name__ == 'MultiDiscrete':
            for i in range(self.envs.action_space[0].shape):
                uc_actions_env = np.eye(self.envs.action_space[0].high[i] + 1)[actions[:, :, i]]
                if i == 0:
                    actions_env = uc_actions_env
                else:
                    actions_env = np.concatenate((actions_env, uc_actions_env), axis=2)
        elif self.envs.action_space[0].__class__.__name__ == 'Discrete':
            actions_env = np.squeeze(np.eye(self.envs.action_space[0].n)[actions], 2)
        elif self.envs.action_space[0].__class__.__name__ == 'Box':
            # 动作维度: [下倾角, 用户打分, 子载波功率打分]
            actions_tanh = np.tanh(actions)
            action_dim = actions_tanh.shape[-1]
            num_users = getattr(self.all_args, "num_users", 20)
            num_subcarriers = action_dim - 1 - num_users
            if num_subcarriers <= 0:
                raise ValueError("Box 动作维度不足以分出子载波部分，请检查配置。")

            dt_idx = 0
            user_slice = slice(1, 1 + num_users)
            power_slice = slice(1 + num_users, 1 + num_users + num_subcarriers)

            actions_env = np.zeros_like(actions_tanh, dtype=np.float32)
            actions_env[..., dt_idx] = (actions_tanh[..., dt_idx] + 1.0) * 0.5 * 180.0
            actions_env[..., user_slice] = softmax(actions_tanh[..., user_slice], axis=-1)
            actions_env[..., power_slice] = softmax(actions_tanh[..., power_slice], axis=-1)
        else:
            raise NotImplementedError

        return values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env

    def insert(self, data):
        obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic = data

        rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones == True] = np.zeros(((dones == True).sum(), *self.buffer.rnn_states_critic.shape[3:]), dtype=np.float32)
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

        if self.use_centralized_V:
            share_obs = obs.reshape(self.n_rollout_threads, -1)
            share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
        else:
            share_obs = obs

        # 始终提取 user_rates（无论 use_cvar）
        user_rates_list = []
        for env_info in infos:
            info_dict = env_info[0] if isinstance(env_info, list) and len(env_info) > 0 else {}
            rates = info_dict.get("user_rates", None) if isinstance(info_dict, dict) else None
            rates_np = np.asarray(rates, dtype=np.float32) if rates is not None else np.zeros(
                (self.all_args.num_users,), dtype=np.float32)
            user_rates_list.append(rates_np)
        user_rates = np.stack(user_rates_list)

        # 只有 use_cvar=True 时才提取 cvar_values/cvar_betas
        cvar_values, cvar_betas = None, None
        if getattr(self.all_args, "use_cvar", False):
            cvar_values_list = []
            cvar_betas_list = []
            for env_info in infos:
                info_dict = env_info[0] if isinstance(env_info, list) and len(env_info) > 0 else {}
                rates = info_dict.get("user_rates", None) if isinstance(info_dict, dict) else None
                beta_val = info_dict.get("min_user_rate_kbps", None) if isinstance(info_dict, dict) else None
                rates_np = np.asarray(rates, dtype=np.float32) if rates is not None else np.zeros(
                    (self.all_args.num_users,), dtype=np.float32)
                beta_np = np.array([[float(beta_val)]], dtype=np.float32) if beta_val is not None else np.array(
                    [[float(rates_np.min()) if rates_np.size > 0 else 0.0]], dtype=np.float32)
                cvar_values_list.append(rates_np)
                cvar_betas_list.append(beta_np.squeeze())
            cvar_values = np.stack(cvar_values_list)
            cvar_betas = np.array(cvar_betas_list, dtype=np.float32).reshape(self.n_rollout_threads, 1)

        self.buffer.insert(share_obs, obs, rnn_states, rnn_states_critic, actions, action_log_probs, values, rewards, masks,
                           cvar_values=cvar_values, cvar_betas=cvar_betas, user_rates=user_rates)

    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_episode_rewards = []
        eval_obs = self.eval_envs.reset()

        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, *self.buffer.rnn_states.shape[2:]), dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        for eval_step in range(self.episode_length):
            self.trainer.prep_rollout()
            eval_action, eval_rnn_states = self.trainer.policy.act(np.concatenate(eval_obs),
                                                np.concatenate(eval_rnn_states),
                                                np.concatenate(eval_masks),
                                                deterministic=True)
            eval_actions = np.array(np.split(_t2n(eval_action), self.n_eval_rollout_threads))
            eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))
            
            if self.eval_envs.action_space[0].__class__.__name__ == 'MultiDiscrete':
                for i in range(self.eval_envs.action_space[0].shape):
                    eval_uc_actions_env = np.eye(self.eval_envs.action_space[0].high[i]+1)[eval_actions[:, :, i]]
                    if i == 0:
                        eval_actions_env = eval_uc_actions_env
                    else:
                        eval_actions_env = np.concatenate((eval_actions_env, eval_uc_actions_env), axis=2)
            elif self.eval_envs.action_space[0].__class__.__name__ == 'Discrete':
                eval_actions_env = np.squeeze(np.eye(self.eval_envs.action_space[0].n)[eval_actions], 2)
            elif self.eval_envs.action_space[0].__class__.__name__ == 'Box':
                eval_actions_env = np.tanh(eval_actions)
                action_dim = eval_actions_env.shape[-1]
                num_users = getattr(self.all_args, "num_users", 20)
                num_subcarriers = action_dim - 1 - num_users
                if num_subcarriers <= 0:
                    raise ValueError("Box 动作维度不足以分出子载波部分，请检查配置。")

                dt_idx = 0
                user_slice = slice(1, 1 + num_users)
                power_slice = slice(1 + num_users, 1 + num_users + num_subcarriers)

                eval_actions_env_processed = np.zeros_like(eval_actions_env, dtype=np.float32)
                eval_actions_env_processed[..., dt_idx] = (eval_actions_env[..., dt_idx] + 1.0) * 0.5 * 180.0
                eval_actions_env_processed[..., user_slice] = softmax(eval_actions_env[..., user_slice], axis=-1)
                eval_actions_env_processed[..., power_slice] = softmax(eval_actions_env[..., power_slice], axis=-1)
                eval_actions_env = eval_actions_env_processed
            else:
                raise NotImplementedError

            # Obser reward and next obs
            eval_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(eval_actions_env)
            eval_episode_rewards.append(eval_rewards)

            eval_rnn_states[eval_dones == True] = np.zeros(((eval_dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
            eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones == True] = np.zeros(((eval_dones == True).sum(), 1), dtype=np.float32)

        eval_episode_rewards = np.array(eval_episode_rewards)
        eval_env_infos = {}
        eval_env_infos['eval_average_episode_rewards'] = np.sum(np.array(eval_episode_rewards), axis=0)
        eval_average_episode_rewards = np.mean(eval_env_infos['eval_average_episode_rewards'])
        print("eval average episode rewards of agent: " + str(eval_average_episode_rewards))
        self.log_env(eval_env_infos, total_num_steps)

    @torch.no_grad()
    def render(self):
        """Visualize the env."""
        envs = self.envs
        
        all_frames = []
        for episode in range(self.all_args.render_episodes):
            obs = envs.reset()
            if self.all_args.save_gifs:
                image = envs.render('rgb_array')[0][0]
                all_frames.append(image)
            else:
                envs.render('human')

            rnn_states = np.zeros((self.n_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
            masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
            
            episode_rewards = []
            
            for step in range(self.episode_length):
                calc_start = time.time()

                self.trainer.prep_rollout()
                action, rnn_states = self.trainer.policy.act(np.concatenate(obs),
                                                    np.concatenate(rnn_states),
                                                    np.concatenate(masks),
                                                    deterministic=True)
                actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
                rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))

                if envs.action_space[0].__class__.__name__ == 'MultiDiscrete':
                    for i in range(envs.action_space[0].shape):
                        uc_actions_env = np.eye(envs.action_space[0].high[i]+1)[actions[:, :, i]]
                        if i == 0:
                            actions_env = uc_actions_env
                        else:
                            actions_env = np.concatenate((actions_env, uc_actions_env), axis=2)
                elif envs.action_space[0].__class__.__name__ == 'Discrete':
                    actions_env = np.squeeze(np.eye(envs.action_space[0].n)[actions], 2)
                elif envs.action_space[0].__class__.__name__ == 'Box':
                    actions_tanh = np.tanh(actions)
                    action_dim = actions_tanh.shape[-1]
                    num_users = getattr(self.all_args, "num_users", 20)
                    num_subcarriers = action_dim - 1 - num_users
                    if num_subcarriers <= 0:
                        raise ValueError("Box 动作维度不足以分出子载波部分，请检查配置。")

                    dt_idx = 0
                    user_slice = slice(1, 1 + num_users)
                    power_slice = slice(1 + num_users, 1 + num_users + num_subcarriers)

                    actions_env = np.zeros_like(actions_tanh, dtype=np.float32)
                    actions_env[..., dt_idx] = (actions_tanh[..., dt_idx] + 1.0) * 0.5 * 180.0
                    actions_env[..., user_slice] = softmax(actions_tanh[..., user_slice], axis=-1)
                    actions_env[..., power_slice] = softmax(actions_tanh[..., power_slice], axis=-1)
                else:
                    raise NotImplementedError

                # Obser reward and next obs
                obs, rewards, dones, infos = envs.step(actions_env)
                episode_rewards.append(rewards)

                rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
                masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
                masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

                if self.all_args.save_gifs:
                    image = envs.render('rgb_array')[0][0]
                    all_frames.append(image)
                    calc_end = time.time()
                    elapsed = calc_end - calc_start
                    if elapsed < self.all_args.ifi:
                        time.sleep(self.all_args.ifi - elapsed)
                else:
                    envs.render('human')

            print("average episode rewards is: " + str(np.mean(np.sum(np.array(episode_rewards), axis=0))))

        if self.all_args.save_gifs:
            imageio.mimsave(str(self.gif_dir) + '/render.gif', all_frames, duration=self.all_args.ifi)
