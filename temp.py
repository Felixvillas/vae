import torch
import torch.nn as nn
import numpy as np

from onpolicy.algorithms.r_mappo.multi_r_mappo import MultiR_MAPPO
from onpolicy.utils.util import get_gard_norm, huber_loss, mse_loss
from onpolicy.algorithms.utils.util import check


class PearlMultiR_MAPPO(MultiR_MAPPO):
    def __init__(self, args, policy, n_agents_list, n_enemies_list, device=torch.device('cpu')):
        super().__init__(args, policy, n_agents_list, n_enemies_list, device)

    
    def ppo_update(self, sample, enc_buffer, update_actor=True):
        value_loss, actor_loss, dist_entropy, imp_weights = 0, 0, 0, 0
        value_loss_dict = dict(zip(self.multi_envs, [0 for _ in self.multi_envs]))
        actor_loss_dict = dict(zip(self.multi_envs, [0 for _ in self.multi_envs]))
        value_loss_ratio = dict(zip(self.multi_envs, [1 for _ in self.multi_envs]))
        actor_loss_ratio = dict(zip(self.multi_envs, [1 for _ in self.multi_envs]))
        idx_count = 0
        
        # infer posterior 
        context = [self.policy.sample_context(multi_envs_id=idx, enc_buffer=enc_buffer) for idx, _ in enumerate(sample)]
        # 这里多个环境的context必须同时输入context encoder, 否则反向传播时会报错
        # 原因: 可能是最后的forward覆盖了前面的forward, 而actor和critic不报错的原因是
        # 多个环境的transitions的多次forward计算得到的loss进行了相加, 所以?
        self.policy.infer_posterior_training(context=context)
        
        for idx, env_sample in enumerate(sample):
            env_value_loss, env_actor_loss, env_dist_entropy, env_imp_weights = self.get_ppo_loss(idx, env_sample)
            dist_entropy += env_dist_entropy
            imp_weights += env_imp_weights.mean()
            value_loss_dict[self.multi_envs[idx]] = env_value_loss
            actor_loss_dict[self.multi_envs[idx]] = env_actor_loss
            idx_count += 1
        
        if self.multi_actor_loss is None:
            self.multi_actor_loss = dict(zip(self.multi_envs, [actor_l.detach().item() for actor_l in actor_loss_dict.values()]))
        if self.multi_critic_loss is None:
            self.multi_critic_loss = dict(zip(self.multi_envs, [value_l.detach().item() for value_l in value_loss_dict.values()]))
        for key, value_l in value_loss_dict.items():
            value_l_ratio = np.clip(value_l.detach().item() / self.multi_critic_loss[key], 1-self.loss_balance_clip, 1+self.loss_balance_clip)
            value_loss += value_l * value_l_ratio * self.loss_balance_alpha
            value_loss_ratio[key] = value_l_ratio
        for key, actor_l in actor_loss_dict.items():
            actor_l_ratio = np.clip(actor_l.detach().item() / self.multi_actor_loss[key], 1-self.loss_balance_clip, 1+self.loss_balance_clip)
            actor_loss += actor_l * actor_l_ratio * self.loss_balance_alpha
            actor_loss_ratio[key] = actor_l_ratio
        value_loss /= idx_count
        actor_loss /= idx_count
        dist_entropy /= idx_count
        imp_weights /= idx_count

        # update Actor
        self.policy.actor_optimizer.zero_grad()

        if update_actor:
            actor_loss.backward()

        if self._use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.max_grad_norm)
        else:
            actor_grad_norm = get_gard_norm(self.policy.actor.parameters())

        self.policy.actor_optimizer.step()

        # update context_encoder when not froze context encoder
        
        self.policy.context_enc_optimizer.zero_grad()
        if self.policy.use_ib:
            kl_div = self.policy.compute_kl_div()
            kl_loss = self.policy.kl_lambda * kl_div
            
            if not self.args.context_enc_frozen:
                kl_loss.backward(retain_graph=True)
        if not self.args.context_enc_frozen:
            (value_loss * self.value_loss_coef).backward(retain_graph=True)

        # update critic
        self.policy.critic_optimizer.zero_grad()
        (value_loss * self.value_loss_coef).backward(retain_graph=True) # 这句有没有必要两次出现?

        '''
        step之后, context_encoder权重梯度下降而发生了改变, 如果context_enc_optimizer.step()放至critic loss.backward()之前, 
        会导致critic loss关于context_encoder权重的梯度计算发生错误, 
        因此self.policy.context_enc_optimizer.step() 应放到critic loss.backward()之后
        ''' 
        
        if self._use_max_grad_norm:
            context_encoder_grad_norm = nn.utils.clip_grad_norm_(self.policy.context_encoder.parameters(), self.max_grad_norm)
        else:
            context_encoder_grad_norm = get_gard_norm(self.policy.context_encoder.parameters())
        
        if not self.args.context_enc_frozen:    
            self.policy.context_enc_optimizer.step()

        if self._use_max_grad_norm:
            critic_grad_norm = nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.max_grad_norm)
        else:
            critic_grad_norm = get_gard_norm(self.policy.critic.parameters())

        self.policy.critic_optimizer.step()

        return value_loss_dict, critic_grad_norm, actor_loss_dict, dist_entropy, actor_grad_norm, imp_weights, actor_loss_ratio, value_loss_ratio, kl_loss, context_encoder_grad_norm
    
    def get_ppo_loss(self, idx, sample):
        
        share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, \
        value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
        adv_targ, available_actions_batch = sample

        n_threads, bs_na, r_n, r_h = rnn_states_batch.shape

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        value_preds_batch = check(value_preds_batch).to(**self.tpdv)
        return_batch = check(return_batch).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)

        # Reshape to do in a single forward pass for all steps
        values, action_log_probs, dist_entropy = self.policy.evaluate_actions(share_obs_batch,
                                                                            obs_batch, 
                                                                            rnn_states_batch, 
                                                                            rnn_states_critic_batch, 
                                                                            actions_batch, 
                                                                            masks_batch, 
                                                                            available_actions_batch,
                                                                            active_masks_batch, 
                                                                            self.n_agents_list[idx],
                                                                            self.n_enemies_list[idx],
                                                                            idx)
        # view
        old_action_log_probs_batch = old_action_log_probs_batch.contiguous().view(n_threads*bs_na, -1)
        adv_targ = adv_targ.contiguous().view(n_threads*bs_na, -1)
        value_preds_batch = value_preds_batch.contiguous().view(n_threads*bs_na, -1)
        return_batch = return_batch.contiguous().view(n_threads*bs_na, -1)
        active_masks_batch = active_masks_batch.contiguous().view(n_threads*bs_na, -1)

        
        # actor update
        imp_weights = torch.exp(action_log_probs - old_action_log_probs_batch)

        surr1 = imp_weights * adv_targ
        surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ

        if self._use_policy_active_masks:
            policy_action_loss = (-torch.sum(torch.min(surr1, surr2),
                                            dim=-1,
                                            keepdim=True) * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            policy_action_loss = -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True).mean()

        policy_loss = policy_action_loss
        actor_loss = policy_loss - dist_entropy * self.entropy_coef

        # critic update
        value_loss = self.cal_value_loss(values, value_preds_batch, return_batch, active_masks_batch) * self.value_loss_coef

        return value_loss, actor_loss, dist_entropy, imp_weights

    def train(self, buffer, enc_buffer, update_actor=True):
        self.multi_actor_loss, self.multi_critic_loss = None, None
        multi_advantages = []
        for idx in range(self.num_multi_envs):
            if self._use_popart or self._use_valuenorm:
                advantages = buffer.buffer_lists[idx].returns[:-1] - self.value_normalizer.denormalize(buffer.buffer_lists[idx].value_preds[:-1])
            else:
                advantages = buffer.buffer_lists[idx].returns[:-1] - buffer.buffer_lists[idx].value_preds[:-1]
            advantages_copy = advantages.copy()
            advantages_copy[buffer.buffer_lists[idx].active_masks[:-1] == 0.0] = np.nan
            mean_advantages = np.nanmean(advantages_copy)
            std_advantages = np.nanstd(advantages_copy)
            advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)
            multi_advantages.append(advantages)
        
        train_info = {}

        train_info['value_loss'] = dict(zip(self.multi_envs, [0 for _ in self.multi_envs]))
        train_info['policy_loss'] = dict(zip(self.multi_envs, [0 for _ in self.multi_envs]))
        train_info['actor_loss_ratio'] = dict(zip(self.multi_envs, [0 for _ in self.multi_envs]))
        train_info['critic_loss_ratio'] = dict(zip(self.multi_envs, [0 for _ in self.multi_envs]))
        train_info['dist_entropy'] = 0
        train_info['actor_grad_norm'] = 0
        train_info['critic_grad_norm'] = 0
        train_info['ratio'] = 0
        train_info['context_kl_loss'] = 0
        train_info['context_encoder_grad_norm'] = 0

        for _ in range(self.ppo_epoch):
            if self._use_recurrent_policy:
                data_generator = buffer.recurrent_generator(multi_advantages, self.num_mini_batch, self.data_chunk_length)
            elif self._use_naive_recurrent:
                data_generator = buffer.naive_recurrent_generator(multi_advantages, self.num_mini_batch)
            else:
                data_generator = buffer.feed_forward_generator(multi_advantages, self.num_mini_batch)

            for idx, sample in enumerate(zip(*data_generator)):
                # print(f"PPO Update | {idx} Time")
                value_loss_dict, critic_grad_norm, policy_loss_dict, dist_entropy, actor_grad_norm, imp_weights, \
                    actor_loss_ratio, critic_loss_ratio, kl_loss, context_encoder_grad_norm = self.ppo_update(sample, enc_buffer, update_actor)

                for v_key, v_value in value_loss_dict.items():
                    train_info['value_loss'][v_key] += v_value.item()
                for p_key, p_value in policy_loss_dict.items():
                    train_info['policy_loss'][p_key] += p_value.item()
                train_info['dist_entropy'] += dist_entropy.item()
                train_info['actor_grad_norm'] += actor_grad_norm
                train_info['critic_grad_norm'] += critic_grad_norm
                train_info['ratio'] += imp_weights
                for a_key, a_value in actor_loss_ratio.items():
                    train_info['actor_loss_ratio'][a_key] += a_value
                for c_key, c_value in critic_loss_ratio.items():
                    train_info['critic_loss_ratio'][c_key] += c_value
                train_info['context_kl_loss'] += kl_loss.item()
                train_info['context_encoder_grad_norm'] += context_encoder_grad_norm

                # stop backprop
                self.policy.detach_z()

        num_updates = self.ppo_epoch * self.num_mini_batch

        for k in train_info.keys():
            if isinstance(train_info[k], dict):
                for key in train_info[k].keys():
                    train_info[k][key] /= num_updates
            else:
                train_info[k] /= num_updates

        return train_info

    def prep_training(self):
        super().prep_training()
        self.policy.context_encoder.train()

    def prep_rollout(self):
        super().prep_rollout()
        self.policy.context_encoder.eval()