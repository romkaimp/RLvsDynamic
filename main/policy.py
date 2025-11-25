import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from tqdm import tqdm

from typing import Callable
from copy import deepcopy

def one_sample_sum(pi1, pi2, R):
    return torch.sum(torch.log(pi1) + torch.log(pi2))*R

def train_model(model1, model2, optimizer1, optimizer2, Task: Callable, params: dict, num_epochs, samples, encoders):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    losses = []
    steppas = []
    Rewards = []
    for epoch in tqdm(range(num_epochs)):
        log_pi_mult_return = []

        sample_steps = []
        sample_rewards = []
        for j in range(samples):
            task = Task(**params)
            max_value = float(task.solve_dynamically())
            log_pi = []

            model1.train()
            model2.train()

            cont = True
            rewards = []
            steps = 0
            while task.not_end() and cont:
                state = task.get_state()
                n = state[0].shape[0]
                k = state[1].shape[0]
                embeddings = []
                for idx, model in enumerate(encoders):
                    model.eval()
                    with torch.no_grad():
                        _, z = model.encode_zero(torch.from_numpy(state[idx]).to(torch.float32)[None, :, None])
                        embeddings.append(z)

                #print("embs:", embeddings[0].shape)
                state1 = torch.cat(embeddings, dim=-1).unsqueeze(1)
                #print("state1:", state1.shape)
                #print("n:", n)
                logits1 = model1(state1, n).squeeze(-1) # size:  - pi(a_1|s), pi(a_2|a_1, s)
                min_val = torch.min(torch.from_numpy(state[1]).float())  # scalar

                state0 = torch.from_numpy(state[0]).float()  # shape = (n,)

                mask1 = (state0 >= min_val)  # shape = (n,)

                mask1 = mask1.unsqueeze(0)  # shape = (1, n)

                masked_logits1 = logits1.clone()
                masked_logits1[~mask1] = -1e9

                dist1 = Categorical(logits=masked_logits1)

                a1 = dist1.sample()
                #print("a:", a1)
                #print('st_a1:',state[0][a1] )
                action = torch.tensor([state[0][a1]]).to(torch.float32)[:, None]
                #print("action:", action.shape)
                embeddings.append(action)
                state2 = torch.cat(embeddings, dim=-1).unsqueeze(1)
                logits2 = model2(state2, k).squeeze(-1)
                mask = (state[0][a1] >= state[1])  # [k]
                #print(mask)
                mask = np.reshape(mask, (1, -1))  # True там, где разрешено
                mask = torch.tensor(mask, dtype=torch.bool)
                masked_logits = logits2.clone()
                masked_logits[~mask] = -1e9  # запрещённые позиции -> -inf

                dist2 = Categorical(logits=masked_logits)

                #dist2 = Categorical(logits=logits2)

                a2 = dist2.sample()

                # Лог-вероятность общей пары — сумма лог-проб компонент
                log_pi.append(dist1.log_prob(a1) + dist2.log_prob(a2))  # shape (N,)

                cont, rew = task.take_action(a1, a2) # s_t, a_t -> s_t+1
                rewards.append(rew)
                steps += 1

            sample_steps.append(steps)
            sample_rewards.append(task.total_sum)
            rews = torch.tensor(rewards, dtype=torch.float, device=device)  # shape (T,)

            # вычислить G в обратном порядке
            # G_rev = cumsum(reverse(rews)), затем reverse обратно
            G_rev = torch.cumsum(rews.flip(dims=(0,)), dim=0)  # (T,)
            G = G_rev.flip(dims=(0,))  # (T,)

            # 2) stack log_probs -> (T,)
            logp = torch.stack(log_pi, dim=0)  # (T,), dtype same as log_pi

            # 3) loss for this episode
            loss_episode = - (logp * G).sum()
            log_pi_mult_return.append(loss_episode)
            sample_steps.append(steps)
        loss = torch.stack(log_pi_mult_return).mean()
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        loss.backward()
        optimizer1.step()
        optimizer2.step()

        steppas.append(sum(sample_steps)/len(sample_steps))
        Rewards.append(sum(sample_rewards)/len(sample_rewards))
        losses.append(loss.item())
        print(f"loss on {epoch} epoch:", losses[-1])
        print("mean steps per sample:", steppas[-1])
        print("mean reward per sample:", Rewards[-1])
    return model1, model2, losses, steppas