import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from tqdm import tqdm

from typing import Callable
from copy import deepcopy

def train_model(model,
                optimizer,
                criterion,
                Task: Callable,
                params: dict,
                num_epochs,
                batch_size,
                object: str,
                verbose: bool):
    """object in ["r_volume", "it_volume", "it_cost"]"""
    losses = []

    for epoch in tqdm(range(num_epochs)):
        samples = []
        for j in range(batch_size):
            task = Task(**params)
            if object=="r_volume":
                samples.append(torch.from_numpy(task.rukzaks_volume))
            elif object=="it_volume":
                samples.append(torch.from_numpy(task.items_volume))
            elif object=="it_cost":
                samples.append(torch.from_numpy(task.items_cost))
        samples = torch.stack(samples).to(torch.float32).unsqueeze(-1)

        y, _ = model(samples)
        loss = criterion(y, torch.flip(samples, dims=[1]))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        if verbose:
            print(f"loss on {epoch} epoch:", loss.item())

    return model, losses



