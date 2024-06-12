import matplotlib.pyplot as plt
import pyGMs as gm
import numpy as np
import torch
import pyro
import pyro.infer
import pyro.optim
import pyro.distributions as dist
import pyro.poutine as poutine

import random
import pandas as pd


def construct_ranking(samples: dict, teams) -> dict:
    return {
        team: rank for rank, team in enumerate(
            sorted(teams,
                   key=lambda i: samples[f"X{i}"].mean(), reverse=True)
        )
    }


def model(matches, n_teams):
    X = [pyro.sample(f"X{i}", dist.Normal(0, 1)) for i in range(n_teams)]

    for i, m in enumerate(matches):
        pyro.sample(f"w{i}",
                    dist.Bernoulli(logits=X[m[0]]-X[m[1]]),
                    obs=torch.tensor(1.))


class MCMC_Session:
    def __init__(self, seed=None, custom_model=None):
        if seed is not None:
            pyro.set_rng_seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

        if model is not None:
            self.model = custom_model if custom_model is not None else model

        self.runs = []

    def run(self, dataset: list, n_samples: list[int], n_teams: int):
        self.runs = []

        # Run MCMC
        nuts_kernel = pyro.infer.NUTS(model, jit_compile=False)
        mcmc = pyro.infer.MCMC(nuts_kernel, num_samples=max(n_samples),
                               warmup_steps=20, num_chains=1)

        mcmc.run(dataset, n_teams)

        # Get samples
        for n in n_samples:
            self.runs.append((n, mcmc.get_samples(n)))

        return self.runs

    def compare(self, other_sample: dict, teams: list):
        if self.runs == []:
            raise ValueError("No runs to compare")

        other_ranking = construct_ranking(other_sample, teams)

        X = [other_ranking[i] for i in teams]

        # Auto shape the plots
        if len(self.runs) < 3:
            cols = len(self.runs)
            rows = 1
        else:
            cols = 3
            rows = int(len(self.runs) / cols + 0.99999)

        fig, axs = plt.subplots(rows, cols, figsize=(15, 10))

        for i, (n, samples) in enumerate(self.runs):
            ranking = construct_ranking(samples, teams)
            Y = [ranking[i] for i in teams]

            mse = np.mean((np.array(X) - np.array(Y))**2)

            if cols == 1:
                ax = axs
            elif cols == 2:
                ax = axs[i]
            else:
                ax = axs[i//cols, i % cols]

            ax.plot(X, Y, 'o', alpha=0.5)
            ax.plot([0, len(teams)], [0, len(teams)], 'k--')
            ax.set_title(f"n={n}, mse={mse:.2f}")

        plt.tight_layout()
        plt.show()
