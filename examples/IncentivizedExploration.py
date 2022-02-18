"""
Multi-armed Bandit examples taken from Reinforcement Learning: An Introduction
by Sutton and Barto, 2nd ed. rev Oct2015.
"""
from bandits.environment import Environment
from bandits.bandit import GaussianBandit
from bandits.agent import Agent
from bandits.policy import (EpsilonGreedyPolicy, GreedyPolicy, UCBPolicy,
                            SoftmaxPolicy)


class EpsilonGreedyExample:
    label = 'Incentivized Exploration'
    bandit = GaussianBandit(10)
    agents = [
        Agent(bandit, GreedyPolicy()),
        Agent(bandit, EpsilonGreedyPolicy(0.01)),
        Agent(bandit, EpsilonGreedyPolicy(0.1)),
    ]




if __name__ == '__main__':
    experiments = 500
    trials = 10000

    example = EpsilonGreedyExample
    # example = OptimisticInitialValueExample
    # example = UCBExample
    # example = GradientExample

    env = Environment(example.bandit, example.agents, example.label)
    scores, optimal,regret = env.run(trials, experiments)
    # env.plot_results(scores, optimal)
    env.plot_regret(regret)
    
    
