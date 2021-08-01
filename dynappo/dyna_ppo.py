"""DyNA-PPO explorer."""
from functools import partial
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.stats
import sklearn
import sklearn.ensemble
import sklearn.gaussian_process
import sklearn.linear_model
import sklearn.tree
import tensorflow as tf
from tf_agents.agents.ppo import ppo_agent
from tf_agents.drivers import dynamic_episode_driver
from tf_agents.environments import tf_py_environment
from tf_agents.environments.utils import validate_py_environment
from tf_agents.networks import actor_distribution_network, value_network
from tf_agents.replay_buffers import tf_uniform_replay_buffer


import baselines
from baselines.explorers.environments.dyna_ppo import (
    DynaPPOEnvironment as DynaPPOEnv,
)
from baselines.explorers.environments.dyna_ppo import (
    DynaPPOEnvironmentMutative as DynaPPOEnvMut,
)
import sequence_utils as s_utils


class DynaPPOEnsemble():
    def __init__(
        self,
        seq_len: int,
        alphabet: str,
        r_squared_threshold: float = 0.5,
        models: Optional[List] = None,
    ):
        """Create the ensemble from `models`."""
        super().__init__(name="DynaPPOEnsemble")

        if models is None:
            models = [
                # FLEXS models
                baselines.models.GlobalEpistasisModel(seq_len, 100, alphabet),
                baselines.models.MLP(seq_len, 200, alphabet),
                baselines.models.CNN(seq_len, 32, 100, alphabet),
                # Sklearn models
                baselines.models.LinearRegression(alphabet),
                baselines.models.RandomForest(alphabet),
                baselines.models.SklearnRegressor(
                    sklearn.neighbors.KNeighborsRegressor(),
                    alphabet,
                    "nearest_neighbors",
                ),
                baselines.models.SklearnRegressor(
                    sklearn.linear_model.Lasso(), alphabet, "lasso"
                ),
                baselines.models.SklearnRegressor(
                    sklearn.linear_model.BayesianRidge(),
                    alphabet,
                    "bayesian_ridge",
                ),
                baselines.models.SklearnRegressor(
                    sklearn.gaussian_process.GaussianProcessRegressor(),
                    alphabet,
                    "gaussian_process",
                ),
                baselines.models.SklearnRegressor(
                    sklearn.ensemble.GradientBoostingRegressor(),
                    alphabet,
                    "gradient_boosting",
                ),
                baselines.models.SklearnRegressor(
                    sklearn.tree.ExtraTreeRegressor(), alphabet, "extra_trees"
                ),
            ]

        self.models = models
        self.r_squared_vals = np.ones(len(self.models))
        self.r_squared_threshold = r_squared_threshold

    def train(self, sequences, labels):
        """Train the ensemble, calculating $r^2$ values on a holdout set."""
        if len(sequences) < 10:
            return

        (train_X, test_X, train_y, test_y,) = sklearn.model_selection.train_test_split(
            np.array(sequences), np.array(labels), test_size=0.25
        )

        # Train each model in the ensemble
        for model in self.models:
            model.train(train_X, train_y)

        # Calculate r^2 values for each model in the ensemble on test set
        self.r_squared_vals = []
        for model in self.models:
            y_preds = model.get_fitness(test_X)

            # If either `y_preds` or `test_y` are constant, we can't calculate r^2,
            # so assign an r^2 value of zero.
            if (y_preds[0] == y_preds).all() or (test_y[0] == test_y).all():
                self.r_squared_vals.append(0)
            else:
                self.r_squared_vals.append(
                    scipy.stats.pearsonr(test_y, model.get_fitness(test_X))[0] ** 2
                )

    def _fitness_function(self, sequences):
        passing_models = [
            model
            for model, r_squared in zip(self.models, self.r_squared_vals)
            if r_squared >= self.r_squared_threshold
        ]

        if len(passing_models) == 0:
            return self.models[np.argmax(self.r_squared_vals)].get_fitness(sequences)

        return np.mean(
            [model.get_fitness(sequences) for model in passing_models], axis=0
        )


class DynaPPO():

    def __init__(
        self,
        landscape,
        rounds: int,
        sequences_batch_size: int,
        model_queries_per_batch: int,
        starting_sequence: str,
        alphabet: str,
        log_file: Optional[str] = None,
        model: Optional = None,
        num_experiment_rounds: int = 10,
        num_model_rounds: int = 1,
        env_batch_size: int = 4,
    ):
        tf.config.run_functions_eagerly(False)#便于调试

        name = f"DynaPPO_Agent_{num_experiment_rounds}_{num_model_rounds}"

        if model is None:
            model = DynaPPOEnsemble(
                len(starting_sequence),
                alphabet,
            )
            # Some models in the ensemble need to be trained on dummy dataset before
            # they can predict
            model.train(
                s_utils.generate_random_sequences(len(starting_sequence), 10, alphabet),
                [0] * 10,
            )

        super().__init__(
            model,
            name,
            rounds,
            sequences_batch_size,
            model_queries_per_batch,
            starting_sequence,
            log_file,
        )

        self.alphabet = alphabet
        self.num_experiment_rounds = num_experiment_rounds
        self.num_model_rounds = num_model_rounds
        self.env_batch_size = env_batch_size

        env = DynaPPOEnv(
            self.alphabet, len(starting_sequence), model, landscape, env_batch_size
        )
        self.tf_env = tf_py_environment.TFPyEnvironment(env)

        actor_net = actor_distribution_network.ActorDistributionNetwork(
            self.tf_env.observation_spec(),
            self.tf_env.action_spec(),
            fc_layer_params=[128],
        )
        value_net = value_network.ValueNetwork(
            self.tf_env.observation_spec(), fc_layer_params=[128]
        )

        print(self.tf_env.action_spec())
        self.agent = ppo_agent.PPOAgent(
            time_step_spec=self.tf_env.time_step_spec(),
            action_spec=self.tf_env.action_spec(),
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
            actor_net=actor_net,
            value_net=value_net,
            num_epochs=10,
            summarize_grads_and_vars=False,
        )
        self.agent.initialize()

    def add_last_seq_in_trajectory(self, experience, new_seqs):
        for is_bound, obs in zip(experience.is_boundary(), experience.observation):
            if is_bound:
                seq = s_utils.one_hot_to_string(obs.numpy()[:, :-1], self.alphabet)
                new_seqs[seq] = self.tf_env.get_cached_fitness(seq)

    def propose_sequences(
        self, measured_sequences_data: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        replay_buffer_capacity = 10001
        replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            self.agent.collect_data_spec,
            batch_size=self.env_batch_size,
            max_length=replay_buffer_capacity,
        )

        sequences = {}
        collect_driver = dynamic_episode_driver.DynamicEpisodeDriver(
            self.tf_env,
            self.agent.collect_policy,
            observers=[
                replay_buffer.add_batch,
                partial(self.add_last_seq_in_trajectory, new_seqs=sequences),
            ],
            num_episodes=1,
        )
        experiment_based_training_budget = self.sequences_batch_size
        self.tf_env.set_fitness_model_to_gt(True)
        previous_landscape_cost = self.tf_env.landscape.cost
        while (
            self.tf_env.landscape.cost - previous_landscape_cost
            < experiment_based_training_budget
        ):
            collect_driver.run()

        trajectories = replay_buffer.gather_all()
        self.agent.train(experience=trajectories)
        replay_buffer.clear()
        sequences.clear()

        # Model-based training rounds
        self.tf_env.set_fitness_model_to_gt(False)
        previous_model_cost = self.model.cost
        for _ in range(self.num_model_rounds):
            if self.model.cost - previous_model_cost >= self.model_queries_per_batch:
                break

            previous_round_model_cost = self.model.cost
            while self.model.cost - previous_round_model_cost < int(
                self.model_queries_per_batch / self.num_model_rounds
            ):
                collect_driver.run()

            trajectories = replay_buffer.gather_all()
            self.agent.train(experience=trajectories)
            replay_buffer.clear()

        sequences = {
            seq: fitness
            for seq, fitness in sequences.items()
            if seq not in set(measured_sequences_data["sequence"])
        }
        new_seqs = np.array(list(sequences.keys()))
        preds = np.array(list(sequences.values()))
        sorted_order = np.argsort(preds)[::-1][: self.sequences_batch_size]

        return new_seqs[sorted_order], preds[sorted_order]


class DynaPPOMutative():

    def __init__(
        self,
        landscape,
        rounds: int,
        sequences_batch_size: int,
        model_queries_per_batch: int,
        starting_sequence: str,
        alphabet: str,
        log_file: Optional[str] = None,
        model: Optional = None,
        num_experiment_rounds: int = 10,
        num_model_rounds: int = 1,
    ):
        tf.config.run_functions_eagerly(False)

        name = f"DynaPPO_Agent_{num_experiment_rounds}_{num_model_rounds}"

        if model is None:
            model = DynaPPOEnsemble(
                len(starting_sequence),
                alphabet,
            )
            model.train(
                s_utils.generate_random_sequences(len(starting_sequence), 10, alphabet),
                [0] * 10,
            )

        super().__init__(
            model,
            name,
            rounds,
            sequences_batch_size,
            model_queries_per_batch,
            starting_sequence,
            log_file,
        )

        self.alphabet = alphabet
        self.num_experiment_rounds = num_experiment_rounds
        self.num_model_rounds = num_model_rounds

        env = DynaPPOEnvMut(
            alphabet=self.alphabet,
            starting_seq=starting_sequence,
            model=model,
            landscape=landscape,
            max_num_steps=model_queries_per_batch,
        )
        validate_py_environment(env, episodes=1)
        self.tf_env = tf_py_environment.TFPyEnvironment(env)

        encoder_layer = tf.keras.layers.Lambda(lambda obs: obs["sequence"])
        actor_net = actor_distribution_network.ActorDistributionNetwork(
            self.tf_env.observation_spec(),
            self.tf_env.action_spec(),
            preprocessing_combiner=encoder_layer,
            fc_layer_params=[128],
        )
        value_net = value_network.ValueNetwork(
            self.tf_env.observation_spec(),
            preprocessing_combiner=encoder_layer,
            fc_layer_params=[128],
        )

        self.agent = ppo_agent.PPOAgent(
            self.tf_env.time_step_spec(),
            self.tf_env.action_spec(),
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
            actor_net=actor_net,
            value_net=value_net,
            num_epochs=10,
            summarize_grads_and_vars=False,
        )
        self.agent.initialize()

    def add_last_seq_in_trajectory(self, experience, new_seqs):
        if experience.is_boundary():
            seq = s_utils.one_hot_to_string(
                experience.observation["sequence"].numpy()[0], self.alphabet
            )
            new_seqs[seq] = experience.observation["fitness"].numpy().squeeze()

            top_fitness = max(new_seqs.values())
            top_sequences = [
                seq for seq, fitness in new_seqs.items() if fitness >= 0.9 * top_fitness
            ]
            if len(top_sequences) > 0:
                self.tf_env.pyenv.envs[0].seq = np.random.choice(top_sequences)
            else:
                self.tf_env.pyenv.envs[0].seq = np.random.choice(
                    [seq for seq, _ in new_seqs.items()]
                )

    def propose_sequences(
        self, measured_sequences_data: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Propose top `sequences_batch_size` sequences for evaluation."""
        num_parallel_environments = 1
        replay_buffer_capacity = 10001
        replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            self.agent.collect_data_spec,
            batch_size=num_parallel_environments,
            max_length=replay_buffer_capacity,
        )

        sequences = {}
        collect_driver = dynamic_episode_driver.DynamicEpisodeDriver(
            self.tf_env,
            self.agent.collect_policy,
            observers=[
                replay_buffer.add_batch,
                partial(self.add_last_seq_in_trajectory, new_seqs=sequences),
            ],
            num_episodes=1,
        )
        current_round = measured_sequences_data["round"].max()
        experiment_based_training_budget = int(
            (self.rounds - current_round + 1)
            / self.rounds
            * self.sequences_batch_size
            / 2
        )
        self.tf_env.envs[0].set_fitness_model_to_gt(True)
        previous_landscape_cost = self.tf_env.envs[0].landscape.cost
        while (
            self.tf_env.envs[0].landscape.cost - previous_landscape_cost
            < experiment_based_training_budget
        ):
            collect_driver.run()

        trajectories = replay_buffer.gather_all()
        self.agent.train(experience=trajectories)
        replay_buffer.clear()
        sequences.clear()

        self.tf_env.envs[0].set_fitness_model_to_gt(False)
        previous_model_cost = self.model.cost
        for _ in range(self.num_model_rounds):
            if self.model.cost - previous_model_cost >= self.model_queries_per_batch:
                break

            previous_round_model_cost = self.model.cost
            while self.model.cost - previous_round_model_cost < int(
                self.model_queries_per_batch / self.num_model_rounds
            ):
                collect_driver.run()

            trajectories = replay_buffer.gather_all()
            self.agent.train(experience=trajectories)
            replay_buffer.clear()

        sequences = {
            seq: fitness
            for seq, fitness in sequences.items()
            if seq not in set(measured_sequences_data["sequence"])
        }
        new_seqs = np.array(list(sequences.keys()))
        preds = np.array(list(sequences.values()))
        sorted_order = np.argsort(preds)[
            : -(self.sequences_batch_size - experiment_based_training_budget) : -1
        ]

        return new_seqs[sorted_order], preds[sorted_order]
