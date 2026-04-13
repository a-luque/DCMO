class Trainer:
    def __init__(self, bandit_alg, system, log_at=100):
        """
        Initializes a Trainer object.

        Parameters:
        - bandit_alg: The bandit algorithm to be used.
        - system: The system to be trained.
        - log_at: The number of steps after which to log the performance of the bandit algorithm.
        """
        self.bandit_alg = bandit_alg
        self.system = system
        self.log_at = log_at

    def train(self, n_steps, logger=None):
        """
        Train the system for n_steps.

        Parameters:
        - n_steps: The number of steps to train the system.
        """
        for t in range(n_steps):
            if logger is not None and self.bandit_alg.t % self.log_at == 0:
                logger.log_data(t, self.bandit_alg, self.system)
            x = self.system.sample_context()
            index = self.bandit_alg.act(x)
            reward = self.system.step(index, x)
            self.bandit_alg.update(index, x, reward)

        return self.bandit_alg, logger.get_log()


class Logger:
    def __init__(self, log_samples=100):
        """
        Initializes a Logger object.
        """
        self.log_samples = log_samples
        self.log = {"t": []} # , "expected_reward": []}

    def log_data(self, t, bandit_alg, system):
        """
        Log the performance of the bandit algorithm.

        Parameters:
        - t: The current time step.
        - bandit_alg: The bandit algorithm.
        - system: The system.
        """

        expected_reward = 0
        for _ in range(self.log_samples):
            x = system.sample_context()
            index = bandit_alg.argmax(x)
            reward = system.step(index, x)
            expected_reward += reward

        expected_reward /= self.log_samples

        self.log["t"].append(t)
        self.log["expected_reward"].append(expected_reward)

    def get_log(self):
        """
        Get the log of the bandit algorithm.

        Returns:
        - The log of the bandit algorithm.
        """
        return self.log