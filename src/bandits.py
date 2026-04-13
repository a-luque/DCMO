import numpy as np
from sklearn.linear_model import LogisticRegression
from scipy.special import expit


class Explorer:
    def __init__(self, n_arms, feature_dim):
        self.n_arms = n_arms
        self.t = -1
        self.feature_dim = feature_dim

    def act(self, X):
        pass

    def argmax(self, X):
        pass

    def update(self, arm, X, y):
        pass


class LinearExplorer(Explorer):
    """
    Algorithm for finding the best policy in contextual bandits.
    The algorithm models the reward from each arm as a linear regression model (continuous rewards both positive and negative).
    """

    def __init__(self, n_arms, feature_dim, regularization=0.1):
        super().__init__(n_arms, feature_dim)
        self.regularization = regularization
        self.arm_data = {i: {"X": [], "y": []} for i in range(n_arms)}
        self.arm_hessians_inv = {
            i: np.eye(feature_dim) * regularization for i in range(n_arms)
        }
        self.weights = np.zeros((n_arms, feature_dim))
        self.b = np.zeros((n_arms, feature_dim))

    def _compute_expected_rewards(self, X):
        """
        Return the expected reward for each arm given the context X
        """
        return np.dot(X, self.weights.T)

    def act(self, X):
        """
        Selects the arm with the highest uncertainty based on the given input features.

        Parameters:
        - X: Input features for arm selection.

        Returns:
        - The index of the arm with the highest uncertainty.
        """
        uncertainties = [
            np.sqrt(np.dot(np.dot(X, self.arm_hessians_inv[i]), X.T))
            for i in range(self.n_arms)
        ]
        return np.argmax(uncertainties)

    def argmax(self, X):
        """
        Selects the arm with the highest expected reward based on the given input features.

        Parameters:
        - X: Input features for arm selection.

        Returns:
        - The index of the arm with the highest expected reward.
        """
        expected_rewards = self._compute_expected_rewards(X)
        return np.argmax(expected_rewards)

    def update(self, arm, X, y):
        """
        Update the model with new data for a specific arm.
        """
        self.arm_data[arm]["X"].append(X)
        self.arm_data[arm]["y"].append(y)
        self.t += 1
        outer = np.outer(X, X)
        # perform sherman morrison update
        self.arm_hessians_inv[arm] -= (
            self.arm_hessians_inv[arm].dot(outer).dot(self.arm_hessians_inv[arm])
        ) / (1 + X.dot(self.arm_hessians_inv[arm]).dot(X.T))
        # update weights
        self.b[arm] += y * X
        self.weights[arm] = self.arm_hessians_inv[arm].dot(self.b[arm])


class LogisticExplorer(Explorer):
    """
    Algorithm for finding the best policy in contextual bandits.
    The algorithm models the reward from each arm as a logistic regression model (This means binary rewards).
    No bias term is used in the model for simplicity but can be added by adding a constant feature to each context.


    """

    def __init__(
        self,
        n_arms,
        feature_dim,
        noise_scaling=False,
        regularization=0.1,
        recompute_every=10,
    ):
        """
        Initializes a Bandits object.

        Parameters:
        - n_arms (int): The number of arms or actions in the bandit problem.
        - feature_dim (int): The dimensionality of the context vectors.
        - noise_scaling (bool): Whether to scale norm with predicted noise.
        - regularization (float): The regularization parameter for logistic regression.
        - recompute_every (int): The number of steps after which to recompute the logistic models.

        """
        super().__init__(n_arms, feature_dim)
        self.noise_scaling = noise_scaling
        self.regularization = regularization
        self.recompute_every = recompute_every
        self.arm_data = {i: {"X": [], "y": []} for i in range(n_arms)}
        self.arm_hessians_inv = {
            i: np.eye(feature_dim) * regularization for i in range(n_arms)
        }
        self.logistic_models = {i: None for i in range(n_arms)}
        self.weights = np.zeros((n_arms, feature_dim))

    def _compute_probabilities(self, X):
        """
        Return the probability of each arm provding a reward given the context X
        """
        return expit(np.dot(X, self.weights.T))

    def act(self, X):
        """
        Selects the arm with the highest uncertainty based on the given input features.

        Parameters:
        - X: Input features for arm selection.

        Returns:
        - The index of the arm with the highest uncertainty.
        """
        if not all(self.logistic_models.values()):
            # If no data is available, return a random arm
            return np.random.choice(self.n_arms)

        if self.noise_scaling:
            # Uncertainty depending on the predicted noise of the arm
            uncertainties = [
                expit(self.weights[i].dot(X))
                * (1 - expit(self.weights[i].dot(X)))
                * np.sqrt(X.dot(self.arm_hessians_inv[i]).dot(X.T))
                for i in range(self.n_arms)
            ]
        else:
            uncertainties = [
                np.sqrt(np.dot(np.dot(X, self.arm_hessians_inv[i]), X.T))
                for i in range(self.n_arms)
            ]

        # Return the index of the arm with the highest uncertainty
        return np.argmax(uncertainties)

    def argmax(self, X):
        """
        Selects the arm with the highest expected reward based on the given input features.

        Parameters:
        - X: Input features for arm selection.

        Returns:
        - The index of the arm with the highest expected reward.
        """
        if not all(self.logistic_models.values()):
            # If no data is available, return a random arm
            return np.random.choice(self.n_arms)
        # Compute the expected reward for each arm
        expected_rewards = self._compute_probabilities(X) 

        # Return the index of the arm with the highest expected reward

        return np.argmax(expected_rewards)

    def update(self, arm, X, y):
        """
        Update the model with new data for a specific arm.

        Parameters:
            arm (int): The index of the arm.
            X (array-like): The input features for the new data.
            y (int or float): reward.

        Returns:
            None
        """
        self.arm_data[arm]["X"].append(X)
        self.arm_data[arm]["y"].append(y)
        self.t += 1

        if self.t > 0 and self.t % self.recompute_every == 0:
            self._recompute_models()
        else:
            if not all(self.logistic_models.values()):
                return
            # perform local update of Hessian around current parameter using Sherman-Morrison formula
            X = np.array(X).reshape(1, -1)
            y_hat = self.logistic_models[arm].predict(X).flatten()
            pred_noise = y_hat * (1 - y_hat)
            outer_product = np.outer(X, X)
            self.arm_hessians_inv[arm] -= (
                pred_noise
                * (
                    self.arm_hessians_inv[arm]
                    .dot(outer_product)
                    .dot(self.arm_hessians_inv[arm])
                )
                / (1 + X.dot(self.arm_hessians_inv[arm]).dot(X.T) * pred_noise)
            )

    def _recompute_models(self):
        """
        Recomputes the logistic regression models for each arm based on the arm data.

        This method fits a logistic regression model for each arm using the arm data.
        It updates the weights and hessian matrices for each arm based on the fitted models.

        Returns:
            None
        """
        for arm in range(self.n_arms):
            if len(self.arm_data[arm]["X"]) > 0:
                # prep data
                X = np.array(self.arm_data[arm]["X"])
                y = np.array(self.arm_data[arm]["y"])
                print("Reward mean: ", str(y.mean()))
                if y.mean() > 0 and y.mean() < 1:
                    # fit model
                    self.logistic_models[arm] = LogisticRegression(
                        C=1 / self.regularization, fit_intercept=False, max_iter=3000
                    )
                    self.logistic_models[arm].fit(X, y)

                    # update weights and hessian
                    self.weights[arm] = self.logistic_models[arm].coef_
                    # recompute hessian
                    H = np.zeros((self.feature_dim, self.feature_dim))
                    for i in range(len(X)):
                        y_hat = expit(self.weights[arm].dot(X[i]))
                        pred_noise = y_hat * (1 - y_hat)
                        H += pred_noise * np.outer(X[i], X[i])
                    self.arm_hessians_inv[arm] = np.linalg.inv(
                        H + np.eye(self.feature_dim) * self.regularization
                    )


if __name__ == "__main__":
    # test algorithm
    n_arms = 3
    feature_dim = 5
    n_samples = 1000
    update_every = 25
    np.random.seed(0)
    X = np.random.randn(n_samples, feature_dim)
    theta = np.random.randn(n_arms, feature_dim)  # / np.sqrt(feature_dim)
    X_val = np.random.randn(400, feature_dim)  # validation data
    y_val = np.argmax(np.dot(X_val, theta.T), axis=1)
    print(y_val)
    bandit1 = LogisticExplorer(
        n_arms,
        feature_dim,
        noise_scaling=True,
        recompute_every=update_every,
        regularization=0.01,
    )
    bandit2 = LinearExplorer(
        n_arms,
        feature_dim,
        regularization=0.01,
    )
    val_error1 = []
    val_error2 = []
    preward1 = []
    preward2 = []
    for i in range(n_samples):
        arm1 = bandit1.act(X[i])
        arm2 = bandit2.act(X[i])
        reward1 = np.random.binomial(1, expit(X[i].dot(theta[arm1])))
        reward2 = np.random.binomial(1, expit(X[i].dot(theta[arm2])))
        bandit1.update(arm1, X[i], reward1)
        bandit2.update(arm2, X[i], reward2)
        if i % update_every == 0:
            y_pred1 = np.argmax(np.dot(X_val, bandit1.weights.T), axis=1)
            y_pred2 = np.argmax(np.dot(X_val, bandit2.weights.T), axis=1)
            val_error1.append(np.mean(y_pred1 != y_val))
            val_error2.append(np.mean(y_pred2 != y_val))
            print(
                f"Step {i}: Validation error with noise scaling: {val_error1[-1]}, Validation error without noise scaling: {val_error2[-1]}"
            )
            arms1 = np.argmax(np.dot(X_val, bandit1.weights.T), axis=1)
            arms2 = np.argmax(np.dot(X_val, bandit2.weights.T), axis=1)
            preward1.append(
                np.mean(
                    expit(X_val.dot(bandit1.weights.T))[np.arange(len(X_val)), arms1]
                )
            )
            preward2.append(
                np.mean(
                    expit(X_val.dot(bandit2.weights.T))[np.arange(len(X_val)), arms2]
                )
            )
    print("Final validation error with noise scaling:", val_error1[-1])
    print("Final validation error without noise scaling:", val_error2[-1])
    import matplotlib.pyplot as plt

    plt.plot(val_error1, label="Logistic model")
    plt.plot(val_error2, label="Linear model")
    plt.xlabel("Update steps")
    plt.ylabel("Probability of selecting wrong arm")
    plt.legend()
    plt.show()