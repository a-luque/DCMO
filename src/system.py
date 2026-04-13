import numpy as np


class System:

    def __init__(self, controllers, scenic, contexts):
        """
        Initializes a System object.

        Parameters:
        - controllers: The controllers for the system.
        - scenic: The scenic object for the system.
        """
        self.controllers = controllers
        self.scenic = scenic
        self.contexts = contexts

    def step(self, index, x):
        """
        Execute the controller corresponding index in the system with context x.


        Parameters:
        - index: The index to be executed.
        - x: The context for the system.

        Returns:
        - The reward for the executed index.
        """

        pass

    def sample_context(self):
        """
        Sample a context for the system.

        Sample a vector from scenic and return it. (Initial configuration of the system)

        Returns:
        - The context for the system.
        """
        pass


from scipy.special import expit


class ToyScenic:
    """
    Toy example of a Scenic object.
    """

    def __init__(self):
        """
        Initializes a ToyScenic object with 3 controllers
        """
        self.controllers = [0, 1, 2]

        # weights are not available in practice but are used here for simplicity
        self.weights = np.array([[1, 0.5, 1], [-1, 1, 0], [-0.5, 0.1, 1]])

    def simulate(self, index, x):
        """
        Simulate the system with the given index and context x.

        Parameters:
        - index: The index of the controller to be executed.
        - x: The context for the system.

        Returns:
        - The reward for the executed index.
        """
        # this function should run a simulation and return the reward
        p = expit(np.dot(x, self.weights[index]))
        return np.random.binomial(1, p)


class ToySystem:
    """
    Toy example of a System object.
    """

    def __init__(self, scenic):
        """
        Initializes a ToySystem object.

        Parameters:
        - scenic: The scenic object for the system.
        """
        self.scenic = scenic
        self.contexts = np.random.randn(100, 3)

    def step(self, index, x):
        """
        Execute the controller corresponding index in the system with context x.


        Parameters:
        - index: The index to be executed.
        - x: The context for the system.

        Returns:
        - The reward for the executed index.
        """
        return self.scenic.simulate(index, x)

    def sample_context(self):
        """
        Sample a context for the system.

        Sample a vector from scenic and return it. (Initial configuration of the system)

        Returns:
        - The context for the system.
        """
        context = self.contexts[np.random.randint(0, 100)]
        return context