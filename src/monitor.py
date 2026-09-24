import numpy as np
from src.utils import Weather, ContextSpace, is_pareto_efficient_simple
from scipy.special import expit

class Monitor:

    def __init__(self, safety_monitor, estimate_rewards_path, threshold_safety, contextSpace):
        """
        Initializes a Monitor object.

        Parameters:
        - safety_monitor: The safety monitor for the system. Shape: C x dim(X), where dim(X) is the number of features of each context X.
        - estimate_rewards_path: The path to the estimate rewards. Shape: C x X x K, where C is the number of controllers, X is the number of contexts, and K is the number of objectives.
        - threshold_safety: The safety threshold for the system.
        - contextSpace: The contexts for the system given by ContextSpace
        """
        self.safety_monitor = np.load(safety_monitor)
        self.estimate_rewards = np.load(estimate_rewards_path)
        self.threshold_safety = threshold_safety
        self.contextSpace = contextSpace

        self.num_controllers = self.safety_monitor.shape[0]

    def safe_controllers(self, context):
        """
        Get the safe controllers for a given context.

        Parameters:
        - context: The context for the system.

        Returns:
        - The safe controllers for the given context.
        """
        (weather, dist_car, speed) = context

        context = np.concatenate([np.array(Weather[weather].value), np.array([dist_car]), np.array([speed]), np.array([1.])])
        safety_probs = expit(np.dot(self.safety_monitor, context))
        safe_controllers = [i for i in range(self.num_controllers) if safety_probs[i] >= self.threshold_safety]
        return {i: safety_probs[i] for i in safe_controllers}

    def safe_pareto_front(self, context):
        """
        Get the safe pareto front for a given context.

        Parameters:
        - context: The context for the system.

        Returns:
        - The safe pareto front for the given context. The output is a dictionary of pairs (controller_index: rewards)
        """
        context_index = self.contextSpace.index(context)
        
        safe_controllers_dict = self.safe_controllers(context)
        safe_controllers = list(safe_controllers_dict.keys())

        if len(safe_controllers) == 0:
            return {}

        safe_controllers_rewards = np.zeros(self.estimate_rewards["mu_hat"][:,context_index,:].shape)
        safe_controllers_rewards[safe_controllers] = self.estimate_rewards["mu_hat"][safe_controllers, context_index, :]

        safe_pareto_front = is_pareto_efficient_simple(safe_controllers_rewards)

        pareto_safe_controllers = [i for i in range(len(safe_pareto_front)) if safe_pareto_front[i]]
        pareto_safe_rewards = safe_controllers_rewards[safe_pareto_front, :].tolist()

        return dict(zip(pareto_safe_controllers, pareto_safe_rewards))

    def optimal_controller(self, context, weights):
        """
        Get the optimal controller for a given context and weights.

        Parameters:
        - context: The context for the system.
        - weights: The weights for the objectives.

        Returns:
        - The optimal controller for the given context and weights.
        """
        pareto_safe_controllers_dict = self.safe_pareto_front(context)
        if pareto_safe_controllers_dict == {}:
            return None
        
        pareto_safe_controllers = list(pareto_safe_controllers_dict.keys())
        pareto_safe_rewards = np.array(list(pareto_safe_controllers_dict.values()))


        weighted_rewards = np.dot(pareto_safe_rewards, weights)
        optimal_index = np.argmax(weighted_rewards)

        return pareto_safe_controllers[optimal_index]


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