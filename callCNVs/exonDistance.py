import logging
import numpy


# set up logger, using inherited config
logger = logging.getLogger(__name__)


###############################################################################
############################ PUBLIC FUNCTIONS #################################
###############################################################################
#####################
# create_trans_func
# Creates a function to dynamically adjust transition probabilities based on distance
# for a given state. The function implements an exponential model to modify the transition
# probability to the CN2 state (z) when the distance exceeds dmax, up to a maximum probability
# adjustment of fmax. This adjustment reflects decreased correlation as the distance increases.
#
# Args:
# - z (float): Initial transition probability to the CN2 state from the previous state.
# - alpha (float): Coefficient for exponential growth of f(d), calculated to ensure that f(d)
#                  reaches fmax at the maximum considered distance (dmax).
# - prevState (int): Index of the previous state in the transition matrix.
# - transMatrix (np.array): Matrix containing the base transition probabilities between states.
# - dmax (float): Threshold distance beyond which the probability begins to adjust exponentially.
# - fmax (float): Maximum value of the adjusted transition probability to CN2, ensuring f(d) does
#                 not grow unbounded and maintains realistic biological correlation.
#
# Returns:
# - function: A function that computes adjusted transition probabilities based on the input distance d.
#             It maintains overall transition probability normalization while specifically adjusting
#             for the CN2 state exponentially and others linearly to ensure total probabilities sum to 1.
def create_trans_func(z, alpha, prevState, transMatrix, dmax, fmax):
    def trans_func(d):
        # Determine f(d) based on the distance d
        if d <= dmax:
            # f(d) remains 1 as long as d is less than or equal to dmax
            f_d = 1
            g_d = 1  # g(d) remains 1 to maintain original transition probabilities
        else:
            # Exponential calculation of f(d) for d > dmax
            f_d = numpy.exp(alpha * (d - dmax))
            # Limit f(d) to fmax to prevent excessive growth
            f_d = min(f_d, fmax)

        # Calculate g(d) to adjust transitions towards other states
        if d > dmax:
            # Adjust g(d) to ensure the sum of probabilities remains 1
            g_d = (1 - z * f_d) / (1 - z) if z < 1 else 0

        # Construct the adjusted transition vector
        transitions = numpy.zeros(len(transMatrix[prevState]))
        # Specific transition to CN2
        transitions[2] = f_d * z
        for state in range(len(transitions)):
            # Apply g(d) to the other states
            if state != 2:
                transitions[state] = g_d * transMatrix[prevState][state]

        return transitions

    return trans_func


########################
# initTransFunc
#  Initializes a list of dynamic transition functions for each state in a Hidden Markov Model,
# using a specified maximum distance and a prior probability for state CN2 to calculate adjustments.
# These functions account for potential changes in correlation due to varying distances between sequence events.
#
# Args:
# - transMatrix (np.array): Matrix of base transition probabilities between states.
# - priorCN2 (float): Prior probability for transitioning to the CN2 state.
# - dmax (float): Distance threshold for activating exponential probability adjustments.
#
# Returns:
# - list of functions: Each function calculates transition probabilities dynamically based on distance,
#                         ensuring that transitions are adjusted to reflect both exponential and linear shifts
#                         in probabilities based on the biological relevance of distance.
def initTransFunc(transMatrix, priorCN2, dmax=18000):

    num_states = transMatrix.shape[0]
    transFuncs = []

    for prevState in range(num_states):
        # Extract z, the transition probability to CN2
        z = transMatrix[prevState, 2]
        # Calculate fmax as priorCN2 divided by z
        fmax = priorCN2 / z
        # Compute alpha to control the growth of f(d)
        alpha = numpy.log(fmax) / dmax

        # Add the state-specific transition function to the list
        transFuncs.append(create_trans_func(z, alpha, prevState, transMatrix, dmax, fmax))

    return transFuncs
