import os
import sys
import inspect
import numpy as np

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from src.util.constants import *
C = 2*C

from src.util.distance_metrics import Similarity

# Need: SxC boolean masking matrix. Init False, flip entries to True as questions are being asked.
#q_tracker = np.zeros((S,C), dtype=bool)
#q_tracker = np.zeros((S,C), dtype=int)
def question_selector(u_tilde, questions, involvements, student_n, q_tracker):
    # Library of different similarity metrics to explore
    similarity_metrics = Similarity(minimum=1e-20)

    # Keeping track of best match and its distance to the optimal input
    min_diff = np.inf
    best_match = None
    best_match_idx = None

    # Brute force find the best matching question to the optimal one.
    for idx, q in enumerate(questions):
        '''
            On the Surprising Behavior of Distance Metrics in High Dimensional Space
            https://bib.dbvis.de/uploadedFiles/155.pdf
        '''
        # Calculate the question's similarity to the optimal solution:
        q_diff = similarity_metrics.fractional_distance(q, u_tilde, fraction=0.5)
        #q_diff = similarity_metrics.euclidean_distance(q, u_tilde)

        # Hold on to the temporary best match
        if q_diff < min_diff and not q_tracker[0,idx]:
            min_diff = q_diff
            best_match = q
            best_match_idx = idx

    # Make this question "spent":
    q_tracker[student_n,best_match_idx] = True
    #q_tracker[best_match_idx] += 1
    w = involvements[best_match_idx]

    return w, best_match.reshape(u_tilde.shape).squeeze(), q_tracker


def benchmark1(psi):
    return psi[np.random.randint(C, size=1), :].reshape(K, -1)

def benchmark2(psi, q_previous):
    #max_difficulty = (1 - x0) / 2 * np.random.normal(1, 0.1)
    for q in psi:
        if np.sum(q ** 2) > np.sum(q_previous ** 2):
            u0 = q.reshape(K, -1)
            return u0

    # If there's no harder question, use the previous one:
    return q_previous
