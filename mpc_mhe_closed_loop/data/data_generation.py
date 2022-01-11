import scipy.stats as stats
import numpy as np
import seaborn as sns
import os
import sys
import inspect


currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
# S students, C contents, K skills in:
from src.util.constants import *

# APPROACH 1 -- Generate Students and Questions as independent variables
def data_generation(plot=False):
    lower, upper = 0, 1
    s_mu, s_sigma = 0.25, 0.15  # Mean in 0.25 since assumed low initial skills level
    q_mu1, q_sigma1 = 0.25, 0.15
    q_mu2, q_sigma2 = 0.75, 0.15
    students = stats.truncnorm.rvs((lower-s_mu)/s_sigma,(upper-s_mu)/s_sigma,loc=s_mu,scale=s_sigma,size=(S,K))
    questions = stats.truncnorm.rvs(
        (lower - q_mu1) / q_sigma1, (upper - q_mu1) / q_sigma1,
        loc=q_mu1,
        scale=q_sigma1,
        size=(C, K)
    )
    questions = np.append(questions, stats.truncnorm.rvs(
        (lower - q_mu2) / q_sigma2, (upper - q_mu2) / q_sigma2,
        loc=q_mu2,
        scale=q_sigma2,
        size=(C, K)
        )
    ).reshape(-1, K)

    # Limiting the skill involvements to max 3:
    # h = [nz, nz, nz, nz, nz, nz, nz, nz, nz]
    # --> [nz, 0,  nz, nz, 0,  0,  0,  0,  0]
    #mask = np.random.choice(a=[False, True], size=questions.shape, p=[.65, 1 - .65])
    #questions = np.where(mask, 0, questions)


    bins = np.arange(0, 1.1, 0.1)
    psi = np.array([])
    for q in questions:
        psi = np.append(psi, np.digitize(q,bins,right=True)/10)
    phi = np.array([])
    # for s in students:
    #     phi = np.append(phi, np.digitize(s,bins,right=True)/10)
    # student_norms = np.linalg.norm(students, axis=1)
    # question_norms = np.linalg.norm(questions, axis=1)

    if plot:
        sns.histplot(psi)

    print("data_generation() constructed student data with shape {} and questions data with shape {}".format(students.shape, questions.shape))
    return students, psi.reshape(-1, K), #phi.reshape(-1,K)

# # APPROACH 2 -- Generate Performance Matrix Y (S x C) and use MF to find students, questions
# def data_generation(plot=False):
    # lower, upper = 0, 1
    # p_mu, p_sigma = 0.5, 0.20
    # performance = stats.truncnorm.rvs((lower-p_mu)/p_sigma,(upper-p_mu)/p_sigma,loc=p_mu,scale=p_sigma,size=(S,C))
    # #number_of_nan = 12500 # Half of the students
    # #performance.ravel()[np.random.choice(performance.size, number_of_nan, replace=False)] = np.nan
    # if plot:
    #     sns.heatmap(performance)

    # # Matrix Factorization
    # from sklearn.decomposition import NMF
    # model = NMF(n_components=K, random_state=42, max_iter=10000)
    # students = model.fit_transform(performance)
    # questions = np.transpose(model.components_)

    # return model, students, questions
