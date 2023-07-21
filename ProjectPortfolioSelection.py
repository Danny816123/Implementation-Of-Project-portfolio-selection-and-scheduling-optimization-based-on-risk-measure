import numpy as np
from scipy.stats import qmc, norm
import warnings
import random
import itertools
import os
from joblib import Parallel, delayed
from scipy.optimize import minimize
from scipy.optimize import Bounds
from scipy.optimize import NonlinearConstraint
from tqdm import tqdm
from TNPV import *
warnings.filterwarnings("ignore")

# Generate random sample of decision variables
def generate_random_decision_variables(seed=None):
    random.seed(seed)
    projects = [] # initialize projects decisions
    for i in range(20):
        if i in [5, 7]: # consider the mandatory projects
            random_number = random.randrange(0, 20) # choose start time for each project
        else:
            random_number = random.randrange(0, 21) # choose start time for each project

        project = np.zeros(20) # initialize project decision

        if random_number < 20:
            project[random_number] = 1 # if a project is selected, its start time index has value 1

        projects.append(project) 

    projects = np.vstack(projects)  

    return projects

# Sample cash inflows at each state using Latin Hypercube Sampling and Normal distribution
def normal_latin_hypercube_monte_carlo(
    project_means, 
    project_stds, 
    n_samples=200, 
    n_jobs=1, 
    backend='loky'
):
    os.environ['JOBLIB_TEMP_FOLDER'] = '/tmp'

    def parallel_job(sample): # parallelize the sampling
        cash_inflow = np.zeros(project_means.shape)
        for i, j in itertools.product(range(project_means.shape[0]), range(project_means.shape[1])):
            project_std = project_stds[i, j]
            if project_std > 0:
                cash_inflow[i, j] = norm(loc=project_means[i, j], scale=project_stds[i, j]).ppf(sample) # use the provided distribution to sample cash inflow
            else:    
                cash_inflow[i, j] = project_means[i, j] # if std. is zero, use the mean

        return cash_inflow         
    
    cash_inflows = []
    lhd = qmc.LatinHypercube(d=1, seed=1).random(n=n_samples) # sample numbers in [0, 1] to obtain final samples from the ppf distribution

    cash_inflows = Parallel(n_jobs=n_jobs, backend=backend)(delayed(parallel_job)(sample) # find cash inflows
                                                           for sample in lhd)

    return cash_inflows   

# Convert project start time array to the decision matrix
def convert_result_to_decisions(project_decisions):
    decisions = [] # initialize projects decisions

    for random_number in project_decisions:
        decision = np.zeros(20) # initialize project decision

        if int(random_number) > 0:
            decision[int(random_number) - 1] = 1 # if a project is selected, its start time index has value 1

        decisions.append(decision) 

    decisions = np.vstack(decisions)

    return decisions

# Optimize the project selection decisions
def optimize_decisions(
    projects, 
    risk_weights, 
    alpha=0.95, 
    lambda_=0.5, 
    n_samples=200, 
    n_jobs=8
):
    project_means = projects.iloc[:, 7:17].values # import project cash inflow distribution means
    project_stds = projects.iloc[:, 17:].values # import project cash inflow distribution stds
    projects_cash_inflows = normal_latin_hypercube_monte_carlo(project_means, project_stds, n_samples, n_jobs) # sample cash inflows

    lower_bounds = 0 * np.ones(20) # set lower bounds for project start times
    upper_bounds = 20 * np.ones(20) # set upper bounds for project start times
    lower_bounds[5] = 1 # correct for mandatory projects
    lower_bounds[7] = 1 # correct for mandatory projects
    bounds = Bounds(lower_bounds, upper_bounds)

    npv = TNPV(projects, risk_weights) # create the TNPV object for further calculations

    # create budget non linear constraint for stage 1
    def non_linear_budget_constraint_stage_1(project_decisions):
        decisions = convert_result_to_decisions(project_decisions)

        return npv.budget_constraint_stage_1(decisions, True)

    # create budget non linear constraint for stage 2
    def non_linear_budget_constraint_stage_2(project_decisions):
        decisions = convert_result_to_decisions(project_decisions)

        return npv.budget_constraint_stage_2(decisions, True)

    # create budget non linear constraint for stage 3
    def non_linear_budget_constraint_stage_3(project_decisions):
        decisions = convert_result_to_decisions(project_decisions)

        return npv.budget_constraint_stage_3(decisions, True)

    # create budget non linear constraint for stage 4
    def non_linear_budget_constraint_stage_4(project_decisions):
        decisions = convert_result_to_decisions(project_decisions)

        return npv.budget_constraint_stage_4(decisions, True)

    # create strategic scores non linear constraint 
    def non_linear_strategic_scores_constraint(project_decisions):
        decisions = convert_result_to_decisions(project_decisions)

        return npv.strategic_scores_constraint(decisions, True)

    # create risk tolerance non linear constraint 
    def non_linear_maximum_risk_constraint(project_decisions):
        decisions = convert_result_to_decisions(project_decisions)

        return npv.maximum_risk_constraint(decisions, True)
    
    # choose the objective function using the lambda value
    if lambda_ == 1:
        objective = lambda project_decisions: -npv.expected_tnpv(
                                                convert_result_to_decisions(project_decisions), 
                                                projects_cash_inflows
                                              )
    elif lambda_ == 0:
        objective = lambda project_decisions: -npv.cvar_tnpv(
                                                convert_result_to_decisions(project_decisions), 
                                                projects_cash_inflows,
                                                alpha
                                              )
    else:
        objective = lambda project_decisions: -npv.weighted_cvar_expected_tnpv(
                                                convert_result_to_decisions(project_decisions), 
                                                projects_cash_inflows,
                                                alpha,
                                                lambda_
                                              )
        
    # choose a set of initial points
    initial_points = {}

    initial_points[1] = [0, 3, 0, 0, 0, 20, 1, 20, 0, 0, 0, 18, 11, 0, 6, 0, 12, 0, 0, 0]
    initial_points[2] = [0, 11, 0, 1, 0, 20, 3, 20, 0, 0, 0, 0, 6, 0, 17, 1, 11, 0, 0, 0]
    initial_points[3] = [0, 8, 0, 1, 0, 20, 4, 20, 0, 0, 0, 1, 11, 0, 0, 6, 11, 0, 16, 0]
    initial_points[4] = [0, 1, 0, 13, 0, 20, 11, 20, 0, 0, 0, 6, 6, 0, 0, 17, 2, 0, 11, 0]
    initial_points[5] = [0, 1, 0, 1, 0, 20, 0, 20, 0, 0, 0, 3, 11, 0, 16, 0, 6, 0, 0, 0]
    initial_points[6] = [0, 1, 0, 0, 0, 20, 1, 20, 0, 0, 0, 1, 16, 0, 11, 0, 16, 0, 0, 0]   

    class MyCallback:
        best_val = np.inf
        best_x = None
        def __call__(self, intermediate_result) -> None:
            x = intermediate_result.x
            fval = intermediate_result.fun
            if fval < self.best_val:
                self.best_val = fval
                self.best_x = x

    # create the callback function to store the optimal value
    my_cb_object = MyCallback()
    callback = lambda xk, intermediate_result: my_cb_object(intermediate_result)

    # run the optimization with the defined constraints and all possible initial points
    for x00 in tqdm(initial_points.values()):
        result = minimize(objective, x00, method='trust-constr', 
                    constraints=[NonlinearConstraint(non_linear_budget_constraint_stage_1, 0, np.inf, keep_feasible=True), 
                                 NonlinearConstraint(non_linear_budget_constraint_stage_2, 0, np.inf, keep_feasible=True),
                                 NonlinearConstraint(non_linear_budget_constraint_stage_3, 0, np.inf, keep_feasible=True),
                                 NonlinearConstraint(non_linear_budget_constraint_stage_4, 0, np.inf, keep_feasible=True),
                                 NonlinearConstraint(non_linear_strategic_scores_constraint, 0, np.inf, keep_feasible=True),
                                 NonlinearConstraint(non_linear_maximum_risk_constraint, 0, np.inf, keep_feasible=True)
                                ],          
                    options={'verbose' : 1, 'maxiter' : 1000}, bounds=bounds, callback=callback) 
    
    return my_cb_object.best_x
