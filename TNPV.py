import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# Class to calculate TNPV, risk measures and check the constraints, 
class TNPV:

    def __init__(
        self, 
        projects, # Projects data
        risk_weights, # Risk weights for all risk types
        risk_max=100,
        st_min=0.2,
        wacc=0.1,
        times=np.array(range(1, 21)), # Portfolio timeline
        cash_inflow_indices = np.array(range(1, 11)), # Cash inflow timestamps
        inv_stage=np.array([175] * 4), # Budget for each investment stage
        stages=np.arange(1, 21).reshape((4, 5)) # Timelines for each stage
    ) -> None:

        self.projects = projects
        self.risk_weights = risk_weights
        self.risk_max = risk_max
        self.st_min = st_min
        self.wacc = wacc
        self.times = times
        self.cash_inflow_indices = cash_inflow_indices
        self.inv_stage = inv_stage
        self.stages = stages

    # Calculate the total NPV given the decisions and projects cash inflows
    def total_npv(self, decisions, projects_cash_inflows):
        projects_start_times = decisions @ self.times # get starting times (if any)
        projects_start_times_discount = (1 + self.wacc) ** projects_start_times # find discount amount because of late starting time
        projects_cash_inflows_discounts = projects_start_times_discount.reshape(-1, 1) @ \
                                          ((1 + self.wacc) ** self.cash_inflow_indices).reshape(1, -1) # add additional discount amount 
                                                                                                       # because of late cash inflows 
        
        projects_discounted_cash_inflows = projects_cash_inflows / projects_cash_inflows_discounts # discount cash inflows
        costs = self.projects['c'].values / projects_start_times_discount # discount costs

        npv = costs + np.sum(projects_discounted_cash_inflows, axis=1) # calculate NPV for each project
        project_overall_decision = np.sum(decisions, axis=1) # find selected projects
        tnpv = npv @ project_overall_decision # calculate total NPV

        return tnpv
    
    # Check the once selection constraint
    def once_selection_constraint(self, decisions):
        project_overall_decision = np.sum(decisions, axis=1) # find selected projects
        constraint_satisfaction = not np.any(project_overall_decision > 1) # check if any project is selected more than once

        return constraint_satisfaction
    
    # Check the budget constraint for stage 1
    def budget_constraint_stage_1(self, decisions, return_value=False):
        costs = np.abs(self.projects['c'].values) # get absolute costs
        remaining_budget = 0 # initialize unutilized investment overflowing to the next stage
        
        for i in range(1):
            stage = self.stages[i]
            stage_decisions = decisions[:, stage - 1] # get decisions at that stage
            stage_cost = np.sum(costs.reshape(1, -1) @ stage_decisions) # calculate the stage cost

            remaining_budget = remaining_budget + self.inv_stage[i] - stage_cost # find unutilized investment overflowing to the next stage

        if return_value: 
            return remaining_budget

        constraint_satisfaction = remaining_budget >= 0 # check if the remaining budget is not negative

        return constraint_satisfaction
    
    # Check the budget constraint for stage 2
    def budget_constraint_stage_2(self, decisions, return_value=False):
        costs = np.abs(self.projects['c'].values) # get absolute costs
        remaining_budget = 0 # initialize unutilized investment overflowing to the next stage
        
        for i in range(2):
            stage = self.stages[i]
            stage_decisions = decisions[:, stage - 1] # get decisions at that stage
            stage_cost = np.sum(costs.reshape(1, -1) @ stage_decisions) # calculate the stage cost

            remaining_budget = remaining_budget + self.inv_stage[i] - stage_cost # find unutilized investment overflowing to the next stage

        if return_value: 
            return remaining_budget

        constraint_satisfaction = remaining_budget >= 0 # check if the remaining budget is not negative

        return constraint_satisfaction
    
    # Check the budget constraint for stage 3
    def budget_constraint_stage_3(self, decisions, return_value=False):
        costs = np.abs(self.projects['c'].values) # get absolute costs
        remaining_budget = 0 # initialize unutilized investment overflowing to the next stage
        
        for i in range(3):
            stage = self.stages[i]
            stage_decisions = decisions[:, stage - 1] # get decisions at that stage
            stage_cost = np.sum(costs.reshape(1, -1) @ stage_decisions) # calculate the stage cost

            remaining_budget = remaining_budget + self.inv_stage[i] - stage_cost # find unutilized investment overflowing to the next stage

        if return_value: 
            return remaining_budget

        constraint_satisfaction = remaining_budget >= 0 # check if the remaining budget is not negative

        return constraint_satisfaction
    
    # Check the budget constraint for stage 4
    def budget_constraint_stage_4(self, decisions, return_value=False):
        costs = np.abs(self.projects['c'].values) # get absolute costs
        remaining_budget = 0 # initialize unutilized investment overflowing to the next stage
        
        for i in range(4):
            stage = self.stages[i]
            stage_decisions = decisions[:, stage - 1] # get decisions at that stage
            stage_cost = np.sum(costs.reshape(1, -1) @ stage_decisions) # calculate the stage cost

            remaining_budget = remaining_budget + self.inv_stage[i] - stage_cost # find unutilized investment overflowing to the next stage

        if return_value: 
            return remaining_budget

        constraint_satisfaction = remaining_budget >= 0 # check if the remaining budget is not negative

        return constraint_satisfaction

    # Check the budget constraint 
    def budget_constraint(self, decisions):
        costs = np.abs(self.projects['c'].values) # get absolute costs
        remaining_budget = 0 # initialize unutilized investment overflowing to the next stage
        
        for i in range(len(self.stages)):
            stage = self.stages[i]
            stage_decisions = decisions[:, stage - 1] # get decisions at that stage
            stage_cost = np.sum(costs.reshape(1, -1) @ stage_decisions) # calculate the stage cost

            if stage_cost > self.inv_stage[i] + remaining_budget: # check if we have sufficient budget

                return False
            else:

                remaining_budget = remaining_budget + self.inv_stage[i] - stage_cost # find unutilized investment overflowing to the next stage

        return True

    # Check the mandatory project selection constraint 
    def mandatory_projects_constraint(self, decisions):
        project_overall_decision = np.sum(decisions, axis=1) # find selected projects
        constraint_satisfaction = np.sum(project_overall_decision[[5, 7]]) == 2 # check if the mandatory projects are selected

        return constraint_satisfaction
    
    # Check the strategic scores constraint 
    def strategic_scores_constraint(self, decisions, return_value=False):
        strategic_scores = self.projects['st_sc'].values # find the strategic scores
        total_strategic_score = strategic_scores @ np.sum(decisions, axis=1) # calculate the overall strategic score

        if return_value: 
            return total_strategic_score - self.st_min

        constraint_satisfaction = total_strategic_score >= self.st_min # check if the overall strategic score is sufficient

        return constraint_satisfaction

    # Check the maximum tolerated risk constraint
    def maximum_risk_constraint(self, decisions, return_value=False):
        project_risks = self.projects[['Tech_risk', 'Sch_risk', 'EP_risk', 'Org_risk', 'Stc_risk']].values # find the risk values
        total_risks = np.sum(decisions.T @ project_risks, axis=0) # calculate the overall risk for each risk
        final_risk = self.risk_weights @ total_risks # calculate the weighted overall risk

        if return_value: 
            return self.risk_max - final_risk

        constraint_satisfaction = final_risk <= self.risk_max # check if the overall risk is tolerated

        return constraint_satisfaction
    
    # Check all constraints
    def all_constraints(self, decisions):
        constraint_satisfaction = self.once_selection_constraint(decisions) and \
                                  self.budget_constraint(decisions) and \
                                  self.mandatory_projects_constraint(decisions) and \
                                  self.strategic_scores_constraint(decisions) and \
                                  self.maximum_risk_constraint(decisions) 
        
        return constraint_satisfaction
    
    # Calculate the TNPV list for all simulated cash inflows
    def all_tnpv(self, decisions, projects_cash_inflows):
        tnpvs = pd.Series(projects_cash_inflows).apply(lambda x: self.total_npv(decisions, x)) # calculate the TNPV list

        return tnpvs

    # Calculate the Expected TNPV given the simulated cash inflows
    def expected_tnpv(self, decisions, projects_cash_inflows):
        tnpvs = self.all_tnpv(decisions, projects_cash_inflows) # get the TNPV list
        mean_tnpv = tnpvs.mean() # calculate the mean

        return mean_tnpv
    
    # Calculate the CVaR for TNPV given the simulated cash inflows
    def cvar_tnpv(self, decisions, projects_cash_inflows, alpha=0.95):
        tnpvs = self.all_tnpv(decisions, projects_cash_inflows) # get the TNPV list
        cvar_tnpv_ = tnpvs.sort_values().head(int((1 - alpha) * len(tnpvs))).mean() # calculate the CVaR

        return cvar_tnpv_
    
    # Calculate the VaR for TNPV given the simulated cash inflows
    def var_tnpv(self, decisions, projects_cash_inflows, alpha=0.95):
        tnpvs = self.all_tnpv(decisions, projects_cash_inflows) # get the TNPV list
        var_tnpv_ = tnpvs.sort_values().head(int((1 - alpha) * len(tnpvs))).iloc[-1] # calculate the VaR

        return var_tnpv_
    
    # Calculate the weighted average of CVaR and Expectation for TNPV given the simulated cash inflows
    def weighted_cvar_expected_tnpv(self, decisions, projects_cash_inflows, alpha=0.95, lambda_=0.5):
        tnpvs = self.all_tnpv(decisions, projects_cash_inflows) # get the TNPV list
        mean_tnpv = tnpvs.mean() # calculate the mean
        cvar_tnpv_ = tnpvs.sort_values().head(int((1 - alpha) * len(tnpvs))).mean() # calculate the CVaR

        weighted_cvar_expected_tnpv = lambda_ * mean_tnpv + (1 - lambda_) * cvar_tnpv_ # calculate the weighted average

        return weighted_cvar_expected_tnpv
    