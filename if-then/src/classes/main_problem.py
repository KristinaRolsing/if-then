import copy
import random
import statistics
from contextlib import redirect_stdout

import numpy as np
import pyomo.environ as pyo


class MainProblem:
    def __init__(self):
        self.settings = {}
        self.data_dir = ""
        self.file_name = ""
        self.dim = {}
        self.M = np.array([])
        self.objective_vector = []
        self.model = pyo.ConcreteModel()
        self.solution_history = []
        self.BP_solution = {}
        self.binary_solution = False

    def build_instance(self, data_dir: str, settings: dict, file_name: str, dim: dict, M):
        self.settings = settings
        self.data_dir = data_dir
        self.file_name = file_name
        self.dim = dim
        self.M = M

    def build_model(self):
        self._add_variables()
        self._add_objective()
        self._add_constraints()

    def _add_variables(self):
        """
        Create a continuous variable for each element in all if-sets and the then-set.
        """
        for name, size in self.dim.items():
            setattr(self.model, name, pyo.Var([i for i in range(size)], bounds=(0, 1)))

    def _add_objective(self):
        random.seed(self.settings["random_seed"])

        if self.settings["main_objective"] == "normal":
            self.model.objective = pyo.Objective(rule=self._normal_objective(), sense=pyo.maximize)
        elif self.settings["main_objective"] == "uniform":
            self.model.objective = pyo.Objective(rule=self._uniform_objective(), sense=pyo.maximize)
        elif self.settings["main_objective"] == "uniform_b_computed":
            self.model.objective = pyo.Objective(rule=self._uniform_b_computed_objective(), sense=pyo.maximize)

    def _normal_objective(self):
        obj_expr = 0
        obj_coeff = []
        for name, size in self.dim.items():
            var = getattr(self.model, name)
            current_coeff = [random.gauss(0, 1) for i in range(size)]
            obj_coeff += current_coeff
            obj_expr += sum(current_coeff[i] * var[i] for i in range(size))
        self.objective_vector = obj_coeff
        return obj_expr

    def _uniform_objective(self):
        obj_expr = 0
        obj_coeff = []
        for name, size in self.dim.items():
            var = getattr(self.model, name)
            current_coeff = [random.uniform(-1, 1) for i in range(size)]
            obj_coeff += current_coeff
            obj_expr += sum(current_coeff[i] * var[i] for i in range(size))
        self.objective_vector = obj_coeff
        return obj_expr

    def _uniform_b_computed_objective(self):
        obj_expr = 0
        random_coefficients = {}
        if_set_dict = {key: value for key, value in self.dim.items() if key.startswith('if-set')}
        for name, size in if_set_dict.items():
            var = getattr(self.model, name)
            coefficients = [random.uniform(-1, 1) for i in range(size)]
            random_coefficients[name] = coefficients
            obj_expr += sum(coefficients[i] * var[i] for i in range(size))

        if_set_keys = list(random_coefficients.keys())
        b_coefficients = {i: [] for i in range(self.dim["then-set"])}
        tensor_dims = [self.dim[key] for key in list(self.dim.keys())[:-1]]

        for idx_tuple in np.ndindex(*tensor_dims[:]):
            b_index = int(self.M[idx_tuple] - 1)
            total_sum = sum(random_coefficients[if_set_keys[i]][idx_tuple[i]] for i in range(len(idx_tuple)))
            b_coefficients[b_index] += [total_sum]

        b_mean_values = [statistics.mean(value) for value in b_coefficients.values()]
        var = getattr(self.model, "then-set")
        obj_expr += sum(b_mean_values[i] * var[i] for i in range(self.dim["then-set"]))
        return obj_expr

    def _add_constraints(self):
        self.model.constraints = pyo.ConstraintList()
        self._add_mc_cormick_constraints()
        self._add_multiple_choice_constraints()

    def _add_mc_cormick_constraints(self):
        set_names = list(self.dim.keys())
        tensor_dims = [self.dim[key] for key in set_names[:-1]]

        for idx_tuple in np.ndindex(*tensor_dims[:]):
            l = int(self.M[idx_tuple] - 1)
            if_vars_sum = sum(getattr(self.model, key)[i] for key, i in zip(set_names, idx_tuple))
            then_var = getattr(self.model, "then-set")[l]
            self.model.constraints.add(expr=if_vars_sum <= then_var + len(self.dim) - 2)

    def _add_multiple_choice_constraints(self):
        """
        Add multiple choice constraints, ensuring exactly one variable in each if-set and then-set is selected.
        """
        for name, size in self.dim.items():
            var = getattr(self.model, name)
            self.model.constraints.add(expr=sum(var[i] for i in range(size)) == 1)

    def optimize(self):
        if self.settings["solver"] == "scip":
            solv_fact_args = ({"executable": self.settings["scip_exe"]})
            solver = pyo.SolverFactory("scip", **solv_fact_args)
        elif self.settings["solver"] == "gurobi":
            solver = pyo.SolverFactory("gurobi", warmstart=True)

        if self.settings["write_solver_output"]:
            with open(f"{self.data_dir}solutions/{self.file_name}_{self.settings["solver"]}_main.txt", 'a') as f:
                with redirect_stdout(f):
                    result = solver.solve(self.model, tee=self.settings["write_solver_output"])
        else:
            result = solver.solve(self.model, tee=self.settings["write_solver_output"])

        solution_dict = {"solving_time": result.solver.time}
        solution_dict.update({name: [getattr(self.model, name)[i].value for i in range(size)]
                              for name, size in self.dim.items()})
        if self.settings["compute_stats"]:
            solution_dict["solver_message"] = result.solver.message

        self.solution_history.append(solution_dict)
        self._check_if_sol_binary()

    def optimize_as_BP(self):
        # copy model and make all vars binary
        binary_model = copy.deepcopy(self.model)
        for var in binary_model.component_data_objects(ctype=pyo.Var):
            var.domain = pyo.Binary

        if self.settings["solver"] == "scip":
            solv_fact_args = ({"executable": self.settings["scip_exe"]})
            solver = pyo.SolverFactory("scip", **solv_fact_args)
        elif self.settings["solver"] == "gurobi":
            solver = pyo.SolverFactory("gurobi")

        if self.settings["write_solver_output"]:
            with open(f"{self.data_dir}solutions/{self.file_name}_{self.settings["solver"]}_BP.txt", 'w') as f:
                with redirect_stdout(f):
                    result = solver.solve(binary_model, tee=self.settings["write_solver_output"])
        else:
            result = solver.solve(binary_model, tee=self.settings["write_solver_output"])

        solution_dict = {"solving_time": result.solver.time}
        if self.settings["compute_stats"]:
            solution_dict["solver_message"] = result.solver.message
            solution_dict.update({name: [getattr(binary_model, name)[i].value for i in range(size)]
                                  for name, size in self.dim.items()})

        self.BP_solution = solution_dict

    def check_solutions(self):
        """
        Check whether the found solution using if-then cuts is the same as the solution of the BP problem
        """
        if not self.settings["solve_BP_also"] or not self.settings["compute_stats"]:
            return None
        BP_solution = {k: v for k, v in self.BP_solution.items() if k.startswith('if') or k.startswith('then')}
        main_solution = {k: v for k, v in self.solution_history[-1].items() if
                         k.startswith('if') or k.startswith('then')}

        for key in main_solution.keys():
            if not all(abs(a - b) < self.settings["tolerance"] for a, b in zip(BP_solution[key], main_solution[key])):
                return False
        return True

    def _check_if_sol_binary(self):
        vars_are_binary = True
        for var in self.model.component_objects(pyo.Var, active=True):
            for index in var:
                if not (abs(var[index].value - 0) <= self.settings["tolerance"] or abs(var[index].value - 1) <=
                        self.settings["tolerance"]):
                    vars_are_binary = False
                    break
        self.binary_solution = vars_are_binary

    def get_average_solving_time(self):
        all_times = [solution["solving_time"] for solution in self.solution_history]
        return statistics.mean(all_times) if all_times else 0

    def get_sum_solving_time(self):
        all_times = [solution["solving_time"] for solution in self.solution_history]
        return sum(all_times) if all_times else 0

    def get_BP_solving_time(self):
        if not self.settings["solve_BP_also"]:
            return None
        return self.BP_solution["solving_time"]
