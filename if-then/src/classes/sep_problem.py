import statistics
from collections import Counter
from contextlib import redirect_stdout

import numpy as np
import pyomo.environ as pyo


def _compute_affrank(vertex_set):
    """
    Compute the rank of the matrix formed by the list of vertex tuples.
    Add rows of ones to have affine linear independence, subtract 1 at the end.
    """
    if len(vertex_set) == 0:
        return 0
    vertices_in_tuples = [tuple(sum((value for value in vertex.values()), [])) for vertex in vertex_set]
    matrix = np.array(vertices_in_tuples).T

    row_of_ones = np.ones((1, matrix.shape[1]))
    matrix = np.vstack([matrix, row_of_ones])

    return int(np.linalg.matrix_rank(matrix)) - 1


class SepProblem:
    def __init__(self):
        self.settings = {}
        self.data_dir = ""
        self.file_name = ""
        self.dim = {}
        self.M = np.array([])
        self.space_dim = 0
        self.facet_dim = 0
        self.vertices = []
        self.vertex_rank = 0
        self.is_full_dim = False
        self.model = pyo.ConcreteModel()
        self.solution_history = []
        self.redundant = False

    def build_instance(self, data_dir: str, settings: dict, file_name: str, dim: dict, M):
        self.settings = settings
        self.data_dir = data_dir
        self.file_name = file_name
        self.dim = dim
        self.M = M
        self.space_dim = sum(self.dim.values())
        self.facet_dim = self.space_dim - len(self.dim) - 1
        self._compute_vertices()
        self.vertex_rank = _compute_affrank(self.vertices)
        self.is_full_dim = self.space_dim - len(self.dim) == self.vertex_rank

    def _compute_vertices(self):
        set_ranges = [self.dim[if_set_name] for if_set_name in self.dim.keys()]
        for idx_tuple in np.ndindex(*set_ranges[:-1]):
            l = self.M[idx_tuple] - 1
            vertex_dict = {dim: [1 if idx == i else 0 for idx in range(size)]
                           for dim, size, i in zip(self.dim.keys(), self.dim.values(), idx_tuple)}
            vertex_dict["then-set"] = [1 if idx == l else 0 for idx in range(self.dim["then-set"])]
            self.vertices.append(vertex_dict)

    def build_model(self):
        self._add_variables()
        self._add_constraints()

    def _add_variables(self):
        for name, size in self.dim.items():
            if 'if-set' in name:
                setattr(self.model, name, pyo.Var([i for i in range(size)], bounds=(0, float('inf'))))
            if 'then-set' in name:
                setattr(self.model, name, pyo.Var([i for i in range(size)], bounds=(-float('inf'), 1)))

    def build_objective(self, last_solution):
        """
        Define the objective function for the sep problem based on the last solution of the main problem.
        """
        self.model.del_component('value')
        last_solution = self._perturb_solution(last_solution)
        expr = 0
        for if_set_name in list(self.dim.keys())[:-1]:
            var = getattr(self.model, if_set_name)
            expr += sum(var[i] * last_solution[if_set_name][i] for i in range(self.dim[if_set_name]))
        expr -= sum(getattr(self.model, "then-set")[i] * last_solution["then-set"][i] for i in
                    range(self.dim["then-set"]))
        self.model.value = pyo.Objective(expr=expr, sense=pyo.maximize)  # - len(self.dim) + 2, sense=pyo.maximize)

    def _perturb_solution(self, solution):
        """
        Perturb the solution slightly to avoid redundant cuts.
        """
        if not self.settings["perturbation"]:
            return solution

        perturbator = self.settings["tolerance"]
        perturbed_sol = {key: [value + perturbator for value in values]
                         for key, values in solution.items() if key.startswith('if-set')}
        perturbed_sol.update({key: [value + perturbator for value in values]
                              for key, values in solution.items() if key.startswith('then-set')})
        return perturbed_sol

    def _add_constraints(self):
        self.model.constraints = pyo.ConstraintList()
        if_set_names = list(self.dim.keys())[:-1]
        index_ranges = [self.dim[name] for name in if_set_names]

        for idx_tuple in np.ndindex(*index_ranges):
            l = self.M[idx_tuple] - 1
            sum_if_vars = sum(getattr(self.model, name)[idx] for name, idx in zip(if_set_names, idx_tuple))
            self.model.constraints.add(expr=sum_if_vars <= getattr(self.model, "then-set")[l] + len(self.dim) - 2)

    def optimize(self):
        if self.settings["solver"] == "scip":
            solv_fact_args = ({"executable": self.settings["scip_exe"]})
            solver = pyo.SolverFactory("scip", **solv_fact_args)
        elif self.settings["solver"] == "gurobi":
            solver = pyo.SolverFactory("gurobi", warmstart=True)

        if self.settings["write_solver_output"]:
            with open(f"{self.data_dir}solutions/{self.file_name}_{self.settings["solver"]}_sep.txt", 'a') as f:
                with redirect_stdout(f):
                    result = solver.solve(self.model, tee=self.settings["write_solver_output"])
        else:
            result = solver.solve(self.model, tee=self.settings["write_solver_output"])

        assert result.solver.primal_bound > len(list(self.dim.keys())[:-1]) - 1, "PRIMALBOUND not greater than n-1"

        solution_dict = {"solving_time": result.solver.time}
        if self.settings["compute_stats"]:
            solution_dict.update({name: [getattr(self.model, name)[i].value for i in range(self.dim[name])]
                                  for name in self.dim.keys()})

            solution_dict.update({
                "solver_message": result.solver.message,
                "is_facet": self._check_facet(),
                "cut_complexity": self._compute_cut_complexity()
            })
            self._check_loop(solution_dict)

        if not self.redundant:
            self.solution_history.append(solution_dict)

    def _check_facet(self):
        vertices_in_facet = []
        for vertex in self.vertices:
            if self._check_vertex_in_hyperplane(vertex):
                vertices_in_facet += [vertex]
        rank = _compute_affrank(vertices_in_facet)
        return rank == self.vertex_rank - 1

    def _check_vertex_in_hyperplane(self, vertex):
        expr = 0
        for if_set_name in list(self.dim.keys())[:-1]:
            expr += sum(getattr(self.model, if_set_name)[i].value * vertex[if_set_name][i] for i in
                        range(self.dim[if_set_name]))
        expr -= sum(
            getattr(self.model, "then-set")[i].value * vertex["then-set"][i] for i in range(self.dim["then-set"]))
        expr -= len(self.dim) - 2
        return abs(expr) <= self.settings["tolerance"]

    def _compute_cut_complexity(self):
        all_coefficients = [getattr(self.model, set_name)[i].value for set_name in self.dim.keys() for i in
                            range(self.dim[set_name])]
        for i in range(1, 51):
            if self._is_int([j * i for j in all_coefficients]):
                return i
        return ">50"

    def _is_int(self, lst):
        return all(
            isinstance(i, int) or (isinstance(i, float) and abs(i - round(i)) <= self.settings["tolerance"]) for i in
            lst)

    def _check_loop(self, solution_dict):
        """
        Check if the current solution is the same as the last one.
        """
        if self.solution_history:
            self.redundant = solution_dict == self.solution_history[-1]

    def get_facet_quota(self):
        if not self.settings["compute_stats"]:
            return None

        true_count = sum(1 for solution in self.solution_history if solution['is_facet'])
        false_count = sum(1 for solution in self.solution_history if not solution['is_facet'])
        if false_count == 0:
            if true_count > 0:
                return 1
            else:
                return 0
        else:
            return float(true_count) / (false_count + true_count)

    def get_all_cut_complexity(self):
        """
        Return the cut complexity of all solutions.
        """
        if not self.settings["compute_stats"]:
            return None
        return dict(Counter(solution["cut_complexity"] for solution in self.solution_history))

    def get_average_cut_complexity(self):
        if not self.settings["compute_stats"]:
            return None
        cut_complexity = self.get_all_cut_complexity()

        if not cut_complexity: return 0

        weighted_sum = sum(k * v for k, v in cut_complexity.items())
        total_values = sum(cut_complexity.values())
        return weighted_sum / total_values

    def get_average_solving_time(self):
        all_times = [solution["solving_time"] for solution in self.solution_history]
        return statistics.mean(all_times) if all_times else 0

    def get_sum_solving_time(self):
        all_times = [solution["solving_time"] for solution in self.solution_history]
        return sum(all_times) if all_times else 0
