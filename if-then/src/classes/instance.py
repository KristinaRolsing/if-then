import time

import numpy as np
from classes.main_problem import MainProblem
from classes.sep_problem import SepProblem


class Instance:
    def __init__(self, data_dir: str, file_name: str, settings: dict):
        self.data_dir = data_dir
        self.file_name = file_name
        self.settings = settings
        self.dim = {}
        self.M = np.array([])  # This will store the multi-dimensional tensor
        self.main_problem = MainProblem()
        self.sep_problem = SepProblem()
        self.enumeration_solution = []
        self.enumeration_solving_time = 0

    def build_instance(self):
        """
        Build the instance by reading the tensor and the settings from the file.
        """
        self._read_input_file(self.data_dir + self.file_name)

    def _read_input_file(self, file_path):
        """
        Read the tensor data from the file and store dimensions and tensor M.
        The file format:
        - First line: space-separated dimensions followed by max value for tensor.
        - Remaining lines: tensor data, separated by empty lines for different layers.

        The dim argument is a dictionary where keys represent the name of if-sets
        and values represent their sizes.
        The last key is then-set and its size.
        Example: dim = {"if-set_1": 3, "if-set_2": 4, "if-set_3": 5, "then-set": 3}
        """
        with open(file_path, "r") as f:
            dimensions_line = f.readline().strip().split()
            *dims, max_value = [int(val) for val in dimensions_line]

            self.M = np.empty(tuple(dims), dtype=int)

            data = f.read().split()
            flat_data = [int(val) for val in data]

            self.M = np.array(flat_data).reshape(tuple(dims))  # Reshape the flat data into the tensor

            dim_keys = [f"if-set_{i}" for i in range(len(dims))]
            self.dim = dict(zip(dim_keys, dims))
            self.dim['then-set'] = max_value

    def build_main_problem(self):
        self.main_problem.build_instance(self.data_dir, self.settings, self.file_name, self.dim, self.M)
        self.main_problem.build_model()

    def optimize_main_problem(self):
        self.main_problem.optimize()

    def optimize_main_problem_as_BP(self):
        self.main_problem.optimize_as_BP()

    def solve_with_enumeration(self):
        objective = np.array(self.main_problem.objective_vector)
        vertices = self.sep_problem.vertices.copy()
        vertices = np.array([tuple(sum((value for value in vertex.values()), [])) for vertex in vertices])
        max_objective_value = 0
        vertex_solution = 0

        start_time = time.time()
        for vertex in vertices:
            scalar_product = np.dot(objective, vertex)
            if scalar_product > max_objective_value:
                max_objective_value = scalar_product
                vertex_solution = vertex
        end_time = time.time()

        self.enumeration_solution = vertex_solution
        self.enumeration_solving_time = end_time - start_time

    def check_same_as_BP_solution(self):
        BP_solution = {k: v for k, v in self.main_problem.BP_solution.items() if
                       k.startswith('if') or k.startswith('then')}
        BP_solution = np.array(tuple(value for values in BP_solution.values() for value in values))
        return np.allclose(BP_solution, self.enumeration_solution)

    def build_sep_problem(self):
        """
        Build the separation problem.
        No objective will be added yet. Objective will be added if add_cut procedure starts.
        """
        self.sep_problem.build_instance(self.data_dir, self.settings, self.file_name, self.dim, self.M)
        self.sep_problem.build_model()

    def add_cut(self):
        """
        Add a cut to the main problem using the solution from the separation problem.
        """
        self.sep_problem.build_objective(self.main_problem.solution_history[-1])
        self.sep_problem.optimize()

        if not self.sep_problem.redundant:
            sum_if_sets = 0
            for if_set_name in list(self.dim.keys())[:-1]:
                sum_if_sets += sum(getattr(self.sep_problem.model, if_set_name)[i].value *
                                   getattr(self.main_problem.model, if_set_name)[i]
                                   for i in range(self.dim[if_set_name]))

            sum_then_set = sum(
                getattr(self.sep_problem.model, "then-set")[i].value * getattr(self.main_problem.model, "then-set")[i]
                for i in range(self.dim["then-set"]))
            self.main_problem.model.constraints.add(expr=sum_if_sets <= sum_then_set + len(self.dim) - 2)

    def get_info_dict(self):
        """
        Return an information dictionary containing details about the instance and solution.
        """
        return {
            "filename": self.file_name,
            "var_dimensions": self.dim,
            "objective": self.main_problem.model.objective.expr.to_string(),
            "binary_solution": self.main_problem.binary_solution,
            "facet_quota": self.sep_problem.get_facet_quota(),
            "added_cuts": len(self.sep_problem.solution_history),
            "ended_with_redundant_cut": self.sep_problem.redundant,
            "main_problem_solutions": self.main_problem.solution_history,
            "sep_problem_solutions": self.sep_problem.solution_history,
            "solution_same_as_BP_sol": self.main_problem.check_solutions(),
            "cut_complexity": self.sep_problem.get_all_cut_complexity(),
            "is_full_dim": self.sep_problem.is_full_dim,
            "space_dim": self.sep_problem.space_dim,
            "vertex_rank": self.sep_problem.vertex_rank,
            "settings": self.settings,
            "average_solving_time_main": self.main_problem.get_average_solving_time(),
            "average_solving_time_sep": self.sep_problem.get_average_solving_time(),
            "sum_solving_time_main": self.main_problem.get_sum_solving_time(),
            "sum_solving_time_sep": self.sep_problem.get_sum_solving_time(),
            "total_solving_time": self.main_problem.get_sum_solving_time() + self.sep_problem.get_sum_solving_time(),
            "BP_solving_time": self.main_problem.get_BP_solving_time(),
            "enumeration_solving_time": self.enumeration_solving_time,
            "enumeration_solution_same_as_BP": self.check_same_as_BP_solution(),
        }
