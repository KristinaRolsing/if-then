import glob
import itertools
import json
import os
import statistics
from collections import defaultdict

import click
import pandas as pd
from scipy.stats import gmean

from classes import instance


@click.command()
@click.option('--instances_dir', default="../data/", help="Base directory where the data folders are located.")
def main(instances_dir):
    data_dirs = glob.glob(os.path.join(instances_dir, 'data*/'))

    for data_dir in data_dirs:
        print(f"\nProcessing folder: {data_dir}")
        setting_filename = "settings_basic.json"
        settings = read_settings("../data/settings/" + setting_filename)
        run_test(data_dir, setting_filename, settings)


def read_settings(file_path):
    """
    Reads settings from a JSON file and saves them as a dictionary.
    """
    with open(file_path, 'r') as f:
        return json.load(f)


def run_test(data_dir, setting_filename, settings):
    test_instances = [
        test_name
        for test_name in os.listdir(data_dir)
        if test_name != "__init__.py" and 'solutions' not in test_name
    ]
    solutions_dir = data_dir + "solutions/"
    if not os.path.exists(solutions_dir):
        os.makedirs(solutions_dir)

    all_results = {}

    for test_instance in test_instances:
        try:
            instance_result_info_dict = optimize_instance(data_dir, test_instance, settings)
        except Exception as e:
            print(f"Error with instance {test_instance} on first attempt: {e}")
            try:
                print(f"Retrying instance {test_instance}...")
                instance_result_info_dict = optimize_instance(data_dir, test_instance, settings)
            except Exception as e:
                print(f"Skipping instance {test_instance} after second failure: {e}")
                continue
        all_results[test_instance] = instance_result_info_dict
        if settings["write_single_results"]: write_instance_result(solutions_dir, instance_result_info_dict)
    write_all_results(solutions_dir, all_results)
    write_summary(solutions_dir, setting_filename, settings, all_results)


def write_instance_result(output_dir, info_dict):
    """
    Write data of a single instance as a dictionary into a txt file.
    """
    csv_filename = os.path.join(output_dir, f"{info_dict["filename"]}_result.csv")
    data_df = pd.DataFrame.from_dict(info_dict, orient='index').T
    data_df.to_csv(csv_filename, index=False)


def write_all_results(output_dir, all_results):
    """
    Create a dataframe and save is as a csv file.
    The rows contain information of every instance in this run.
    """
    columns = ["filename", "binary_solution", "is_full_dim", "added_cuts", "facet_quota", "solution_same_as_BP_sol",
               "enumeration_solution_same_as_BP",
               "cut_complexity", "average_solving_time_main", "average_solving_time_sep", "sum_solving_time_main",
               "sum_solving_time_sep", "total_solving_time", "BP_solving_time", "enumeration_solving_time", "space_dim",
               "vertex_rank"]

    filtered_results = [
        {col: info_dict[col] for col in columns if col in info_dict}
        for info_dict in all_results.values()
    ]

    df = pd.DataFrame(filtered_results)
    csv_filename = os.path.join(output_dir, "all_results.csv")
    df.to_csv(csv_filename, index=False)


def write_summary(output_dir, setting_filename, settings, all_results):
    """
    Create a dataframe and save is as a csv file.
    The dataframe contains a single row with a summary of all instances in this run.
    """
    all_results = [value for value in all_results.values()]
    all_results_with_cuts = [result for result in all_results if result['added_cuts'] > 0]

    num_all_if_sets = [len(entry["var_dimensions"]) - 1 for entry in all_results]
    card_all_if_sets = list(itertools.chain(*[list(entry["var_dimensions"].values())[:-1] for entry in all_results]))
    card_all_then_set = [entry["var_dimensions"]["then-set"] for entry in all_results]

    non_zero_cuts = [entry['added_cuts'] for entry in all_results_with_cuts]
    all_facet_quota = [entry['facet_quota'] for entry in all_results_with_cuts] if settings["compute_stats"] else None

    merged_cut_complexity = defaultdict(int)
    if settings["compute_stats"]:
        for entry in all_results_with_cuts:
            for key, value in entry.get('cut_complexity', {}).items():
                merged_cut_complexity[key] += value
        merged_cut_complexity = dict(merged_cut_complexity)

    true_count = 0
    false_count = 0
    for entry in all_results_with_cuts:
        for item in entry['sep_problem_solutions']:
            if item['is_facet']:
                true_count += 1
            else:
                false_count += 1

    if false_count == 0:
        if true_count > 0:
            average_overall_facet_quota = 1
        else:
            average_overall_facet_quota = 0
    else:
        average_overall_facet_quota = float(true_count) / (false_count + true_count)

    all_average_solving_time_main = [entry['average_solving_time_main'] for entry in all_results]
    all_average_solving_time_sep = [entry['average_solving_time_sep'] for entry in all_results_with_cuts]
    all_sum_solving_time_main = [entry['sum_solving_time_main'] for entry in all_results]
    all_sum_solving_time_sep = [entry['sum_solving_time_sep'] for entry in all_results_with_cuts]
    all_sum_solving_time = [entry['total_solving_time'] for entry in all_results]

    summary_dict = {
        'setting_filename': setting_filename,
        'number_instances': len(all_results),
        'average_num_if_set': gmean(num_all_if_sets),
        'average_if_set_card': gmean(card_all_if_sets),
        'average_then_set_card': gmean(card_all_then_set),
        'k_ratio': gmean(card_all_then_set) / (gmean(card_all_if_sets) ** gmean(num_all_if_sets)),
        'average_space_dim': gmean(num_all_if_sets) * gmean(card_all_if_sets) + gmean(
            card_all_then_set),
        'found_binary': sum(1 for entry in all_results if entry['binary_solution']) / len(all_results),
        'solution_same_as_BP_sol': sum(1 for entry in all_results if entry['solution_same_as_BP_sol']) / len(
            all_results),
        'enumeration_solution_same_as_BP': sum(
            1 for entry in all_results if entry['enumeration_solution_same_as_BP']) / len(
            all_results),
        'perturbation': all_results[0]["settings"]["perturbation"],
        'main_objective': all_results[0]["settings"]["main_objective"],
        'full_dim': sum(1 for entry in all_results if entry['is_full_dim']) / len(all_results),
        'no_cuts': sum(1 for entry in all_results if entry['added_cuts'] == 0) / len(all_results),
        'range_added_cuts': max(non_zero_cuts) - min(non_zero_cuts) if non_zero_cuts else None,
        'num_cuts_mean': statistics.mean(non_zero_cuts) if non_zero_cuts else None,
        'num_cuts_std_dev': statistics.stdev(non_zero_cuts) if len(non_zero_cuts) > 1 else None,
        'num_cuts_gmean': gmean(non_zero_cuts) if non_zero_cuts else None,
        'num_cuts_median': statistics.median(non_zero_cuts) if non_zero_cuts else None,
        'average_facet_quota': gmean([value + 1 for value in all_facet_quota]) - 1 if all_facet_quota else None,
        'average_overall_facet_quota': average_overall_facet_quota,
        'merged_cut_complexity': merged_cut_complexity if merged_cut_complexity else None,
        'average_solving_time_main': gmean([value + 1 for value in all_average_solving_time_main]) - 1,
        'average_solving_time_sep': gmean(
            [value + 1 for value in all_average_solving_time_sep]) - 1 if all_average_solving_time_sep else None,
        'average_sum_solving_time_main': gmean([value + 1 for value in all_sum_solving_time_main]) - 1,
        'average_sum_solving_time_sep': gmean(
            [value + 1 for value in all_sum_solving_time_sep]) - 1 if all_sum_solving_time_sep else None,
        'average_total_solving_time': gmean([value + 1 for value in all_sum_solving_time]) - 1,
        'average_solving_time_BP': gmean([entry['BP_solving_time'] + 1 for entry in all_results]) - 1,
        'average_enumeration_solving_time': gmean([entry['enumeration_solving_time'] + 1 for entry in all_results]) - 1,
    }

    df = pd.DataFrame([summary_dict])
    csv_filename = os.path.join(output_dir, "result_summary.csv")
    df.to_csv(csv_filename, index=False)


def optimize_instance(data_dir: str, instance_name: str, settings: dict):
    max_cuts = settings["max_cuts"]

    problem_instance = instance.Instance(data_dir, instance_name, settings)
    problem_instance.build_instance()
    print(f"\nBUILD MODEL {instance_name}")
    problem_instance.build_main_problem()

    if settings["solve_BP_also"]:
        print("OPTIMIZE MAIN MODEL AS BP")
        problem_instance.optimize_main_problem_as_BP()

    print("OPTIMIZE MAIN MODEL USING IF THEN")
    problem_instance.optimize_main_problem()
    problem_instance.build_sep_problem()
    for run in range(max_cuts):
        if problem_instance.main_problem.binary_solution:
            print("FOUND BINARY SOLUTION")
            break
        print(f"ADD CUT NR {run + 1}")
        problem_instance.add_cut()
        if problem_instance.sep_problem.redundant:
            print("FOUND REDUNDANT CUT")
            break
        print("OPTIMIZE MAIN PROBLEM AGAIN")
        problem_instance.optimize_main_problem()

    if settings["solve_with_enumeration"]:
        print("SOLVE BP WITH ENUMERATION")
        problem_instance.solve_with_enumeration()
    return problem_instance.get_info_dict()


if __name__ == "__main__":
    main()
