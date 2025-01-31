import math
import os
import random

import numpy as np


def generate_tensor(cardinality_if_sets, k):
    check_k_value(cardinality_if_sets, k)
    tensor = generate_random_tensor(cardinality_if_sets, k)
    tensor = ensure_all_numbers_in_tensor(tensor, k)
    return tensor


def check_k_value(cardinality_if_sets, k):
    if k > math.prod(cardinality_if_sets):
        raise ValueError("Variable k must be smaller than number of entries in tensor")


def generate_random_tensor(cardinality_if_sets, k):
    if len(cardinality_if_sets) == 1:
        return [random.randint(1, k) for _ in range(cardinality_if_sets[0])]
    else:
        return [generate_random_tensor(cardinality_if_sets[1:], k) for _ in range(cardinality_if_sets[0])]


def ensure_all_numbers_in_tensor(tensor, k):
    flat_tensor = np.array(tensor).flatten()
    flat_tensor[:k] = list(range(1, k + 1))
    np.random.shuffle(flat_tensor)
    reshaped_tensor = flat_tensor.reshape(np.array(tensor).shape)
    return reshaped_tensor.tolist()


def save_tensor_to_file(tensor, filename, cardinality_if_sets, k):
    with open(filename, 'w') as f:
        # Write the dimensions and max_value to the first line
        f.write(f"{' '.join(map(str, cardinality_if_sets))} {k}\n\n")

        # Function to flatten and write the tensor recursively
        def write_tensor(tensor, depth=0):
            if isinstance(tensor[0], list):
                for sub_tensor in tensor:
                    write_tensor(sub_tensor, depth + 1)
                    f.write('\n' * (1 if depth == len(cardinality_if_sets) - 2 else 2))
            else:
                f.write(' '.join(map(str, tensor)) + '\n')

        write_tensor(tensor)


def main():
    random.seed(10)
    # Define the range of dimensions for the tensor
    num_if_sets = 2
    cardinality = 10
    k = 40

    num_files = 400  # Number of files to generate
    output_dir = f'../data/data_{num_if_sets}_{cardinality}_{k}'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i in range(num_files):
        cardinality_if_sets = [cardinality for _ in range(num_if_sets)]

        tensor = generate_tensor(cardinality_if_sets, k)
        filename = os.path.join(output_dir, f"{'_'.join(map(str, cardinality_if_sets))}_k{k}_{i + 1}.dat")
        save_tensor_to_file(tensor, filename, cardinality_if_sets, k)
        print(f"Generated and saved {filename} with cardinalities {cardinality_if_sets}")


if __name__ == "__main__":
    main()
