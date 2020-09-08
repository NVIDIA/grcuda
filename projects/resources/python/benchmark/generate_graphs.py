from random import sample
import os
import pickle
import argparse
import time
import numpy as np

DEBUG = True
DEGREE = 3

def get_pickle_filename(size: int) -> str:
    pickle_folder = f"{os.getenv('GRCUDA_HOME')}/data/pickle"
    if not os.path.exists(pickle_folder):
        os.makedirs(pickle_folder)
    return os.path.join(pickle_folder, f"graph_{size}")


def create_csr_from_coo(x_in, y_in, val_in, size, degree=None):
    if degree:
        ptr_out = [degree] * (size + 1)
        ptr_out[0] = 0
    else:
        # values, count = np.unique(x_in, return_counts=True)
        ptr_out = [0] * (size + 1)
        for x in x_in:
            ptr_out[x] += 1
    for i in range(len(ptr_out) - 1):
        ptr_out[i + 1] += ptr_out[i]
    # ptr_out = np.cumsum(ptr_out, dtype=np.int32)
    return ptr_out, y_in, val_in


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate graphs used by the HITS (B7) benchmark")

    parser.add_argument("-d", "--debug", action="store_true",
                        help="If present, print debug messages")
    parser.add_argument("-n", "--size", metavar="N", type=int, nargs="*",
                        help="Sizes of the graph")
    parser.add_argument("-e", "--degree", metavar="N", type=int, nargs="?",
                        help="Degree of the graph")

    args = parser.parse_args()
    debug = args.debug if args.debug else DEBUG
    degree = args.degree if args.degree else DEGREE
    sizes = args.size
    
    for size in sizes:
        if debug:
            print(f"creating graph with N={size}, E={size * degree}")

        # Create a random COO graph;
        start = time.time()
        x = [0] * size * degree
        y = [0] * size * degree
        val = [1] * size * degree
        for i in range(size):
            # Create degree random edges;
            x[(i * degree):((i + 1) * degree)] = [i] * degree
            y[(i * degree):((i + 1) * degree)] = sorted(sample(range(size), degree))
        if debug:
            print(f"1. created COO, {time.time() - start:.2f} sec")
    
        # Turn the COO into CSR and CSC representations;
        start = time.time()
        ptr_cpu, idx_cpu, val_cpu = create_csr_from_coo(x, y, val, size, degree=degree)
        if debug:
            print(f"2. created CSR1, {time.time() - start:.2f} sec")

        start = time.time()
        x2, y2 = zip(*sorted(zip(y, x)))
        if debug:
            print(f"3. sorted CSR1, {time.time() - start:.2f} sec")

        start = time.time()
        ptr2_cpu, idx2_cpu, val2_cpu = create_csr_from_coo(x2, y2, val, size)
        if debug:
            print(f"4. created CSR2, {time.time() - start:.2f} sec")

        # Store to pickle file;
        start = time.time()
        pickle_file_name = get_pickle_filename(size)
        if debug:
            print(f"store pickled data to {pickle_file_name}...")
        with open(pickle_file_name, "wb+") as f:
            pickle.dump([ptr_cpu, idx_cpu, val_cpu, ptr2_cpu, idx2_cpu, val2_cpu], f)
        if debug:
            print(f"5. stores pickled data, {time.time() - start:.2f} sec")
