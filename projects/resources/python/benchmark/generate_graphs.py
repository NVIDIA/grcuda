# Copyright (c) 2020, 2021, NECSTLab, Politecnico di Milano. All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NECSTLab nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#  * Neither the name of Politecnico di Milano nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from random import sample
import os
import pickle
import argparse
import time
import json
import numpy as np

DEBUG = True
DEGREE = 3

def get_pickle_filename(size: int) -> str:
    pickle_folder = f"{os.getenv('GRCUDA_HOME')}/data/pickle"
    if not os.path.exists(pickle_folder):
        os.makedirs(pickle_folder)
    return os.path.join(pickle_folder, f"graph_{size}")


def get_json_filename(size: int) -> str:
    pickle_folder = f"{os.getenv('GRCUDA_HOME')}/data/json"
    if not os.path.exists(pickle_folder):
        os.makedirs(pickle_folder)
    return os.path.join(pickle_folder, f"graph_{size}.json")


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
    parser.add_argument("-p", "--pickle", action="store_true",
                        help="If present, store graphs with pickle instead of JSON")
    parser.add_argument("-n", "--size", metavar="N", type=int, nargs="*",
                        help="Sizes of the graph")
    parser.add_argument("-e", "--degree", metavar="N", type=int, nargs="?",
                        help="Degree of the graph")

    OLD = False
    args = parser.parse_args()
    debug = args.debug if args.debug else DEBUG
    degree = args.degree if args.degree else DEGREE
    sizes = args.size
    use_pickle = args.pickle
    
    for size in sizes:
        if debug:
            print(f"creating graph with N={size}, E={size * degree}")

        # Create a random COO graph;
        if OLD:
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
        else:
            start = time.time()
            ptr_cpu = [0] * (size + 1)
            idx_cpu = [0] * size * degree
            val_cpu = [1] * size * degree
            val2_cpu = val_cpu
            csc_dict = {}
            for i in range(size):
                # Create degree random edges;
                ptr_cpu[i + 1] = ptr_cpu[i] + degree
                edges = sample(range(size), degree)
                idx_cpu[(i * degree):((i + 1) * degree)] = edges
                for y_i in edges:
                    if y_i in csc_dict:
                        csc_dict[y_i] += [i]
                    else:
                        csc_dict[y_i] = [i]
            if debug:
                print(f"1. created CSR, {time.time() - start:.2f} sec")

            ptr2_cpu = [0] * (size + 1)
            idx2_cpu = [0] * size * degree
            start = time.time()
            for i in range(size):
                if i in csc_dict:
                    edges = csc_dict[i]
                    ptr2_cpu[i + 1] = ptr2_cpu[i] + len(edges)
                    idx2_cpu[ptr2_cpu[i]:ptr2_cpu[i + 1]] = edges
            if debug:
                print(f"2. created CSC, {time.time() - start:.2f} sec")

        # Store to pickle file;
        start = time.time()
        if use_pickle:
            pickle_file_name = get_pickle_filename(size)
            if debug:
                print(f"store pickled data to {pickle_file_name}...")
            with open(pickle_file_name, "wb") as f:
                pickle.dump([ptr_cpu, idx_cpu, val_cpu, ptr2_cpu, idx2_cpu, val2_cpu], f, pickle.HIGHEST_PROTOCOL)
            if debug:
                print(f"5. stores pickled data, {time.time() - start:.2f} sec")
        else:
            json_file_name = get_json_filename(size)
            if debug:
                print(f"store json data to {json_file_name}...")
            with open(json_file_name, "wb") as f:
                import codecs
                json.dump({"ptr_cpu": ptr_cpu, "idx_cpu": idx_cpu, "val_cpu": val_cpu,
                           "ptr2_cpu": ptr2_cpu, "idx2_cpu": idx2_cpu, "val2_cpu": val2_cpu}, codecs.getwriter('utf-8')(f), ensure_ascii=False)
            if debug:
                print(f"5. stored json data, {time.time() - start:.2f} sec")