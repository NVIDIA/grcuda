/*
 * Copyright (c) 2020, 2021, NECSTLab, Politecnico di Milano. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NECSTLab nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *  * Neither the name of Politecnico di Milano nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
package com.nvidia.grcuda.runtime.computation.dependency;

import com.nvidia.grcuda.runtime.computation.ComputationArgumentWithValue;
import com.nvidia.grcuda.runtime.computation.GrCUDAComputationalElement;

import java.util.ArrayList;
import java.util.List;

/**
 * If two computations have the same argument, but it is read-only in both cases (i.e. const),
 * there is no reason to create a dependency between the two ;
 */
public class WithConstDependencyComputation extends DependencyComputation {

    WithConstDependencyComputation(List<ComputationArgumentWithValue> argumentList) {
        activeArgumentSet = new ArrayList<>(argumentList);
    }

    @Override
    public List<ComputationArgumentWithValue> computeDependencies(GrCUDAComputationalElement other) {
        List<ComputationArgumentWithValue> dependencies = new ArrayList<>();
        List<ComputationArgumentWithValue> newArgumentSet = new ArrayList<>();
        for (ComputationArgumentWithValue arg : activeArgumentSet) {
            boolean dependencyFound = false;
            for (ComputationArgumentWithValue otherArg : other.getDependencyComputation().getActiveArgumentSet()) {
                // If both arguments are const, we skip the dependency;
                if (arg.equals(otherArg) && !(arg.isConst() && otherArg.isConst())) {
                    dependencies.add(arg);
                    dependencyFound = true;
                    // If the other argument is const, the current argument must be added to newArgumentSet
                    //   as it could cause other dependencies in the future;
                    if (otherArg.isConst()) {
                        newArgumentSet.add(arg);
                    }
                    break;
                }
            }
            if (!dependencyFound) {
                // Otherwise, the current argument is still "active", and could enforce a dependency on a future computation;
                newArgumentSet.add(arg);
            }
        }
        // Arguments that are not leading to a new dependency could still create new dependencies later on!
        activeArgumentSet = newArgumentSet;
        // Return the list of arguments that created dependencies with the new computation;
        return dependencies;
    }

    /**
     * If the array was attached to a stream, and now it is a const parameter, reset its visibility to the default stream.
     * For simplicity, we keep the visibility of all arguments currently used as const to the default stream.
     * This allow the scheduling of multiple computations that use the same argument as const;
     * @param arg an argument to analyse
     * @return if this argument visibility should be reset or not
     */
    @Override
    public boolean streamResetAttachFilter(ComputationArgumentWithValue arg) {
        return arg.isConst();
    }
}
