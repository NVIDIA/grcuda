/*
 * Copyright (c) 2021, NECSTLab, Politecnico di Milano. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
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
package com.nvidia.grcuda.runtime;

import java.util.HashMap;
import java.util.ArrayList;
import java.util.List;

/**
 * Abstract class that stores the historical execution data of a given computation (for example, a certain GPU kernel).
 * Classes that need to store execution times or other profiling information should use this class.
 * How the execution time (or other information) are actually measured is not specified by this class,
 * which simply defines how such information is stored for future utilization;
 */
public abstract class ProfiledComputation {

    // Track all the execution times associated to the GPU on which a kernel was executed;
    HashMap<Integer, List<Float>> collectionOfExecutions;

    public ProfiledComputation() {
        collectionOfExecutions = new HashMap<>();
    }

    public void addExecutionTime(int deviceId, float executionTime) {
        collectionOfExecutions.putIfAbsent(deviceId, new ArrayList<>());
        collectionOfExecutions.get(deviceId).add(executionTime);
    }

    public List<Float> getExecutionTimesOnDevice(int deviceId) throws RuntimeException {
        if (collectionOfExecutions.containsKey(deviceId)) {
            return collectionOfExecutions.get(deviceId);
        } else {
            throw new RuntimeException("Execution times for device=" + deviceId + "have not been collected");
        }
    }
}