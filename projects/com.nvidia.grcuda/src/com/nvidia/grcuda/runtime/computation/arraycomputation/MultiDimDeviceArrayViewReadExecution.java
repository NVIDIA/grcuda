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
package com.nvidia.grcuda.runtime.computation.arraycomputation;

import com.nvidia.grcuda.runtime.CPUDevice;
import com.nvidia.grcuda.runtime.array.MultiDimDeviceArrayView;
import com.oracle.truffle.api.profiles.ValueProfile;

public class MultiDimDeviceArrayViewReadExecution extends ArrayAccessExecution<MultiDimDeviceArrayView> {

    private final long index;
    private final ValueProfile elementTypeProfile;

    public MultiDimDeviceArrayViewReadExecution(MultiDimDeviceArrayView array,
                                                long index,
                                                ValueProfile elementTypeProfile) {
        super(array.getGrCUDAExecutionContext(), new ArrayAccessExecutionInitializer<>(array.getMdDeviceArray(), true), array);
        this.index = index;
        this.elementTypeProfile = elementTypeProfile;
    }

    @Override
    public void updateLocationOfArrays() {
        if (array.getGrCUDAExecutionContext().isConstAware()) {
            array.addArrayUpToDateLocations(CPUDevice.CPU_DEVICE_ID);
        } else {
            // Clear the list of up-to-date locations: only the CPU has the updated array;
            array.resetArrayUpToDateLocations(CPUDevice.CPU_DEVICE_ID);
        }
    }

    @Override
    public Object execute() {
        Object result = array.readNativeView(index, elementTypeProfile);
        this.setComputationFinished();
        return result;
    }

    @Override
    public String toString() {
        return "MultiDimDeviceArrayViewReadExecution(" +
                "array=" + array +
                ", index=" + index + ")";
    }
}
