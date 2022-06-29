package com.nvidia.grcuda.test.util.mock;

import com.nvidia.grcuda.GrCUDAOptionMap;
import com.nvidia.grcuda.GrCUDAOptions;
import com.nvidia.grcuda.runtime.stream.policy.DeviceSelectionPolicy;
import com.nvidia.grcuda.runtime.stream.policy.DeviceSelectionPolicyEnum;
import com.nvidia.grcuda.runtime.stream.policy.RetrieveNewStreamPolicyEnum;
import com.nvidia.grcuda.runtime.stream.policy.RetrieveParentStreamPolicyEnum;
import com.nvidia.grcuda.runtime.stream.policy.GrCUDAStreamPolicy;

public class GrCUDAStreamPolicyMock extends GrCUDAStreamPolicy {

    public GrCUDAStreamPolicyMock(
            RetrieveNewStreamPolicyEnum retrieveNewStreamPolicyEnum,
            RetrieveParentStreamPolicyEnum retrieveParentStreamPolicyEnum,
            DeviceSelectionPolicyEnum deviceSelectionPolicyEnum,
            String bandwidthMatrixPath,
            int numberOfAvailableGPUs,
            int numberOfGPUsToUse) {
        super(
                new GrCUDADevicesManagerMock(new DeviceListMock(numberOfAvailableGPUs), numberOfGPUsToUse),
                retrieveNewStreamPolicyEnum,
                retrieveParentStreamPolicyEnum,
                deviceSelectionPolicyEnum,
                bandwidthMatrixPath,
                GrCUDAOptionMap.DEFAULT_DATA_THRESHOLD
        );
    }

    public DeviceSelectionPolicy getDeviceSelectionPolicy() {
        return this.deviceSelectionPolicy;
    }
}
