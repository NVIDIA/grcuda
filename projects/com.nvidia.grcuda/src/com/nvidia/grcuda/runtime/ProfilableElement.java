package com.nvidia.grcuda.runtime;

import com.nvidia.grcuda.GrCUDAContext;
import com.nvidia.grcuda.GrCUDAOptions;

import java.util.Hashtable;

public abstract class ProfilableElement {
    private final boolean profilable;
    // contains latest execution time associated to the GPU on which it was executed
    Hashtable<Integer, Float> collectionOfExecution;
    public ProfilableElement(boolean profilable){
        collectionOfExecution = new Hashtable<Integer, Float>();
        this.profilable = profilable;
    }

    public boolean isProfilable(){
        return this.profilable;
    }

    public void addExecutionTime(int deviceId, float executionTime ){
        collectionOfExecution.put(deviceId, executionTime);
    }

    public float getExecutionTimeOnDevice(int deviceId){
        if(collectionOfExecution.get(deviceId) == null){
            return (float) 0.0;
        }else{
            return collectionOfExecution.get(deviceId);
        }

    }

    // Log execution time on specified device
    public void logExecutionTimeOnDevice(int deviceId){
        if(collectionOfExecution.get(deviceId) == null){
            // log "null"
            // -> how to import LOGGER from GrCUDAContext?
            // LOGGER.info("...");
            System.out.println("DeviceId: " + deviceId + " no exec time found.");
        }else{
            // log "execution time of deviceId: float
            System.out.println("DeviceId: " + deviceId + " -> " + collectionOfExecution.get(deviceId) + " exec time");
        }
    }


}