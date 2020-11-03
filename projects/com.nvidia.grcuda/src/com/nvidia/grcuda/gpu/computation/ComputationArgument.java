//package com.nvidia.grcuda.gpu.computation;
//
//import com.nvidia.grcuda.gpu.ArgumentType;
//
///**
// * Defines a {@link GrCUDAComputationalElement} argument representing the elements of a NFI signature.
// * For each argument, store its type, if it's a pointer,
// * and if it's constant (i.e. its content cannot be modified in the computation);
// */
//public class ComputationArgument {
//    protected final ArgumentType type;
//    protected final boolean isArray;
//    protected final boolean isConst;
//
//    public ComputationArgument(ArgumentType type, boolean isArray, boolean isConst) {
//        this.type = type;
//        this.isArray = isArray;
//        this.isConst = isConst;
//    }
//
//    public ArgumentType getType() {
//        return type;
//    }
//
//    public boolean isArray() {
//        return isArray;
//    }
//
//    public boolean isConst() {
//        return isConst;
//    }
//
//    @Override
//    public String toString() {
//        return "ComputationArgument(" +
//                "type=" + type +
//                ", isArray=" + isArray +
//                ", isConst=" + isConst +
//                ')';
//    }
//}