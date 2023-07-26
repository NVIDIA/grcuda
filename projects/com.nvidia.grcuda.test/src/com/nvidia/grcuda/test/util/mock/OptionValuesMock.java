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
package com.nvidia.grcuda.test.util.mock;

import com.nvidia.grcuda.GrCUDALanguage;
import org.graalvm.options.OptionDescriptors;
import org.graalvm.options.OptionKey;
import org.graalvm.options.OptionValues;

import java.util.HashMap;
import java.util.Map;

public class OptionValuesMock implements OptionValues {

    private final Map<OptionKey<?>, Object> values;

    public OptionValuesMock() {
        this.values =  new HashMap<>();
        GrCUDALanguage.getOptionDescriptorsStatic().forEach(o -> values.put(o.getKey(), o.getKey().getDefaultValue()));
    }

    @Override
    public OptionDescriptors getDescriptors() {
        return GrCUDALanguage.getOptionDescriptorsStatic();
    }

    @Override
    public <T> void set(OptionKey<T> optionKey, T value) {
        this.values.put(optionKey, value);
    }

    @Override
    public <T> T get(OptionKey<T> optionKey) {
        return (T) this.values.get(optionKey);
    }

    @Override
    public boolean hasBeenSet(OptionKey<?> optionKey) {
        return values.containsKey(optionKey);
    }

    @Override
    public boolean hasSetOptions() {
        return OptionValues.super.hasSetOptions();
    }
}

