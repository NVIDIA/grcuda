// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'use strict'

const fs = require('fs')

// get GrCUDA root namespace and trt namespace object
const cu = Polyglot.eval('grcuda', 'CU')
const trt = Polyglot.eval('grcuda', 'CU::TRT')

// helper function to read PGM image file
function readPGM (imageFile, imageDeviceArray) {
  const fileBuffer = fs.readFileSync(imageFile)
  let offset = 0
  let nlPos = fileBuffer.indexOf('\n', offset)
  const magic = fileBuffer.slice(offset, nlPos).toString()
  if (magic !== 'P5') {
    throw Error(`invalid file magic in ${imageFile}`)
  }
  console.log(magic)

  offset = nlPos + 1
  nlPos = fileBuffer.indexOf(' ', offset)
  const width = parseInt(fileBuffer.slice(offset, nlPos).toString())

  offset = nlPos + 1
  nlPos = fileBuffer.indexOf('\n', offset)
  const height = parseInt(fileBuffer.slice(offset, nlPos).toString())

  offset = nlPos + 1
  nlPos = fileBuffer.indexOf('\n', offset)
  const maxLevels = parseInt(fileBuffer.slice(offset, nlPos).toString())
  offset = nlPos + 1
  const data = fileBuffer.slice(offset)

  const useUint16 = maxLevels > 255
  const numElements = width * height
  if (imageDeviceArray.length < numElements) {
    throw Error(`image file contains ${numElements} elements but buffer can hold only ${imageDeviceArray.length}`)
  }
  for (let row = 0; row < height; row += 1) {
    for (let col = 0; col < width; col += 1) {
      const idx = row * width + col
      let value
      if (useUint16) {
        value = data.readUInt16LE(idx)
      } else {
        value = data.readUInt8(idx)
      }
      imageDeviceArray[idx] = 1.0 - value / maxLevels
    }
  }
}

if (process.argv.length !== 3) {
  console.error('one single argument: pgm image file')
  process.exit(1)
}
const imageFile = process.argv[2]

// create buffers
const width = 28
const height = 28
const numInputElements = width * height
const numClasses = 10
const classProbabilities = cu.DeviceArray('float', numClasses)
const inputImage = cu.DeviceArray('float', numInputElements)

// load image file
console.log(`loading ${imageFile}...`)
readPGM(imageFile, inputImage)
let str = ''
for (let row = 0; row < height; row += 1) {
  for (let col = 0; col < width; col += 1) {
    str += inputImage[row * width + col] < 0.5 ? '@' : ' '
  }
  str += '\n'
}
console.log(str)

// instantiate inference runtime
console.log('createInferRuntime()')
const runtime = trt.createInferRuntime()

// load engine file
const engineFile = 'models/lenet5.engine'
console.log(`deserializeCudaEngine(runtime=${runtime}, engineFileName='${engineFile}')`)
const engine = trt.deserializeCudaEngine(runtime, engineFile)

// get binding indices for input and output layers
const inputName = 'conv2d_input'
const outputName = 'dense_2/Softmax'
const inputIndex = trt.getBindingIndex(engine, inputName)
const outputIndex = trt.getBindingIndex(engine, outputName)
console.log(`input index = ${inputIndex} for '${inputName}'`)
console.log(`output index = ${outputIndex} for '${outputName}'`)
const maxBatchSize = trt.getMaxBatchSize(engine)
console.log(`max batch size = ${maxBatchSize}`)

// create execution context
console.log(`createExecutionContext(engine=${engine})`)
const executionContext = trt.createExecutionContext(engine)

// submit inference job
const batchSize = 1
const buffers = []
buffers[inputIndex] = inputImage
buffers[outputIndex] = classProbabilities
trt.enqueue(engine, batchSize, buffers)
cu.cudaDeviceSynchronize()

// print inferred class probabilities and make prediction
let maxProb = 0
let maxProbDigit = 0
for (let digit = 0; digit < numClasses; digit += 1) {
  if (classProbabilities[digit] > maxProb) {
    maxProb = classProbabilities[digit]
    maxProbDigit = digit
  }
  console.log(`${digit}: ${classProbabilities[digit].toFixed(4)}`)
}
console.log(`prediction: ${maxProbDigit}`)

// tear down
trt.destroyExecutionContext(executionContext)
trt.destroyEngine(engine)
trt.destroyInferRuntime(runtime)
