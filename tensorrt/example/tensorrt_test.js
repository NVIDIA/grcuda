let fs = require('fs')

// get grCUDA function object
const DeviceArray = Polyglot.eval('grcuda', 'DeviceArray')
const cudaDeviceSynchronize = Polyglot.eval('grcuda', 'cudaDeviceSynchronize')

// get function objects from TRT namespace
const createInferRuntime = Polyglot.eval('grcuda', 'TRT::createInferRuntime')
const deserializeCudaEngine = Polyglot.eval('grcuda', 'TRT::deserializeCudaEngine')
const destroyInferRuntime = Polyglot.eval('grcuda', 'TRT::destroyInferRuntime')
const createExecutionContext = Polyglot.eval('grcuda', 'TRT::createExecutionContext')
const getBindingIndex = Polyglot.eval('grcuda', 'TRT::getBindingIndex')
const getMaxBatchSize = Polyglot.eval('grcuda', 'TRT::getMaxBatchSize')
const enqueue = Polyglot.eval('grcuda', 'TRT::enqueue')
const destroyEngine = Polyglot.eval('grcuda', 'TRT::destroyEngine')
const destroyExecutionContext = Polyglot.eval('grcuda', 'TRT::destroyExecutionContext')


// helper function to read PGM image file
function readPGM(imageFile, imageDeviceArray) {
  fileBuffer = fs.readFileSync(imageFile)
  let offset = 0;
  let nlPos = fileBuffer.indexOf('\n', offset)
  let magic = fileBuffer.slice(offset, nlPos).toString()

  offset = nlPos + 1
  nlPos = fileBuffer.indexOf(' ', offset)
  let width = parseInt(fileBuffer.slice(offset, nlPos).toString())

  offset = nlPos + 1
  nlPos = fileBuffer.indexOf('\n', offset)
  let height = parseInt(fileBuffer.slice(offset, nlPos).toString())

  offset = nlPos + 1
  nlPos = fileBuffer.indexOf('\n', offset)
  let maxLevels = parseInt(fileBuffer.slice(offset, nlPos).toString())
  offset = nlPos + 1
  data = fileBuffer.slice(offset)

  let useUint16 = maxLevels > 255
  let numElements = width * height;
  if (imageDeviceArray.length < numElements) {
    throw `image file contains ${numElements} elements but buffer can hold only ${imageDeviceArray.length}`
  }
  for (let row = 0; row < height; row += 1) {
    for (let col = 0; col < width; col += 1) {
      let idx = row * width + col
      let value;
      if (useUint16) {
        value = data.readUInt16LE(idx)
      } else {
        value = data.readUInt8(idx)
      }
      imageDeviceArray[idx] = 1.0 - value / maxLevels
    }
  }
}


if (process.argv.length != 3) {
  console.error('one single argument: pgm image file')
  process.exit(1)
}
imageFile = process.argv[2]

// create buffers
const width = 28
const height = 28
const numInputElements = width * height
const numClasses = 10
let classProbabilities = DeviceArray('float', numClasses)
let inputImage = DeviceArray('float', numInputElements)

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
let runtime = createInferRuntime()

// load engine file
const engineFile = '/data/proj/oracle/workspace_dev/grcuda/tensorrt/models/lenet5.engine'
console.log(`deserializeCudaEngine(runtime=${runtime}, engineFileName='${engineFile}')`)
engine = deserializeCudaEngine(runtime, engineFile)

// get binding indices for input and output layers
const inputName = 'conv2d_input'
const outputName = 'dense_2/Softmax'
const inputIndex = getBindingIndex(engine, inputName)
const outputIndex = getBindingIndex(engine, outputName)
console.log(`input index = ${inputIndex} for '${inputName}'`)
console.log(`output index = ${outputIndex} for '${outputName}'`)
const maxBatchSize = getMaxBatchSize(engine)
console.log(`max batch size = ${maxBatchSize}`)

// create execution context
console.log(`createExecutionContext(engine=${engine})`)
const executionContext = createExecutionContext(engine)

// submit inference job
const batchSize = 1
const buffers = new Array(2)
buffers[inputIndex] = inputImage
buffers[outputIndex] = classProbabilities
enqueue(engine, batchSize, buffers)
cudaDeviceSynchronize()

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
console.log(`destroyExecutionContext(executionContext=${executionContext})`)
destroyExecutionContext(executionContext)
console.log(`destroyEngine(engine=${engine})`)
destroyEngine(engine)
console.log(`destroyInferRuntime(runtime=${runtime})`)
destroyInferRuntime(runtime)

