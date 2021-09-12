export const MOCK_OPTIONS = {
  DELAY: 10,              //ms
  DELAY_JITTER_SYNC: 30,  //ms
  DELAY_JITTER_ASYNC: 0,  //ms
  DELAY_JITTER_NATIVE: 50 //ms
}

export const CONFIG_OPTIONS = {
  MAX_PHOTOS: 60,
  SEND_BATCH_SIZE: 1
}

export const COMPUTATION_MODES: Array<string> = ["sync", "async", "cuda-native", "race-sync", "race-async", "race-cuda-native"]

// Convert images to black and white
export const BW = false

// Edge width (in pixel) of input images.
// If a loaded image has lower width than this, it is rescaled
export const RESIZED_IMG_WIDTH = 1024
// Edge width (in pixel) of output images.
// We store processed images in 2 variants: small and large
export const RESIZED_IMG_WIDTH_OUT_SMALL = 40
export const RESIZED_IMG_WIDTH_OUT_LARGE = RESIZED_IMG_WIDTH


// Constant parameters used in the image processing
export const KERNEL_SMALL_DIAMETER = 7
export const KERNEL_SMALL_VARIANCE = 0.1
export const KERNEL_LARGE_DIAMETER = 9
export const KERNEL_LARGE_VARIANCE = 20
export const KERNEL_UNSHARPEN_DIAMETER = 7
export const KERNEL_UNSHARPEN_VARIANCE = 5
export const UNSHARPEN_AMOUNT = 30
export const CDEPTH = 256
export const FACTOR = 0.8

// CUDA parameters
export const NUM_BLOCKS = 2
export const THREADS_1D = 32
export const THREADS_2D = 8
export const IMAGE_IN_DIRECTORY = `../frontend/images/dataset${RESIZED_IMG_WIDTH}`
export const IMAGE_OUT_SMALL_DIRECTORY = "../frontend/images/thumb"
export const IMAGE_OUT_BIG_DIRECTORY = "../frontend/images/full_res"

export const CUDA_NATIVE_IMAGE_IN_DIRECTORY = IMAGE_IN_DIRECTORY
export const CUDA_NATIVE_EXEC_FILE = "../../image_pipeline_local/cuda/build/image_pipeline"
export const CUDA_NATIVE_IMAGE_OUT_SMALL_DIRECTORY = IMAGE_OUT_SMALL_DIRECTORY // "$HOME/grcuda/projects/demos/web_demo/frontend/images/thumb/"
export const CUDA_NATIVE_IMAGE_OUT_BIG_DIRECTORY = IMAGE_OUT_BIG_DIRECTORY //"$HOME/grcuda/projects/demos/web_demo/frontend/images/full_res/"

export const DEBUG = false

