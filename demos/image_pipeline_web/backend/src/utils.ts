import cv from "opencv4nodejs"
import fs from "fs"

import {
  RESIZED_IMG_WIDTH,
  BW,
  RESIZED_IMG_WIDTH_OUT_LARGE,
  RESIZED_IMG_WIDTH_OUT_SMALL,
  IMAGE_IN_DIRECTORY,
  IMAGE_OUT_BIG_DIRECTORY,
  IMAGE_OUT_SMALL_DIRECTORY,
  MOCK_OPTIONS,
  CDEPTH
} from "./options"



export const _sleep = (ms: number) => {
  return new Promise((resolve) => {
    setTimeout(resolve, ms);
  });
}

export const _gaussianKernel = (buffer: any, diameter: number, sigma: number) => {
  const mean = diameter / 2;
  let sum_tmp = 0;
  for (let x = 0; x < diameter; x++) {
    for (let y = 0; y < diameter; y++) {
      const val = Math.exp(-0.5 * (Math.pow(x - mean, 2) + Math.pow(y - mean, 2)) / Math.pow(sigma, 2));
      buffer[x][y] = val;
      sum_tmp += val;
    }
  }
  // Normalize;
  for (let x = 0; x < diameter; x++) {
    for (let y = 0; y < diameter; y++) {
      buffer[x][y] /= sum_tmp;
    }
  }
}

export const _getDelayJitter = (computationType: string) => {

  const {
    DELAY_JITTER_ASYNC,
    DELAY_JITTER_SYNC,
    DELAY_JITTER_NATIVE
  } = MOCK_OPTIONS

  switch (computationType) {
    case "sync": {
      return DELAY_JITTER_SYNC
    }
    case "async": {
      return DELAY_JITTER_ASYNC
    }
    case "cuda-native": {
      return DELAY_JITTER_NATIVE
    }
    case "race-sync": {
      return DELAY_JITTER_SYNC
    }
    case "race-async": {
      return DELAY_JITTER_ASYNC
    }
    case "race-cuda-native": {
      return DELAY_JITTER_NATIVE
    }
  }

}

export async function loadImage(imgName: string | number, resizeWidth = RESIZED_IMG_WIDTH, resizeHeight = RESIZED_IMG_WIDTH, fileFormat = ".jpg") {
  const imagePath = `${IMAGE_IN_DIRECTORY}/${imgName}${fileFormat}`
  const image = await cv.imreadAsync(imagePath, BW ? cv.IMREAD_GRAYSCALE : cv.IMREAD_COLOR)
  return image
}

export async function storeImageInner(img: cv.Mat, imgName: string | number, resolution: number, kind: string, imgFormat: string = ".jpg", blackAndWhite: boolean = BW) {
  const imgResized = img.resize(resolution, resolution);
  const buffer = await cv.imencodeAsync('.jpg', imgResized, [cv.IMWRITE_JPEG_QUALITY, 80])
  const writeDirectory = kind === "full_res" ? IMAGE_OUT_BIG_DIRECTORY : IMAGE_OUT_SMALL_DIRECTORY
  fs.writeFileSync(`${writeDirectory}/${imgName}${imgFormat}`, buffer);
}

// Store the output of the image processing into 2 images,
// with low and high resolution;
export async function storeImage(img: cv.Mat, imgName: string | number, resizedImageWidthLarge = RESIZED_IMG_WIDTH_OUT_LARGE, resizedImageWidthSmall = RESIZED_IMG_WIDTH_OUT_SMALL, blackAndWhite: boolean = BW) {
  storeImageInner(img, imgName, resizedImageWidthLarge, "full_res", ".jpg", blackAndWhite);
  storeImageInner(img, imgName, resizedImageWidthSmall, "thumb", ".jpg", blackAndWhite);
}

export function _intervalToMs(start: number, end: number) {
  return (end - start) / 1e6;
}

export const copyFrom = (arrayFrom: any, arrayTo: any) => {
  for (let i = 0; i < arrayTo.length; ++i) {
    arrayTo[i] = arrayFrom[i]
  }
}

// Beziér curve defined by 3 points.
// The input is used to map points of the curve to the output LUT,
// and can be used to combine multiple LUTs.
// By default, it is just [0, 1, ..., 255];
function spline3(input: any, lut: any, P: any) {
  for (let i = 0; i < CDEPTH; i++) {
    const t = i / CDEPTH;
    const x = Math.pow((1 - t), 2) * P[0] + 2 * t * (1 - t) * P[1] + Math.pow(t, 2) * P[2];
    lut[i] = input[(x * CDEPTH) >> 0]; // >> 0 is an evil hack to cast float to int;
  }
}

// Beziér curve defined by 5 points;
function spline5(input: any, lut: any, P: any) {
  for (let i = 0; i < CDEPTH; i++) {
    const t = i / CDEPTH;
    const x = Math.pow((1 - t), 4) * P[0] + 4 * t * Math.pow((1 - t), 3) * P[1] + 6 * Math.pow(t, 2) * Math.pow((1 - t), 2) * P[2] + 4 * Math.pow(t, 3) * (1 - t) * P[3] + Math.pow(t, 4) * P[4];
    lut[i] = input[(x * CDEPTH) >> 0];
  }
}

function lut_r(lut: any) {
  // Create a temporary LUT to swap values;
  let lut_tmp = new Array(CDEPTH).fill(0);

  // Initialize LUT;
  for (let i = 0; i < CDEPTH; i++) {
    lut[i] = i;
  }
  // Apply 1st curve;
  const P = [0.0, 0.2, 1.0];
  spline3(lut, lut_tmp, P);
  // Apply 2nd curve;
  const P2 = [0.0, 0.3, 0.5, 0.99, 1];
  spline5(lut_tmp, lut, P2);
}

function lut_g(lut: any) {
  // Create a temporary LUT to swap values;
  let lut_tmp = new Array(CDEPTH).fill(0);
  // Initialize LUT;
  for (let i = 0; i < CDEPTH; i++) {
    lut[i] = i;
  }
  // Apply 1st curve;
  const P = [0.0, 0.01, 0.5, 0.99, 1];
  spline5(lut, lut_tmp, P);
  // Apply 2nd curve;
  const P2 = [0.0, 0.1, 0.5, 0.75, 1];
  spline5(lut_tmp, lut, P2);
}

function lut_b(lut: any) {
  // Create a temporary LUT to swap values;
  let lut_tmp = new Array(CDEPTH).fill(0);
  // Initialize LUT;
  for (let i = 0; i < CDEPTH; i++) {
    lut[i] = i;
  }
  // Apply 1st curve;
  const P = [0.0, 0.01, 0.5, 0.99, 1];
  spline5(lut, lut_tmp, P);
  // Apply 2nd curve;
  const P2 = [0.0, 0.25, 0.5, 0.70, 1];
  spline5(lut_tmp, lut, P2);
}

// Initialize LUTs;
export const LUT = [new Array(CDEPTH).fill(0), new Array(CDEPTH).fill(0), new Array(CDEPTH).fill(0)];
lut_r(LUT[0]);
lut_g(LUT[1]);
lut_b(LUT[2]);
