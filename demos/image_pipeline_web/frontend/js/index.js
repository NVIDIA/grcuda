const websockets = {
  "sync": new WebSocket("ws://localhost:8080"),
  "async": new WebSocket("ws://localhost:8083"),
  "cuda-native": new WebSocket("ws://localhost:8082"),
}

const sendWSMessage = document.getElementById("btn-send-msg-ws")
const progressBar = document.getElementById("progress-bar")
const imageGallery = document.getElementById("images")
const selectElement = document.getElementById("computation-type")
const containerInfo = document.getElementById("container-info")
const raceModeContainer = document.getElementById("race-mode")
const progressBarSync = document.getElementById("progress-bar-sync")
const progressBarAsync = document.getElementById("progress-bar-async")
const progressBarCudaNative = document.getElementById("progress-bar-cuda-native")

const imageGallerySync = document.getElementById("image-gallery-sync")
const imageGalleryAsync = document.getElementById("image-gallery-async")
const imageGalleryCudaNative = document.getElementById("image-gallery-cuda-native")


const COMPUTATION_MODES = ["race-sync", "race-async", "race-cuda-native"]

const progressBarsRace = {
  "race-sync": progressBarSync,
  "race-async": progressBarAsync,
  "race-cuda-native": progressBarCudaNative
}

const imageGalleriesRace = {
  "race-sync": imageGallerySync,
  "race-async": imageGalleryAsync,
  "race-cuda-native": imageGalleryCudaNative
}

const imageGalleriesRaceContent = {
  "race-sync": "",
  "race-async": "",
  "race-cuda-native": ""
}

const progressBarRaceColor = {
  "race-sync": "progress-bar-striped",
  "race-async": "progress-bar-striped",
  "race-cuda-native": "progress-bar-striped"
}

const labelMap = {
  "race-sync": "Sync",
  "race-async": "Async",
  "race-cuda-native": "Cuda Native",
  "sync": "Sync",
  "async": "Async",
  "cuda-native": "Cuda Native"
}

let progressSync = 0
let progressAsync = 0
let progressNative = 0

let lastProgress = 0

const progressBarsCompletionAmount = {

}

let imageGalleryContent = ""

for(const wsKey of Object.keys(websockets)) {
  console.log(wsKey)
  websockets[wsKey].addEventListener("open", (evt) => {
    console.log(`Connection to websocket for computation mode ${wsKey}`)
  })

  websockets[wsKey].addEventListener("message", (evt) => {
    const data = JSON.parse(evt.data)
  
    if (data.type === "progress") {
      processProgressMessage(evt)
    }
  
    if (data.type === "image") {
      processImageMessage(evt)
    }

    if (data.type === "executionTime"){
      processExecutionTimeMessage(evt)
    }
  })
}

sendWSMessage.onclick = () => {
  clearAll()
  const { value: computationType } = document.getElementById("computation-type")
  console.log(`Beginning computation on ${computationType}`)

  lastProgress = 0
  Object.keys(progressBarsCompletionAmount).forEach(k => progressBarsCompletionAmount[k] = 0)

  if(computationType !== "race-mode") {
    websockets[computationType].send(computationType)
  } else {
    Object.keys(websockets).forEach(ct => websockets[ct].send(`race-${ct}`))
  }

  progressBar.innerHTML = window.getProgressBarTemplate(0, false)

  containerInfo.innerHTML = ""
}

const clearAll = () => {
  progressBar.innerHTML = ""
  imageGallery.innerHTML = ""
  imageGalleryContent = ""

  Object.keys(imageGalleriesRaceContent)
    .forEach(key => imageGalleriesRaceContent[key] = "")

  Object.keys(imageGalleriesRace)
    .forEach(key => imageGalleriesRace[key].innerHTML = "")
  
  COMPUTATION_MODES
    .forEach(cm => progressBarsRace[cm].innerHTML = "")
}

selectElement.onchange = () => {
  const { value: computationType } = document.getElementById("computation-type")

  // Remove progressbar if present
  clearAll()

  console.log(`Value changed to ${computationType}`)

  switch (computationType) {
    case "sync": {
      containerInfo.innerHTML = window.getSyncTemplate()
      break
    }
    case "async": {
      containerInfo.innerHTML = window.getAsyncTemplate()
      break
    }
    case "cuda-native": {
      containerInfo.innerHTML = window.getCudaNativeTemplate()
      break
    }
    case "race-mode": {
      containerInfo.innerHTML = window.getRaceModeTemplate()
      break
    }
  }
}


openLightBox = (imageId) => {
  const mainContainer = document.getElementById("main-container")
  const overlayImage = document.getElementById('overlay');
  const paddedImageId = ("0000" + imageId).slice(-4)
  const imageElement = window.getImageLightBoxTemplate(paddedImageId, imageId)
         
  overlayImage.innerHTML = imageElement
  overlayImage.style.display = 'block';
  mainContainer.setAttribute('class', 'blur');
  const currentImage = document.getElementById(`${imageId}-full-res`)
  currentImage.onclick = () => {
    const mainContainer = document.getElementById("main-container")
    const overlayImage = document.getElementById('overlay');
    overlayImage.style.display = 'none';
    mainContainer.removeAttribute('class', 'blur');
  }
}

const processExecutionTimeMessage = (evt) => {
  const data = JSON.parse(evt.data)
  const { data: executionTime, computationType } = data
  console.log(`${computationType} took: ${executionTime / 1000}s`)
  const formattedExecutionTime = executionTime / 1000
  document.getElementById(`${labelMap[computationType]}-execution-time`).innerHTML = `
    <b>Took ${formattedExecutionTime.toFixed(2)}s</b>
  `
}

const processImageMessage = (evt) => {
  const data = JSON.parse(evt.data)
  const { images, computationType } = data

    console.log(`Received: ${images}`)

    if (!computationType.includes("race")) {

      for (const image of images) {
        const imageId = image.split("/").pop().replace(".jpg", "")
        imageGalleryContent += window.getGalleryImageContentTemplate(image, imageId)
      }

      imageGallery.innerHTML = imageGalleryContent

    } else {

      imageGalleriesRaceContent[computationType] = images.reduce((rest, image) => {
        const imageId = image.split("/").pop().replace(".jpg", "")
        const imgContent = window.getGalleryImageContentTemplate(image, imageId)
        return rest + imgContent
      }, imageGalleriesRaceContent[computationType])

      imageGalleriesRace[computationType].innerHTML = imageGalleriesRaceContent[computationType]
    }
}

const processProgressMessage = (evt) => {
  const data = JSON.parse(evt.data)
  const { data: progressData, computationType } = data
  
  if (!computationType.includes("race")) {
    if(lastProgress > 99.99) return
    lastProgress = Math.max(progressData, lastProgress)

    if (progressData < 99.99) {
      progressBar.innerHTML = window.getProgressBarTemplate(lastProgress, false)
    } else {
      progressBar.innerHTML =  window.getProgressBarTemplate(lastProgress, true)
    }

  } else {

    progressBar.innerHTML = ""
    if(progressBarsCompletionAmount[computationType] > 99.99) return
    progressBarsCompletionAmount[computationType] = Math.max(progressData, progressBarsCompletionAmount[computationType] || 1)
    if (progressData > 99.99) {
      progressBarRaceColor[computationType] = "bg-success"
    } else {
      progressBarRaceColor[computationType] = "progress-bar-striped"
    }

    const label = labelMap[computationType]
    progressBarsRace[computationType].innerHTML = window.getProgressBarWithWrapperTemplate(label, progressBarsCompletionAmount, progressBarRaceColor, computationType)
  }
}

console.log("JS is loaded.")