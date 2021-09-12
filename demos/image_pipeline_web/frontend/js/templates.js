window.getHeader = (computationMode) => `
  <h3 class="display-4">Computation Mode: ${computationMode}</h3>
`

window.getSyncTemplate = () => `
<div class="row" id="sync-pipeline-description">
<div class="col-sm-8">
  ${window.getHeader("Sync")}
  <p class="lead">In this demo, we bring your photo collection back in time and give it a nice vintage look that everybody loves!</p>
  <p>But there's a lot going on behind the scenes. 
  First of all, we make the subject pop! Through a complex pipeline of gaussian blur, edge-detection and sharpening filters, we can identify the subject contour and make it sharper, while slightly blurrying the background and other smooth textures.
  Then, we apply a retro touch to the pictures, with a custom vintage LUT. </p>
</div>


<div class="col-sm-4">
<img src="./images/description/sync/pipeline-sync.png" class="img-fluid" style="max-width: 100%; height: auto;" alt="Responsive image">
</div>
</div>

<div class="row" id="sync-pipeline-description">
<div class="col-sm-8">
  <p class="lead">In the <b>Sync</b> pipeline, we adopt the original GrCUDA implementation.</p>
  <p> In this version, every computation is executed on the default CUDA stream, meaning that we don't see any overlap between computations and data transfer, or even between multiple image processing requests. 
  As a result, a lot of performance is left on the table and most GPU resources are wasted.
  </p>
</div>


<div class="col-sm-4">
<img src="./images/description/async/1.png" class="img-fluid" style="max-width: 100%; height: auto;" alt="Responsive image">
</div>
</div>

<div class="row" id="sync-pipeline-description">
<div class="col-sm-8">
  <p class="lead">In the <b>Async</b> pipeline, we show you the power of our new GrCUDA scheduler.</p>
  <p>On the surface, the code you write is 100% identical to the SYNC pipeline.
  However, all computations happens asynchronously: requests are overlapped, and so are GPU computations and data transfer.
  Moreover, we transparently perform many other optimizations, such as prefetching data to the GPU to be even faster.
  Just by making better use of wasted resources, we get a large 30% speedup with no code change whatsoever. Pretty impressive, isn't it?</p>
</div>


<div class="col-sm-4">
<img src="./images/description/async/2.png" class="img-fluid" style="max-width: 100%; height: auto;" alt="Responsive image">
</div>
</div>

<div class="row" id="sync-pipeline-description">
<div class="col-sm-8">
  <p class="lead">But how does GrCUDA fare against code written by a ninja programmer in C++, with direct access to the CUDA API?</p>
  <p>In the <b>Native</b> pipeline, we build an entirely separate CUDA application to load and process images, and call it from JavaScript. 
  It is significantly more complex, with a lot of programming overhead (e.g. to handle input options). 
  Is it worth having direct access to all the lowest level CUDA APIs? It turns out that GrCUDA provides the same perfomrance, with much simpler code!</p>
</div>


<div class="col-sm-4">
<img src="./images/description/async/3.png" class="img-fluid" style="max-width: 100%; height: auto;" alt="Responsive image">
</div>
</div>
`
window.getAsyncTemplate = () => `
  <div class="row" id="sync-pipeline-description">
<div class="col-sm-8">
    ${window.getHeader("Async")}
    <p class="lead">In this demo, we bring your photo collection back in time and give it a nice vintage look that everybody loves!</p>
    <p>But there's a lot going on behind the scenes. 
    First of all, we make the subject pop! Through a complex pipeline of gaussian blur, edge-detection and sharpening filters, we can identify the subject contour and make it sharper, while slightly blurrying the background and other smooth textures.
    Then, we apply a retro touch to the pictures, with a custom vintage LUT. </p>
</div>


<div class="col-sm-4">
  <img src="./images/description/async/pipeline-async.png" class="img-fluid" style="max-width: 100%; height: auto;" alt="Responsive image">
</div>
</div>

<div class="row" id="sync-pipeline-description">
<div class="col-sm-8">
    <p class="lead">In the <b>Sync</b> pipeline, we adopt the original GrCUDA implementation.</p>
    <p> In this version, every computation is executed on the default CUDA stream, meaning that we don't see any overlap between computations and data transfer, or even between multiple image processing requests. 
    As a result, a lot of performance is left on the table and most GPU resources are wasted.
    </p>
</div>

<div class="col-sm-4">
  <img src="./images/description/async/1.png" class="img-fluid" style="max-width: 100%; height: auto;" alt="Responsive image">
</div>
</div>

<div class="row" id="sync-pipeline-description">
<div class="col-sm-8">
    <p class="lead">In the <b>Async</b> pipeline, we show you the power of our new GrCUDA scheduler.</p>
    <p>On the surface, the code you write is 100% identical to the SYNC pipeline.
    However, all computations happens asynchronously: requests are overlapped, and so are GPU computations and data transfer.
    Moreover, we transparently perform many other optimizations, such as prefetching data to the GPU to be even faster.
    Just by making better use of wasted resources, we get a large 30% speedup with no code change whatsoever. Pretty impressive, isn't it?</p>
</div>


<div class="col-sm-4">
  <img src="./images/description/async/2.png" class="img-fluid" style="max-width: 100%; height: auto;" alt="Responsive image">
</div>
</div>

<div class="row" id="sync-pipeline-description">
<div class="col-sm-8">
    <p class="lead">But how does GrCUDA fare against code written by a ninja programmer in C++, with direct access to the CUDA API?</p>
    <p>In the <b>Native</b> pipeline, we build an entirely separate CUDA application to load and process images, and call it from JavaScript. 
    It is significantly more complex, with a lot of programming overhead (e.g. to handle input options). 
    Is it worth having direct access to all the lowest level CUDA APIs? It turns out that GrCUDA provides the same perfomrance, with much simpler code!</p>
</div>


<div class="col-sm-4">
  <img src="./images/description/async/3.png" class="img-fluid" style="max-width: 100%; height: auto;" alt="Responsive image">
</div>
</div>
`
window.getCudaNativeTemplate = () => `
<div class="row" id="sync-pipeline-description">
<div class="col-sm-8">
  ${window.getHeader("Cuda Native")}
  <p class="lead">In this demo, we bring your photo collection back in time and give it a nice vintage look that everybody loves!</p>
  <p>But there's a lot going on behind the scenes. 
  First of all, we make the subject pop! Through a complex pipeline of gaussian blur, edge-detection and sharpening filters, we can identify the subject contour and make it sharper, while slightly blurrying the background and other smooth textures.
  Then, we apply a retro touch to the pictures, with a custom vintage LUT. </p>
</div>


<div class="col-sm-4">
<img src="./images/description/async/pipeline-async.png" class="img-fluid" style="max-width: 100%; height: auto;" alt="Responsive image">
</div>
</div>

<div class="row" id="sync-pipeline-description">
<div class="col-sm-8">
  <p class="lead">In the <b>Sync</b> pipeline, we adopt the original GrCUDA implementation.</p>
  <p> In this version, every computation is executed on the default CUDA stream, meaning that we don't see any overlap between computations and data transfer, or even between multiple image processing requests. 
  As a result, a lot of performance is left on the table and most GPU resources are wasted.
  </p>
</div>


<div class="col-sm-4">
<img src="./images/description/async/1.png" class="img-fluid" style="max-width: 100%; height: auto;" alt="Responsive image">
</div>
</div>

<div class="row" id="sync-pipeline-description">
<div class="col-sm-8">
  <p class="lead">In the <b>Async</b> pipeline, we show you the power of our new GrCUDA scheduler.</p>
  <p>On the surface, the code you write is 100% identical to the SYNC pipeline.
  However, all computations happens asynchronously: requests are overlapped, and so are GPU computations and data transfer.
  Moreover, we transparently perform many other optimizations, such as prefetching data to the GPU to be even faster.
  Just by making better use of wasted resources, we get a large 30% speedup with no code change whatsoever. Pretty impressive, isn't it?</p>
</div>


<div class="col-sm-4">
<img src="./images/description/async/2.png" class="img-fluid" style="max-width: 100%; height: auto;" alt="Responsive image">
</div>
</div>

<div class="row" id="sync-pipeline-description">
<div class="col-sm-8">
  <p class="lead">But how does GrCUDA fare against code written by a ninja programmer in C++, with direct access to the CUDA API?</p>
  <p>In the <b>Native</b> pipeline, we build an entirely separate CUDA application to load and process images, and call it from JavaScript. 
  It is significantly more complex, with a lot of programming overhead (e.g. to handle input options). 
  Is it worth having direct access to all the lowest level CUDA APIs? It turns out that GrCUDA provides the same perfomrance, with much simpler code!</p>
</div>


<div class="col-sm-4">
<img src="./images/description/async/3.png" class="img-fluid" style="max-width: 100%; height: auto;" alt="Responsive image">
</div>
</div>
`

window.getRaceModeTemplate = () => `
      <div class="row">
          <div class="col-sm-12">
            <div id="container-info" class="">
              <div class="row" id="race-mode-pipeline-description">
                <div class="col-sm-8">
                    <h3 class="display-4">Race Mode</h3>
                    <p class="lead">In this mode we run the three pipelines in parallel to see which one processes images the fastest.</p>
                    <p> The three pipelines are handled by three separate backend and execute on three different GPUs to independently evaluate their execution times.</p>
                </div>
              </div>
            </div>
        </div>
`


window.getImageLightBoxTemplate = (paddedImageId, imageId) => `<img src="./images/full_res/${paddedImageId}.jpg" id="${imageId}-full-res" onclick="openLightBox('${imageId}')">`
window.getGalleryImageContentTemplate = (image, imageId) => `<img class="image-pad image" src="${image}" id="${imageId}" onclick="openLightBox('${imageId}')">`

window.getProgressBarTemplate = (progressData, completed) => {
  if (!completed) {
    return `<div class="progress">
            <div style="width: ${progressData}%" class="progress-bar progress-bar-striped progress-bar-animated" role="progressbar" aria-valuenow="${progressData}" aria-valuemin="0" aria-valuemax="100">${Math.round(progressData)}%</div>
          </div>`
  } else {
    return `<div class="progress">
              <div style="width: ${progressData}%" class="progress-bar bg-success" role="progressbar" aria-valuenow="100" aria-valuemin="0" aria-valuemax="100">100%</div>
            </div>`
  }
}
window.getProgressBarWithWrapperTemplate = (
  label, 
  progressBarsCompletionAmount, 
  progressBarRaceColor, 
  computationType
) => `
    <div class="m-3">
      <div class="row">
        <div class="col-sm-12 mb-3">
          <span> Compute Mode: ${label} </span>
          <span id="${label}-execution-time" ></span>
        </div>
      </div>
      <div class="row">
        <div class="col-sm-12">
          <div class="progress">
            <div style="width: ${progressBarsCompletionAmount[computationType]}%" class="progress-bar ${progressBarRaceColor[computationType]}" role="progressbar" aria-valuenow="${progressBarsCompletionAmount[computationType]}" aria-valuemin="0" aria-valuemax="100">${Math.round(progressBarsCompletionAmount[computationType])}%</div>
          </div>
        </div>
      </div>
    </div>  
    `

