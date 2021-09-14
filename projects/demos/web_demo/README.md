# Web Demo for SeptembeRSE
The goal of this demo is to showcase an image processing pipeline in _GrCUDA_.

* **Abstract from the demo**
    ```
    GPUs are readily available in cloud computing and personal devices, but their use for data processing acceleration has been slowed down by their limited integration with common programming languages such as Python or Java.
    Moreover, very few ninja programmers have the expert knowledge of asynchronous programming required to use GPUs at their best.
    The GrCUDA polyglot API is a significant step forward in the never-ending quest of making GPU programming more accessible. 
    GrCUDA exposes the CUDA API to all the high-level languages supported by GraalVM, such as JavaScript, R, Python and Scala, drastically lowering the integration efforst with these languages.
    But that's not all: we have recently improved GrCUDA to transparently provide asynchronous execution, hardware space-sharing, and transfer-computation overlap without requiring in advance any information about the program dependency structure.
    We achieve an average of 44% speedup against synchronous execution, with no code change whatsoever!

    In this tutorial, we show the strengths of GrCUDA by showcasing a complex image processing application.
    But no fear: you will see how easy it is to accelerate JavaScript using GPUs, achieving the same performance as the low-level C++ CUDA API, with drastically simpler code!
    ```
 
## Installation and setup

1. `./setup_demo.sh` compiles GrCUDA, downloads the image dataset, installs dependencies for the demo, builds the backend (including the native CUDA implementation), and starts the demo
2. The `./setup_demo.sh` script launches an HTTP accessible as `localhost:8085` from your web browser
3. If you need to change some of the ports (e.g. because they are already in use), modify `backend/package.json` and `frontend/index.json`, then rebuild the demo
4. If running the demo on a remote machine, you need to setup port forwarding before connecting. From the terminal, `ssh -f -L LOCAL_PORT:DESTINATION_IP:DESTINATION_PORT user@DESTINATION_IP`. If you are running the demo inside Visual Studio Code, open the `PORTS` tab (right of the `TERMINAL` tab), and type the ports you have to forward (e.g. 8080, 8082, 8083)
5. `./run_demo.sh` simply starts the demo, without building it first

## Backend
The backend is in charge of receiving signal (via `websockets`) associated to the beginning of the computation and the computation mode (either `sync`, `async` or `cuda-native`) from the frontend and initiate the actual computation using the specified mode.
At each image processed, the backend signals the frontend of the current progresses and of which images (in batch) are ready to be displayed to the final user.

### Install dependencies and run
To install the dependencies run `npm install` in the `backend` directory and compile the cuda binary in the `../image_pipeline` directory using `cmake`.
When in development, it is advisable to run `npm run devall` to compile and run the servers at each code save.
In production, first compile the `typescript` files using the `typescript` compiler (`tsc`) or the `npm build` command. The compiled files can be found in the `dist` directory and can be executed by running `npm run runall`.

## Frontend
The frontend is in charge of signaling the beginning of the computation to the backend, showing the progress and, when the computation is finished, display a grid of the computed images. By clicking on any thumbnail in the grid, the user is displayed the full resolution image.

### Install dependencies and run
Open the `index.html` file, requires the backend to be already running in the local server (`localhost`) on port 8080 (sync), 8083 (async), 8082 (cuda-native).

If running on a remote machine remember setup port forwarding using ssh: 
```
ssh -f -L LOCAL_PORT:DESTINATION_IP:DESTINATION_PORT user@DESTINATION_IP
```