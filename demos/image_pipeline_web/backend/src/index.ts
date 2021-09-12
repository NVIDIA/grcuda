import express from 'express'
import WebSocket from 'ws'
import http from 'http'
import { GrCUDAProxy } from './GrCUDAProxy'


const app = express()
const server = http.createServer(app)
const PORT = parseInt(process.argv[2])
let deviceNumber = parseInt(process.argv[3])
//@ts-ignore
const cu = Polyglot.eval("grcuda", `CU`)

const numDevices = cu.cudaGetDeviceCount()
if (deviceNumber >= numDevices) {
  console.log("warning: device number (" + deviceNumber + ") is bigger than the number of GPUs (" + numDevices + "), using GPU 0 instead");
  deviceNumber = 0;
}
cu.cudaSetDevice(deviceNumber);

const wss = new WebSocket.Server({ server })

wss.on('connection', (ws: WebSocket) => {
  console.log(`[${PORT}] A new client connected`)
  const grCUDAProxy = new GrCUDAProxy(ws)

  ws.on('message', async (message: string) => {
    await grCUDAProxy.beginComputation(message)
  })
})

app.get('/', (req: any, res: any) => {
  res.send("Everithing is working properly")
})

server.listen(PORT, () => console.log(`Running on port ${PORT} - Using GPU ${deviceNumber}`))
