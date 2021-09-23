// Copyright (c) 2021, NECSTLab, Politecnico di Milano. All rights reserved.

// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NECSTLab nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//  * Neither the name of Politecnico di Milano nor the names of its
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
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

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
