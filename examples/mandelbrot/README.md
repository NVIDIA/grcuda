# Mandelbrot Web Application with Express

This example demonstrates how GrCUDA can be used in a Node.js
web application with the Express framework.

The code is described in the
[NVIDIA Developer Blog on GrCUDA](https://devblogs.nvidia.com/grcuda-a-polyglot-language-binding-for-cuda-in-graalvm/).
For more details, see the blog.

## How to run the Example

```console
$ npm i
added 272 packages from 152 contributors and audited 272 packages in 90.639s

26 packages are looking for funding                                                                       run `npm fund` for details

found 0 vulnerabilities

$ node --polyglot --jvm \
  --vm.Dtruffle.class.path.append=../../mxbuild/dists/jdk1.8/grcuda.jar \
  app.js
Mandelbrot app listening on port 3000!
```
