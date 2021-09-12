#include <string>
#include <iostream>
#include <ctime>    // For time()
#include "options.hpp"
#include "opencv_interface.hpp"
#include "image_pipeline.cuh"

// Remember to turn persistence-mode on for GPUs: sudo nvidia-smi -pm 1
// Install OpenCV following: https://docs.opencv.org/master/d7/d9f/tutorial_linux_install.html

int main(int argc, char *argv[])
{ 
    Options options = Options(argc, argv);
    OpenCVInterface interface = OpenCVInterface(options);
    ImagePipeline pipeline = ImagePipeline(options);

    // Read the image;
    auto* img = interface.read_input();
    // Process the image;
    pipeline.run(img);
    // Write the image to output;
    interface.write_output(img);
}
