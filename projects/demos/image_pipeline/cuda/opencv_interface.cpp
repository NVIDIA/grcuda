
#include "opencv_interface.hpp"

uchar* OpenCVInterface::read_input() {
    auto start = clock_type::now();
    // Read image;
    std::string input_path;
    if (!full_input_path.empty()) {
        input_path = full_input_path;
    } else {
        std::stringstream ss;
        ss << "../../" << INPUT_IMAGE_FOLDER << "/" << image_name << ".jpg";
        input_path = ss.str();
    }
    image_matrix = imread(input_path,  black_and_white ? cv::IMREAD_GRAYSCALE : cv::IMREAD_COLOR);
    
    if (debug) std::cout << "loaded image=" << image_name << " of size " << image_matrix.rows << "x" << image_matrix.cols << std::endl;
    // Resize image if necessary;
    bool resized = false;
    if (image_matrix.rows != image_width || image_matrix.cols != image_width) {
        cv::resize(image_matrix, resized_image, cv::Size(image_width, image_width));
        if (debug) std::cout << "resized image to " << image_width << "x" << image_width << std::endl;
        resized = true;
    }
    auto end = clock_type::now();
    if (debug) std::cout << "read image time=" << chrono::duration_cast<chrono::microseconds>(end - start).count() / 1000 << " ms" << std::endl;

    if (resized) {
        image_array_length = resized_image.total() * resized_image.channels();
        return resized_image.data;
    } else {
        image_array_length = image_matrix.total() * image_matrix.channels();
        return image_matrix.data;
    }
}

void OpenCVInterface::write_output_inner(std::string kind, int resize_width) {
    // Resize image;
    cv::Mat image_matrix_out;
    cv::resize(output_matrix, image_matrix_out, cv::Size(resize_width, resize_width));
    // Write to file;
    std::vector<int> compression_params;
    compression_params.push_back(cv::IMWRITE_JPEG_QUALITY);
    compression_params.push_back(80);

    std::string output_path;
    if (kind == "small" && !full_output_path_small.empty()) {
        output_path = full_output_path_small;
    } else if (kind == "large" && !full_output_path_large.empty()) {
        output_path = full_output_path_large;
    } else if (!image_name.empty()) {
         std::stringstream ss;
        ss << "../../" << OUTPUT_IMAGE_FOLDER << "/" << image_name << "_" << kind << ".jpg";
        output_path = ss.str();
    } else {
        if (debug) std::cout << "error: missing output path or image name, cannot write output" << std::endl;
        return;
    }
    imwrite(output_path, image_matrix_out, compression_params);
}

void OpenCVInterface::write_output(unsigned char* buffer) {
    auto start = clock_type::now();
    // Turn buffer into matrix;
    output_matrix = cv::Mat(image_width, image_width, black_and_white ? CV_8UC1 : CV_8UC3, buffer);
    // Write to output;
    write_output_inner("large", RESIZED_IMAGE_WIDTH_OUT_LARGE);
    write_output_inner("small", RESIZED_IMAGE_WIDTH_OUT_SMALL);
    auto end = clock_type::now();
    if (debug) std::cout << "write image time=" << chrono::duration_cast<chrono::microseconds>(end - start).count() / 1000 << " ms" << std::endl;
}