#include <torch/torch.h>
#include <iostream>

int main() {
    // Check CUDA availability
    if (torch::cuda::is_available()) {
        std::cout << "CUDA is available! GPU support enabled." << std::endl;

        // Get the number of CUDA devices
        int device_count = torch::cuda::device_count();
        std::cout << "Number of CUDA devices: " << device_count << std::endl;

        for (int i = 0; i < device_count; ++i) {
            auto device_properties = torch::cuda::getDeviceProperties(i);
            std::cout << "CUDA device [" << i << "]: " << device_properties.name << std::endl;
        }

        // Get the current CUDA device index
        int current_device = torch::cuda::current_device();
        std::cout << "Current CUDA device index: " << current_device << std::endl;
        auto current_device_properties = torch::cuda::getDeviceProperties(current_device);
        std::cout << "Current CUDA device name: " << current_device_properties.name << std::endl;
    } else {
        std::cout << "CUDA is not available. Running on CPU." << std::endl;
    }

    // Create a tensor
    torch::Tensor tensor = torch::rand({2, 3});
    std::cout << "Random Tensor:" << std::endl << tensor << std::endl;

    // Perform a simple operation
    torch::Tensor result = tensor * 2;
    std::cout << "Tensor multiplied by 2:" << std::endl << result << std::endl;

    // Check libtorch version
    std::cout << "PyTorch version: " << TORCH_VERSION << std::endl;

    return 0;
}
