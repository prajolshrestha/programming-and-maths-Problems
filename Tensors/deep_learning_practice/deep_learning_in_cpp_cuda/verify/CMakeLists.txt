cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(verify_torch)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Torch REQUIRED)

# Add your executable
add_executable(verify_torch verify_torch.cpp)

# Link against PyTorch libraries
target_link_libraries(verify_torch "${TORCH_LIBRARIES}")

# Set C++ flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")