cmake_minimum_required(VERSION 3.22.1)

project(Math)

# add the library we want to build
add_library(math SHARED add.cpp multiply.cpp)

# Target
add_executable(main main.cpp)

# Specify that we need to link against our created library
target_link_libraries(main math) # Target will depend on library 