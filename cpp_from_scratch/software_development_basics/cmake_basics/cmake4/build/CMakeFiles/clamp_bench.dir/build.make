# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/codebind/programming-and-maths-Problems/cpp_from_scratch/software_development_basics/cmake_basics/cmake4

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/codebind/programming-and-maths-Problems/cpp_from_scratch/software_development_basics/cmake_basics/cmake4/build

# Include any dependencies generated for this target.
include CMakeFiles/clamp_bench.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/clamp_bench.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/clamp_bench.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/clamp_bench.dir/flags.make

CMakeFiles/clamp_bench.dir/clamp_bench.cpp.o: CMakeFiles/clamp_bench.dir/flags.make
CMakeFiles/clamp_bench.dir/clamp_bench.cpp.o: ../clamp_bench.cpp
CMakeFiles/clamp_bench.dir/clamp_bench.cpp.o: CMakeFiles/clamp_bench.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/codebind/programming-and-maths-Problems/cpp_from_scratch/software_development_basics/cmake_basics/cmake4/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/clamp_bench.dir/clamp_bench.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/clamp_bench.dir/clamp_bench.cpp.o -MF CMakeFiles/clamp_bench.dir/clamp_bench.cpp.o.d -o CMakeFiles/clamp_bench.dir/clamp_bench.cpp.o -c /home/codebind/programming-and-maths-Problems/cpp_from_scratch/software_development_basics/cmake_basics/cmake4/clamp_bench.cpp

CMakeFiles/clamp_bench.dir/clamp_bench.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/clamp_bench.dir/clamp_bench.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/codebind/programming-and-maths-Problems/cpp_from_scratch/software_development_basics/cmake_basics/cmake4/clamp_bench.cpp > CMakeFiles/clamp_bench.dir/clamp_bench.cpp.i

CMakeFiles/clamp_bench.dir/clamp_bench.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/clamp_bench.dir/clamp_bench.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/codebind/programming-and-maths-Problems/cpp_from_scratch/software_development_basics/cmake_basics/cmake4/clamp_bench.cpp -o CMakeFiles/clamp_bench.dir/clamp_bench.cpp.s

# Object files for target clamp_bench
clamp_bench_OBJECTS = \
"CMakeFiles/clamp_bench.dir/clamp_bench.cpp.o"

# External object files for target clamp_bench
clamp_bench_EXTERNAL_OBJECTS =

clamp_bench: CMakeFiles/clamp_bench.dir/clamp_bench.cpp.o
clamp_bench: CMakeFiles/clamp_bench.dir/build.make
clamp_bench: /usr/lib/x86_64-linux-gnu/libbenchmark.so.1.6.1
clamp_bench: CMakeFiles/clamp_bench.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/codebind/programming-and-maths-Problems/cpp_from_scratch/software_development_basics/cmake_basics/cmake4/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable clamp_bench"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/clamp_bench.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/clamp_bench.dir/build: clamp_bench
.PHONY : CMakeFiles/clamp_bench.dir/build

CMakeFiles/clamp_bench.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/clamp_bench.dir/cmake_clean.cmake
.PHONY : CMakeFiles/clamp_bench.dir/clean

CMakeFiles/clamp_bench.dir/depend:
	cd /home/codebind/programming-and-maths-Problems/cpp_from_scratch/software_development_basics/cmake_basics/cmake4/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/codebind/programming-and-maths-Problems/cpp_from_scratch/software_development_basics/cmake_basics/cmake4 /home/codebind/programming-and-maths-Problems/cpp_from_scratch/software_development_basics/cmake_basics/cmake4 /home/codebind/programming-and-maths-Problems/cpp_from_scratch/software_development_basics/cmake_basics/cmake4/build /home/codebind/programming-and-maths-Problems/cpp_from_scratch/software_development_basics/cmake_basics/cmake4/build /home/codebind/programming-and-maths-Problems/cpp_from_scratch/software_development_basics/cmake_basics/cmake4/build/CMakeFiles/clamp_bench.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/clamp_bench.dir/depend

