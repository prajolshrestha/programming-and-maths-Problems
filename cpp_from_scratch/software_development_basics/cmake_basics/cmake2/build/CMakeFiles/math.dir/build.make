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
CMAKE_SOURCE_DIR = /home/codebind/programming-and-maths-Problems/cpp_from_scratch/software_development_basics/cmake_basics/cmake2

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/codebind/programming-and-maths-Problems/cpp_from_scratch/software_development_basics/cmake_basics/cmake2/build

# Include any dependencies generated for this target.
include CMakeFiles/math.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/math.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/math.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/math.dir/flags.make

CMakeFiles/math.dir/add.cpp.o: CMakeFiles/math.dir/flags.make
CMakeFiles/math.dir/add.cpp.o: ../add.cpp
CMakeFiles/math.dir/add.cpp.o: CMakeFiles/math.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/codebind/programming-and-maths-Problems/cpp_from_scratch/software_development_basics/cmake_basics/cmake2/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/math.dir/add.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/math.dir/add.cpp.o -MF CMakeFiles/math.dir/add.cpp.o.d -o CMakeFiles/math.dir/add.cpp.o -c /home/codebind/programming-and-maths-Problems/cpp_from_scratch/software_development_basics/cmake_basics/cmake2/add.cpp

CMakeFiles/math.dir/add.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/math.dir/add.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/codebind/programming-and-maths-Problems/cpp_from_scratch/software_development_basics/cmake_basics/cmake2/add.cpp > CMakeFiles/math.dir/add.cpp.i

CMakeFiles/math.dir/add.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/math.dir/add.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/codebind/programming-and-maths-Problems/cpp_from_scratch/software_development_basics/cmake_basics/cmake2/add.cpp -o CMakeFiles/math.dir/add.cpp.s

CMakeFiles/math.dir/multiply.cpp.o: CMakeFiles/math.dir/flags.make
CMakeFiles/math.dir/multiply.cpp.o: ../multiply.cpp
CMakeFiles/math.dir/multiply.cpp.o: CMakeFiles/math.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/codebind/programming-and-maths-Problems/cpp_from_scratch/software_development_basics/cmake_basics/cmake2/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/math.dir/multiply.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/math.dir/multiply.cpp.o -MF CMakeFiles/math.dir/multiply.cpp.o.d -o CMakeFiles/math.dir/multiply.cpp.o -c /home/codebind/programming-and-maths-Problems/cpp_from_scratch/software_development_basics/cmake_basics/cmake2/multiply.cpp

CMakeFiles/math.dir/multiply.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/math.dir/multiply.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/codebind/programming-and-maths-Problems/cpp_from_scratch/software_development_basics/cmake_basics/cmake2/multiply.cpp > CMakeFiles/math.dir/multiply.cpp.i

CMakeFiles/math.dir/multiply.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/math.dir/multiply.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/codebind/programming-and-maths-Problems/cpp_from_scratch/software_development_basics/cmake_basics/cmake2/multiply.cpp -o CMakeFiles/math.dir/multiply.cpp.s

# Object files for target math
math_OBJECTS = \
"CMakeFiles/math.dir/add.cpp.o" \
"CMakeFiles/math.dir/multiply.cpp.o"

# External object files for target math
math_EXTERNAL_OBJECTS =

libmath.so: CMakeFiles/math.dir/add.cpp.o
libmath.so: CMakeFiles/math.dir/multiply.cpp.o
libmath.so: CMakeFiles/math.dir/build.make
libmath.so: CMakeFiles/math.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/codebind/programming-and-maths-Problems/cpp_from_scratch/software_development_basics/cmake_basics/cmake2/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX shared library libmath.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/math.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/math.dir/build: libmath.so
.PHONY : CMakeFiles/math.dir/build

CMakeFiles/math.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/math.dir/cmake_clean.cmake
.PHONY : CMakeFiles/math.dir/clean

CMakeFiles/math.dir/depend:
	cd /home/codebind/programming-and-maths-Problems/cpp_from_scratch/software_development_basics/cmake_basics/cmake2/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/codebind/programming-and-maths-Problems/cpp_from_scratch/software_development_basics/cmake_basics/cmake2 /home/codebind/programming-and-maths-Problems/cpp_from_scratch/software_development_basics/cmake_basics/cmake2 /home/codebind/programming-and-maths-Problems/cpp_from_scratch/software_development_basics/cmake_basics/cmake2/build /home/codebind/programming-and-maths-Problems/cpp_from_scratch/software_development_basics/cmake_basics/cmake2/build /home/codebind/programming-and-maths-Problems/cpp_from_scratch/software_development_basics/cmake_basics/cmake2/build/CMakeFiles/math.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/math.dir/depend

