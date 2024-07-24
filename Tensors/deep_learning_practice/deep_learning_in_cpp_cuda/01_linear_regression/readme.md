# Setting up CUDA PyTorch C++ on HPC

1. Connect to the HPC:
   ```
   ssh your_username@hpc_address
   ```

2. Load required modules:
   Most HPCs use a module system to manage software. You'll typically need to load modules for CUDA, CMake, and a compatible compiler.
   ```bash
   module load cuda/11.3  # Replace with available version
   module load cmake/3.21.0
   module load gcc/9.3.0  # Or another compatible compiler
   ```

3. Download and extract libtorch:
   ```bash
   wget https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.3.1%2Bcu118.zip
   unzip libtorch-cxx11-abi-shared-with-deps-2.3.1+cu118.zip
   ```
   Note: Replace the URL with the version compatible with your CUDA version.

4. Set up environment variables:
   Add these to your `~/.bashrc` or load them in your job script:
   ```bash
   export TORCH_PATH=$HOME/libtorch  # Adjust path as needed
   export LD_LIBRARY_PATH=$TORCH_PATH/lib:$LD_LIBRARY_PATH
   ```

5. Create your C++ source file (e.g., `linear_regression.cpp`) and `CMakeLists.txt` as in previous examples.

6. Build your project:
   ```bash
   mkdir build && cd build
   cmake -DCMAKE_PREFIX_PATH=$TORCH_PATH ..
   make
   ```

   if errors remove previous builds first and repeat the build process above:
   ```bash
   cd build
   make clean
   cmake ..
   make
   ```

7. Create a job submission script (e.g., `job.sh`):
   ```bash
   #!/bin/bash
   #SBATCH --job-name=linear_regression
   #SBATCH --output=result.out
   #SBATCH --error=result.err
   #SBATCH --time=00:10:00
   #SBATCH --ntasks=1
   #SBATCH --gpus=1

   module load cuda/11.3  # Match with your setup
   module load cmake/3.21.0
   module load gcc/9.3.0

   export TORCH_PATH=$HOME/libtorch
   export LD_LIBRARY_PATH=$TORCH_PATH/lib:$LD_LIBRARY_PATH

   ./linear_regression
   ```

8. Submit your job:
   ```bash
   sbatch job.sh
   ```

9. Monitor your job:
   ```bash
   squeue -u your_username
   ```

10. Check results:
    ```bash
    cat result.out
    ```

Note: 
- The exact commands and available software versions may vary depending on your specific HPC setup.
- Always consult your HPC's documentation or support team for system-specific instructions.
- Be mindful of resource allocation and job scheduling policies on the HPC.