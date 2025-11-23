# Building PartUV

## 1) Manage dependencies

### Update code with external libraries (with submodules, recursively)

```bash
# first time
git clone --recurse-submodules https://github.com/EricWang12/partuv.git

# Or later updates
git pull
git submodule update --init --recursive
```

### Install additional dependencies

```bash
sudo apt-get update
sudo apt-get install libcgal-dev libyaml-cpp-dev libtbb-dev -y
```


## 2) Build Python (wheel or dev)

```bash
# Install build dependencies
python -m pip install -U pip wheel build  pybind11[global]  scikit-build-core

python -m pip install -e . --no-build-isolation -v
```


## 3) Build C++ test program

We provide the CMakeLists.txt for building the C++ code in CMakeLists_main.txt. Please modify the CMakeLists.txt to your needs.

Then you can build the main.cpp with the following command:

```bash
target_path=all_build/release
main_file=main.cpp

rm -rf $target_path
mkdir -p $target_path
cd  $target_path
cmake   -DMAIN_FILE=$main_file \
        -DCMAKE_BUILD_TYPE=Release \
        -DUSE_ALL_SRC_FILES=ON \
        -DCMAKE_CUDA_COMPILER=/usr/bin/nvcc \
        ../.. 
make -j

```


This would build a standalone executable in the target_path/release/program. 

you can run the executable with the following command:
```bash
./program mesh_path/{preprocessed_mesh_name}.obj
```
given the following directory layout:

```text
mesh_path/
├─ {preprocessed_mesh_name}.obj
└─ bin/
   └─ {preprocessed_mesh_name}.bin
```


Here the mesh is the preprocessed mesh from the preprocess.py script, and bin is the binary file representing the part field tree hierarchy.
