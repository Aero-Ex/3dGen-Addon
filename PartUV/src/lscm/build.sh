rm -rf build
mkdir build
cd build
cmake ..
make
./main /ariesdv0/zhaoning/workspace/IUV/lscm/data/part_1_processed.obj ./ear_out.obj
# ./main /ariesdv0/zhaoning/workspace/IUV/lscm/data/part_1.obj ./ear_out.obj igl
# ./main /ariesdv0/zhaoning/workspace/IUV/lscm/data/bear_head_optcut.obj ./ear_out.obj igl
