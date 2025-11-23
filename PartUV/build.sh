
target_path=all_build/release
main_file=main.cpp

rm -rf $target_path
mkdir -p $target_path
cd  $target_path
cmake   -DMAIN_FILE=$main_file \
        -DCMAKE_BUILD_TYPE=Release \
        -DUSE_ALL_SRC_FILES=ON \
        ../.. 
        
make -j