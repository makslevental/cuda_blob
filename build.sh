cd build || exit
cmake ..
cmake --build . --config Debug -- -j$(nproc)