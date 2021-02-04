cmake_minimum_required(VERSION 3.10)
project(bench)

set(CMAKE_CXX_STANDARD 17)

add_subdirectory(deps/bvh)

add_executable(bench bench.cpp ScopedTimer.cpp)
target_compile_definitions(bench PUBLIC -DASSET_PATH="${PROJECT_SOURCE_DIR}/assets/lucy.ply")
target_link_libraries(bench PRIVATE bvh)
