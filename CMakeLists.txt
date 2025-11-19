cmake_minimum_required(VERSION 3.5)
project(MEEM)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# pybind11
add_subdirectory(pybind11)

include_directories("C:/msys64/ucrt64/include/eigen3")

# executable
add_executable(main_exec1 cppumap.cpp utilspy.cpp) 






# link against Python and pybind11
target_link_libraries(main_exec1 PRIVATE pybind11::embed)

# add Python includes/libs
target_include_directories(main_exec1 PRIVATE "C:/Users/H/AppData/Local/Programs/Python/Python312/include")
target_link_directories(main_exec1 PRIVATE "C:/Users/H/AppData/Local/Programs/Python/Python312/libs")

target_link_libraries(main_exec1 PRIVATE python312)
