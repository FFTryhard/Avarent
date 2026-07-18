cmake_minimum_required(VERSION 3.20)
project(MGA LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(VCPKG_APPLOCAL_DEPS OFF)


set(PYBIND11_FINDPYTHON ON)

find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
find_package(Eigen3 CONFIG)
if(NOT TARGET Eigen3::Eigen)
    include(FetchContent)
    FetchContent_Declare(
      eigen
      GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
      GIT_TAG        3.4.0
    )
    FetchContent_MakeAvailable(eigen)
endif()

set(SRC
    engine.cpp
    utils.cpp
    python_runtime.cpp
)

pybind11_add_module(mga_module
    mainpybind.cpp
    engine.cpp
    utils.cpp
)

target_link_libraries(mga_module PRIVATE
    Eigen3::Eigen
)

target_include_directories(mga_module PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

add_executable(run main.cpp ${SRC}) 

target_link_libraries(run PRIVATE
    Eigen3::Eigen
    pybind11::embed 
)

target_include_directories(run PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)