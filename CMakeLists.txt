cmake_minimum_required(VERSION 3.23)
project(AI)

set(CMAKE_CXX_STANDARD 23)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_DEBUG "-g -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

#file(GLOB SOURCE_FILES src/*.hpp src/*.cpp)

add_compile_definitions(USE_PARALLEL)

add_executable(AI src/main.cpp
        src/containers/tensor.hpp src/containers/tensor.tpp
        src/activation.hpp
        src/cost.hpp
        src/layer.hpp
        src/optimizer.hpp
        src/dataHandler.cpp src/dataHandler.hpp src/containers/tensor.cpp
        src/activations/sigmoid.hpp src/activations/sigmoid.cpp
        src/costs/meanSquaredError.hpp src/costs/meanSquaredError.cpp
        src/layers/denseLayer.hpp src/layers/denseLayer.cpp
        src/layers/inputLayer.hpp src/layers/inputLayer.cpp
        src/network.hpp src/network.cpp
        src/optimizers/sgd.cpp src/optimizers/sgd.hpp)

