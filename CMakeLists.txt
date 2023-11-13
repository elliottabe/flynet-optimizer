cmake_minimum_required (VERSION 3.0)
project(MDL_PSO)

include(ExternalProject)

set(CMAKE_CXX_COMPILER "g++")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -O3")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(PythonLibs 3.8 COMPONENTS Interpreter Development REQUIRED)
find_package(Boost COMPONENTS python numpy REQUIRED)
find_package(Armadillo REQUIRED)
find_package(VTK REQUIRED)
include(${VTK_USE_FILE})

include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(${Boost_INCLUDE_DIR})
include_directories(${VTK_DIR})

# Build and link the pylib module
add_library(_mdl_pso_lib SHARED flynet_optimizer/mdl_pso_lib.cpp flynet_optimizer/model_pso.cpp)
target_link_libraries(
    _mdl_pso_lib 
    ${Boost_LIBRARIES} 
    ${PYTHON_LIBRARIES} 
    ${ARMADILLO_LIBRARIES} 
    ${PNG_LIBRARY} 
    ${VTK_LIBRARIES} 
    tbb
    )

# Tweaks the name of the library to match what Python expects
set_target_properties(_mdl_pso_lib PROPERTIES SUFFIX .so)
set_target_properties(_mdl_pso_lib PROPERTIES PREFIX "")

install(TARGETS _mdl_pso_lib LIBRARY DESTINATION flynet_optimizer)
