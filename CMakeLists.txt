cmake_minimum_required(VERSION 3.20)
project(intruder_alarm CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_compile_options(-Wall -Wextra)

find_package(OpenCV REQUIRED)

add_library(motion STATIC src/motion.cpp)
target_include_directories(motion PUBLIC src ${OpenCV_INCLUDE_DIRS})
target_link_libraries(motion PUBLIC ${OpenCV_LIBS})

enable_testing()
add_executable(test_motion tests/test_motion.cpp)
target_link_libraries(test_motion motion)
add_test(NAME test_motion COMMAND test_motion)
