# Add the executable for your main program
add_executable(my_program main.cpp)

target_include_directories(my_program PRIVATE ${CMAKE_SOURCE_DIR}/hyperscan/src)

# Link the Hyperscan library to your program
target_link_libraries(my_program PRIVATE hs)
