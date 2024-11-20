cmake_minimum_required(VERSION 3.10)
project(MyHyperscanProject)

# Add the Hyperscan subdirectory
add_subdirectory(hyperscan)

# Add your main program subdirectory
add_subdirectory(src)
