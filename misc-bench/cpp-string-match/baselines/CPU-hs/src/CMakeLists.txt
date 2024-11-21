SET(EXEC "string-match.out")
SET(SRC "string-match.cpp")
SET(INCLUDE ${CMAKE_SOURCE_DIR}/hyperscan/src ${CMAKE_SOURCE_DIR}/../../../../PIMbench)

# Add the executable for your main program
add_executable(${EXEC} ${SRC})

target_include_directories(${EXEC} PRIVATE ${INCLUDE})

# Link the Hyperscan library to your program
target_link_libraries(${EXEC} PRIVATE hs)

set_target_properties(${EXEC} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)
