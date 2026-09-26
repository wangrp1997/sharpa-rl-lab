include_directories(${PROJECT_SOURCE_DIR})

add_executable(sharpa_wave sharpa_wave.cc)
target_link_libraries(sharpa_wave
        drake::drake
        example_base
        find_resource
        gflags)

configure_file(sharpa_wave.yaml
        ${CMAKE_BINARY_DIR}/idto/examples/sharpa_wave/sharpa_wave.yaml COPYONLY)

add_test(NAME sharpa_wave_test COMMAND sharpa_wave --test)
