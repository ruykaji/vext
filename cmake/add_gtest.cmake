function(add_gtest name file)
    add_executable(${name} ${file})
    target_link_libraries(${name} GTest::gtest_main ${ARGN})
    gtest_discover_tests(${name} DISCOVERY_TIMEOUT 600 PROPERTIES TIMEOUT 0)
endfunction()

