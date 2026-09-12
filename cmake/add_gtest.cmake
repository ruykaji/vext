function(add_gtest name file)
    add_executable(${name} ${file})
    target_link_libraries(${name} GTest::gtest_main ${ARGN})

    if(file MATCHES "\\.cu$")
        vext_apply_project_options(${name} NO_SANITIZERS)
    else()
        vext_apply_project_options(${name})
    endif()
    
    gtest_discover_tests(${name} DISCOVERY_TIMEOUT 600 PROPERTIES TIMEOUT 0)
endfunction()
