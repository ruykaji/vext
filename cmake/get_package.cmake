function(get_package package_name)
   set(version "" CACHE STRING "Optional version argument")

   # Find package that have .cmake configuration file
   if(TRUE)
      if(NOT version STREQUAL "")
         find_package(${package_name} QUIET ${version})
      else()
         find_package(${package_name} QUIET)
      endif()

      if(${package_name}_FOUND)
         message(STATUS "Found ${package_name}.")
         return()
      endif()
   endif()

   # Find package that don't have .cmake configuration file
   if(TRUE)
      find_library(${package_name}_LIBRARY NAMES ${package_name})
      if(${package_name}_LIBRARY)
         message(STATUS "Found ${package_name}.")
         return()
      endif()
   endif()

   message(FATAL_ERROR "Package ${package_name} not found.")
endfunction()
