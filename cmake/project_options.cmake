include_guard(GLOBAL)

function(vext_apply_project_options target)
	cmake_parse_arguments(ARG "NO_SANITIZERS" "" "" ${ARGN})

	target_compile_options(${target} 
        PRIVATE
		    $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang>:-Wall;-Wextra;-Wpedantic>
		    $<$<AND:$<BOOL:${VEXT_WARNINGS_AS_ERRORS}>,$<COMPILE_LANG_AND_ID:CXX,GNU,Clang>>:-Werror>
    )

	if(VEXT_ENABLE_SANITIZERS AND NOT ARG_NO_SANITIZERS)
		if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
			target_compile_options(${target} 
                PRIVATE
				    $<$<COMPILE_LANGUAGE:CXX>:-fsanitize=address,undefined;-fno-omit-frame-pointer>
            )
			target_link_options(${target} 
                PRIVATE 
                -fsanitize=address,undefined
            )
		else()
			message(WARNING "Sanitizers are only configured for GNU and Clang")
		endif()
	endif()
endfunction()
