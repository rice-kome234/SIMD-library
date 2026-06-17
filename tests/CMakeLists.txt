add_executable(simd_benchmark
	main.cpp
	benchmark.cpp
	benchmark.h
)

target_link_libraries(simd_benchmark PRIVATE SIMDLibrary::simd)
target_include_directories(simd_benchmark
	PRIVATE
		${PROJECT_SOURCE_DIR}/src
)

set_target_properties(simd_benchmark PROPERTIES
	VS_DEBUGGER_WORKING_DIRECTORY "$<TARGET_FILE_DIR:simd_benchmark>"
)

add_test(NAME simd_benchmark COMMAND simd_benchmark)
set_tests_properties(simd_benchmark PROPERTIES
	LABELS benchmark
	TIMEOUT 300
)
