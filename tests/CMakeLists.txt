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

add_test(NAME simd_benchmark COMMAND simd_benchmark)
set_tests_properties(simd_benchmark PROPERTIES
	LABELS benchmark
	TIMEOUT 300
)
