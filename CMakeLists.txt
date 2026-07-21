cmake_minimum_required(VERSION 3.20)

if(POLICY CMP0091)
	cmake_policy(SET CMP0091 NEW)
endif()

project(SIMDLibrary VERSION 0.1.0 LANGUAGES CXX)

option(SIMD_LIBRARY_BUILD_TESTS "Build benchmark/test executable" ON)
option(SIMD_LIBRARY_BUILD_EXAMPLES "Build example executable" OFF)
option(SIMD_LIBRARY_DIST_ON_BUILD "Build dist package with the default build target" OFF)
set(SIMD_LIBRARY_DIST_NAME "manual" CACHE STRING "Dist subdirectory name")

add_library(simd STATIC
	src/simd.cpp
	include/simd.h
	src/simd_backend.h
	src/simd_debug.h
	src/simd_internal.h
	src/simd_low_level.h
)
add_library(SIMDLibrary::simd ALIAS simd)

target_compile_features(simd PUBLIC cxx_std_20)
target_include_directories(simd
	PUBLIC
		$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
	PRIVATE
		${CMAKE_CURRENT_SOURCE_DIR}/src
)

if(MSVC)
	target_compile_options(simd PUBLIC /arch:AVX2 /utf-8)
else()
	target_compile_options(simd PUBLIC -mavx2 -mfma)
endif()

set_target_properties(simd PROPERTIES
	OUTPUT_NAME simd
)

if(SIMD_LIBRARY_DIST_NAME STREQUAL "")
	set(SIMD_LIBRARY_DIST_DIR ${CMAKE_CURRENT_SOURCE_DIR}/dist)
else()
	set(SIMD_LIBRARY_DIST_DIR ${CMAKE_CURRENT_SOURCE_DIR}/dist/${SIMD_LIBRARY_DIST_NAME})
endif()

set(SIMD_LIBRARY_DIST_DEFAULT_TARGET)
if(SIMD_LIBRARY_DIST_ON_BUILD)
	set(SIMD_LIBRARY_DIST_DEFAULT_TARGET ALL)
endif()

add_custom_target(simd_dist ${SIMD_LIBRARY_DIST_DEFAULT_TARGET}
	COMMAND ${CMAKE_COMMAND} -E rm -rf ${SIMD_LIBRARY_DIST_DIR}
	COMMAND ${CMAKE_COMMAND} -E make_directory ${SIMD_LIBRARY_DIST_DIR}
	COMMAND ${CMAKE_COMMAND} -E copy_if_different
		${CMAKE_CURRENT_SOURCE_DIR}/include/simd.h
		${SIMD_LIBRARY_DIST_DIR}/simd.h
	COMMAND ${CMAKE_COMMAND} -E copy_if_different
		$<TARGET_FILE:simd>
		${SIMD_LIBRARY_DIST_DIR}/$<TARGET_FILE_NAME:simd>
	DEPENDS simd
	COMMENT "Create SIMD library package"
)

if(SIMD_LIBRARY_BUILD_TESTS)
	include(CTest)
	add_subdirectory(tests)

	if(TARGET simd_benchmark)
		set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY
			VS_STARTUP_PROJECT simd_benchmark
		)
	endif()
endif()

if(SIMD_LIBRARY_BUILD_EXAMPLES)
	add_executable(simd_minimal_example
		examples/minimal.cpp
	)
	target_link_libraries(simd_minimal_example PRIVATE simd)

	add_custom_target(run_simd_minimal_example
		COMMAND $<TARGET_FILE:simd_minimal_example>
		DEPENDS simd_minimal_example
		WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
		COMMENT "Run minimal SIMD example"
		USES_TERMINAL
	)
endif()
