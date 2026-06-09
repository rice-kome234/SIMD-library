cmake_minimum_required(VERSION 3.20)

project(SIMDLibrary VERSION 0.1.0 LANGUAGES CXX)

include(CMakePackageConfigHelpers)

option(SIMD_LIBRARY_BUILD_TESTS "Build benchmark/test executable" ON)

add_library(simd STATIC
	src/simd.cpp
	include/simd.h
	src/simd_debug.h
	src/simd_internal.h
	src/simd_low_level.h
)
add_library(SIMDLibrary::simd ALIAS simd)

target_compile_features(simd PUBLIC cxx_std_20)
target_include_directories(simd
	PUBLIC
		$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
		$<INSTALL_INTERFACE:include>
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
	PUBLIC_HEADER include/simd.h
)

set(SIMD_LIBRARY_DIST_DIR ${CMAKE_CURRENT_SOURCE_DIR}/dist)

add_custom_target(simd_dist
	COMMAND ${CMAKE_COMMAND} -E rm -rf ${SIMD_LIBRARY_DIST_DIR}
	COMMAND ${CMAKE_COMMAND} -E make_directory ${SIMD_LIBRARY_DIST_DIR}
	COMMAND ${CMAKE_COMMAND} -E copy_if_different
		${CMAKE_CURRENT_SOURCE_DIR}/include/simd.h
		${SIMD_LIBRARY_DIST_DIR}/simd.h
	COMMAND ${CMAKE_COMMAND} -E copy_if_different
		$<TARGET_FILE:simd>
		${SIMD_LIBRARY_DIST_DIR}/$<TARGET_FILE_NAME:simd>
	DEPENDS simd
	COMMENT "Create minimal distribution package in dist/"
)

if(SIMD_LIBRARY_BUILD_TESTS)
	include(CTest)
	add_subdirectory(tests)
endif()

install(TARGETS simd
	EXPORT SIMDLibraryTargets
	ARCHIVE DESTINATION lib
	LIBRARY DESTINATION lib
	RUNTIME DESTINATION bin
	PUBLIC_HEADER DESTINATION include
)

install(EXPORT SIMDLibraryTargets
	NAMESPACE SIMDLibrary::
	DESTINATION lib/cmake/SIMDLibrary
)

configure_package_config_file(
	cmake/SIMDLibraryConfig.cmake.in
	${CMAKE_CURRENT_BINARY_DIR}/SIMDLibraryConfig.cmake
	INSTALL_DESTINATION lib/cmake/SIMDLibrary
)

write_basic_package_version_file(
	${CMAKE_CURRENT_BINARY_DIR}/SIMDLibraryConfigVersion.cmake
	VERSION ${PROJECT_VERSION}
	COMPATIBILITY SameMajorVersion
)

install(FILES
	${CMAKE_CURRENT_BINARY_DIR}/SIMDLibraryConfig.cmake
	${CMAKE_CURRENT_BINARY_DIR}/SIMDLibraryConfigVersion.cmake
	DESTINATION lib/cmake/SIMDLibrary
)
