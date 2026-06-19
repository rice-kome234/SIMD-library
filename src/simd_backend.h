#pragma once

#include <cstddef>
#include <immintrin.h>

namespace rice::simd::internal
{
using SimdBlock = __m256;
inline constexpr std::size_t SIMD_WIDTH{8};
inline constexpr std::size_t SIMD_ALIGNMENT{32};

inline SimdBlock zeroBlock() noexcept { return _mm256_setzero_ps(); }
inline SimdBlock set1Block(float value) noexcept { return _mm256_set1_ps(value); }
inline SimdBlock loadAlignedBlock(const float *values) noexcept { return _mm256_load_ps(values); }
inline SimdBlock loadUnalignedBlock(const float *values) noexcept
{
	return _mm256_loadu_ps(values);
}
inline void storeAlignedBlock(float *values, SimdBlock block) noexcept
{
	_mm256_store_ps(values, block);
}
inline void storeUnalignedBlock(float *values, SimdBlock block) noexcept
{
	_mm256_storeu_ps(values, block);
}
inline SimdBlock addBlock(SimdBlock a, SimdBlock b) noexcept { return _mm256_add_ps(a, b); }
inline SimdBlock subBlock(SimdBlock a, SimdBlock b) noexcept { return _mm256_sub_ps(a, b); }
inline SimdBlock mulBlock(SimdBlock a, SimdBlock b) noexcept { return _mm256_mul_ps(a, b); }
inline SimdBlock divBlock(SimdBlock a, SimdBlock b) noexcept { return _mm256_div_ps(a, b); }
inline SimdBlock multiplyAddBlock(SimdBlock a, SimdBlock b, SimdBlock c) noexcept
{
	return _mm256_fmadd_ps(a, b, c);
}
inline SimdBlock negativeMultiplyAddBlock(SimdBlock a, SimdBlock b, SimdBlock c) noexcept
{
	return _mm256_fnmadd_ps(a, b, c);
}
}
