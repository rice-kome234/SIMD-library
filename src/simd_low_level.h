#pragma once

#include "simd_internal.h"

#include <cassert>
#include <cstddef>

namespace rice::simd::internal
{
struct LowLevelAccess {
	static FloatArray &storage(Array &array) noexcept
	{
		assert(array.impl_ != nullptr);
		return array.impl_->storage;
	}

	static const FloatArray &storage(const Array &array) noexcept
	{
		assert(array.impl_ != nullptr);
		return array.impl_->storage;
	}
};
}

namespace rice::simd::low_level
{
using Block = internal::SimdBlock;

/*!
 *   @brief 1回の内部SIMD処理で扱うfloat要素数
 *   @details ベンチマークや手書きSIMD比較でだけ使う低レベル定数です。
 */
inline constexpr std::size_t SIMD_WIDTH{internal::SIMD_WIDTH};
inline constexpr std::size_t SIMD_ALIGNMENT{internal::SIMD_ALIGNMENT};

/*!
 *   @brief 内部SIMDブロック数を取得
 *   @param[in] array 参照する配列
 *   @return 内部に保持しているSIMDブロック数
 *   @note 通常利用ではArray::size()を使ってください。
 */
inline std::size_t blockCount(const Array &array) noexcept
{
	return internal::LowLevelAccess::storage(array).blockCount();
}

/*!
 *   @brief 指定した内部SIMDブロックへ可変アクセス
 *   @param[in,out] array 参照する配列
 *   @param[in] index 取得する内部ブロック番号
 *   @return 指定した内部ブロックへの参照
 *   @warning 最後のブロックにはパディングレーンが含まれる場合があります。
 */
inline Block &block(Array &array, std::size_t index) noexcept
{
	return internal::LowLevelAccess::storage(array).block(index);
}

/*!
 *   @brief 指定した内部SIMDブロックへ読み取り専用アクセス
 *   @param[in] array 参照する配列
 *   @param[in] index 取得する内部ブロック番号
 *   @return 指定した内部ブロックへの読み取り専用参照
 *   @warning 最後のブロックにはパディングレーンが含まれる場合があります。
 */
inline const Block &block(const Array &array, std::size_t index) noexcept
{
	return internal::LowLevelAccess::storage(array).block(index);
}

/*!
 *   @brief 内部SIMDブロック列の先頭ポインタを取得
 *   @param[in,out] array 参照する配列
 *   @return 内部ストレージの先頭ポインタ
 */
inline Block *data(Array &array) noexcept
{
	return internal::LowLevelAccess::storage(array).data();
}

/*!
 *   @brief 内部SIMDブロック列の読み取り専用先頭ポインタを取得
 *   @param[in] array 参照する配列
 *   @return 内部ストレージの読み取り専用先頭ポインタ
 */
inline const Block *data(const Array &array) noexcept
{
	return internal::LowLevelAccess::storage(array).data();
}

inline Block set1(float value) noexcept { return internal::set1Block(value); }
inline Block loadAligned(const float *values) noexcept
{
	return internal::loadAlignedBlock(values);
}
inline void storeAligned(float *values, Block value) noexcept
{
	internal::storeAlignedBlock(values, value);
}
inline Block add(Block lhs, Block rhs) noexcept { return internal::addBlock(lhs, rhs); }
inline Block mul(Block lhs, Block rhs) noexcept { return internal::mulBlock(lhs, rhs); }
inline Block multiplyAdd(Block lhs, Block rhs, Block addend) noexcept
{
	return internal::multiplyAddBlock(lhs, rhs, addend);
}
}
