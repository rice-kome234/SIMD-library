#pragma once

#include <cstddef>

namespace simdbench
{
/*!
 *   @brief 式融合ベンチマークの結果
 *   @details
 *   `x = a * b + c`、`y = a * b + d`、`z = a * b + e` という3つの式を、
 *   非SIMDスカラー実装、手書きSIMD実装、通常式APIで比較します。
 */
struct FusedExpressionBenchmarkResult {
	std::size_t elementCount{};
	int repeatCount{};

	double scalarMs{};
	double manualSimdMs{};
	double normalExpressionMs{};

	float scalarXError{};
	float expressionXError{};
};

/*!
 *   @brief 3成分更新ベンチマークの結果
 *   @details
 *   位置と速度の更新を、非SIMDスカラー実装、手書き3成分SIMD実装、
 *   ベンチマーク内部の専用更新コードで比較します。
 */
struct ThreeComponentUpdateBenchmarkResult {
	std::size_t elementCount{};
	int repeatCount{};
	float deltaTime{};

	double scalarMs{};
	double manualSimdMs{};
	double specializedUpdateMs{};

	float scalarPositionXError{};
	float specializedPositionXError{};
};

/*!
 *   @brief 式融合ベンチマークを実行
 *   @param[in] elementCount 処理するfloat要素数
 *   @param[in] repeatCount 同じ処理を繰り返す回数
 *   @return 計測時間と数値誤差をまとめた結果
 *   @note 乱数シードは関数内で固定しているため、実行ごとの入力は再現可能です。
 */
FusedExpressionBenchmarkResult runFusedExpressionBenchmark(std::size_t elementCount,
                                                           int repeatCount);

/*!
 *   @brief 3成分更新ベンチマークを実行
 *   @param[in] elementCount 処理する3成分要素数
 *   @param[in] repeatCount 同じ更新処理を繰り返す回数
 *   @param[in] deltaTime 速度と位置の更新に使う時間差分
 *   @return 計測時間と誤差をまとめた結果
 *   @note スカラー実装は自動ベクトル化を抑え、SIMDを使わない基準値として扱います。
 */
ThreeComponentUpdateBenchmarkResult
runThreeComponentUpdateBenchmark(std::size_t elementCount, int repeatCount, float deltaTime);
}
