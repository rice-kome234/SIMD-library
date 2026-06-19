#pragma once

#include <cstddef>

namespace simdbench
{
/*!
 *   @brief 式融合ベンチマークの結果
 *   @details
 *   `ab = a * b` を共有し、`x = ab + c`、`y = ab + d`、`z = ab + e`
 *   という3つの式を、非SIMDスカラー実装、手書きSIMD実装、DirectXMath実装、
 *   通常式APIで比較します。
 */
struct FusedExpressionBenchmarkResult {
	std::size_t elementCount{};
	int repeatCount{};

	double scalarMs{};
	double manualSimdMs{};
	double directXMathMs{};
	double normalExpressionMs{};

	float scalarXError{};
	float directXMathXError{};
	float expressionXError{};
};

/*!
 *   @brief 3成分更新ベンチマークの結果
 *   @details
 *   位置と速度の更新を、非SIMDスカラー実装、手書きSIMD実装、DirectXMath実装、
 *   通常式APIで比較します。
 */
struct ThreeComponentUpdateBenchmarkResult {
	std::size_t elementCount{};
	int repeatCount{};
	float deltaTime{};

	double scalarMs{};
	double manualSimdMs{};
	double directXMathMs{};
	double specializedUpdateMs{};

	float scalarPositionXError{};
	float directXMathPositionXError{};
	float specializedPositionXError{};
};

/*!
 *   @brief 遅延実行で複数の独立した出力をまとめるベンチマークの結果
 *   @details
 *   同じ入力から複数の出力配列を作る処理を、1つずつexecute()する場合と、
 *   まとめて1回のexecute()で処理する場合で比較します。
 */
struct BatchedExecutionBenchmarkResult {
	std::size_t elementCount{};
	std::size_t outputCount{};
	int repeatCount{};

	double scalarMs{};
	double manualSimdMs{};
	double directXMathMs{};
	double sequentialExpressionMs{};
	double batchedExpressionMs{};

	float manualSimdFirstOutputError{};
	float directXMathFirstOutputError{};
	float sequentialFirstOutputError{};
	float batchedFirstOutputError{};
};

/*!
 *   @brief 重い式ベンチマークの結果
 *   @details
 *   乗算と加算を複数含む式を、非SIMDスカラー実装、手書きSIMD実装、
 *   DirectXMath実装、通常式APIで比較します。
 */
struct HeavyExpressionBenchmarkResult {
	std::size_t elementCount{};
	int repeatCount{};

	double scalarMs{};
	double manualSimdMs{};
	double directXMathMs{};
	double expressionMs{};

	float manualSimdError{};
	float directXMathError{};
	float expressionError{};
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
 *   @brief 位置と速度の更新ベンチマークを実行
 *   @param[in] elementCount 処理する要素数
 *   @param[in] repeatCount 同じ更新処理を繰り返す回数
 *   @param[in] deltaTime 速度と位置の更新に使う時間差分
 *   @return 計測時間と誤差をまとめた結果
 *   @note スカラー実装は自動ベクトル化を抑え、SIMDを使わない基準値として扱います。
 */
ThreeComponentUpdateBenchmarkResult
runThreeComponentUpdateBenchmark(std::size_t elementCount, int repeatCount, float deltaTime);

/*!
 *   @brief 遅延実行で複数の独立出力をまとめるベンチマークを実行
 *   @param[in] elementCount 1つの出力配列あたりのfloat要素数
 *   @param[in] outputCount まとめて処理する出力配列数
 *   @param[in] repeatCount 同じ処理を繰り返す回数
 *   @return 計測時間と誤差をまとめた結果
 */
BatchedExecutionBenchmarkResult runBatchedExecutionBenchmark(std::size_t elementCount,
                                                             std::size_t outputCount,
                                                             int repeatCount);

/*!
 *   @brief 重い式ベンチマークを実行
 *   @param[in] elementCount 処理するfloat要素数
 *   @param[in] repeatCount 同じ処理を繰り返す回数
 *   @return 計測時間と誤差をまとめた結果
 */
HeavyExpressionBenchmarkResult runHeavyExpressionBenchmark(std::size_t elementCount,
                                                           int repeatCount);
}
