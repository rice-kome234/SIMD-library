#include "benchmark.h"

#if defined(_WIN32)
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

#include <cstddef>
#include <iostream>

namespace
{
// ここを変更すると、すべてのベンチマーク条件をまとめて調整できます。
constexpr std::size_t ELEMENT_COUNT{static_cast<std::size_t>(1) << 21};
constexpr int REPEAT_COUNT{512};
constexpr float DELTA_TIME{0.016f};
constexpr std::size_t BATCHED_OUTPUT_COUNT{16};

void setupConsoleEncoding()
{
#if defined(_WIN32)
	// Visual Studioのデバッグ実行では既定の文字コードがUTF-8ではない場合があります。
	SetConsoleOutputCP(CP_UTF8);
	SetConsoleCP(CP_UTF8);
#endif
}
}

int main()
{
	setupConsoleEncoding();

	std::cout << "ベンチマークの開始" << std::endl;

	// 通常式でのベンチマーク
	const simdbench::FusedExpressionBenchmarkResult fused{
	    simdbench::runFusedExpressionBenchmark(ELEMENT_COUNT, REPEAT_COUNT)};

	// 位置と速度をまとめて更新するベンチマーク
	const simdbench::ThreeComponentUpdateBenchmarkResult motionUpdate{
	    simdbench::runThreeComponentUpdateBenchmark(ELEMENT_COUNT, REPEAT_COUNT, DELTA_TIME)};

	// 複数の独立した出力をまとめて実行する場合のベンチマーク
	const simdbench::BatchedExecutionBenchmarkResult batched{
	    simdbench::runBatchedExecutionBenchmark(ELEMENT_COUNT, BATCHED_OUTPUT_COUNT,
	                                            REPEAT_COUNT)};

	// 乗算と加算を多めに含む重い式のベンチマーク
	const simdbench::HeavyExpressionBenchmarkResult heavy{
	    simdbench::runHeavyExpressionBenchmark(ELEMENT_COUNT, REPEAT_COUNT)};

	// 入力サイズと繰り返し回数を先に表示して、実行条件を結果と一緒に確認できるようにしています。
	std::cout << "要素数      : " << fused.elementCount << "\n";
	std::cout << "繰り返し回数: " << fused.repeatCount << "\n\n";

	std::cout << "[計算式: x/y/z = a * b + c/d/e の比較]\n";
	std::cout << "SIMDなしの通常ループ         : " << fused.scalarMs << " ms\n";
	std::cout << "手書きSIMD                   : " << fused.manualSimdMs << " ms\n";
	std::cout << "DirectXMath                  : " << fused.directXMathMs << " ms\n";
	std::cout << "式API                        : " << fused.normalExpressionMs << " ms\n";
	std::cout << "数値誤差(通常ループ と SIMD) : " << fused.scalarXError << "\n";
	std::cout << "数値誤差(通常ループ と DXMath): " << fused.directXMathXError << "\n";
	std::cout << "数値誤差(通常ループ と 式API): " << fused.expressionXError << "\n";
	std::cout << "\n";

	std::cout << "[位置と速度の更新処理の比較]\n";
	std::cout << "SIMDなしの通常ループ                  : " << motionUpdate.scalarMs << " ms\n";
	std::cout << "手書きSIMD                            : " << motionUpdate.manualSimdMs << " ms\n";
	std::cout << "DirectXMath                           : " << motionUpdate.directXMathMs << " ms\n";
	std::cout << "式API                                 : " << motionUpdate.specializedUpdateMs << " ms\n";
	std::cout << "数値誤差(通常ループ と SIMD)          : " << motionUpdate.scalarPositionXError << "\n";
	std::cout << "数値誤差(通常ループ と DXMath)        : " << motionUpdate.directXMathPositionXError << "\n";
	std::cout << "数値誤差(通常ループ と 式API)         : " << motionUpdate.specializedPositionXError << "\n";
	std::cout << "\n";

	std::cout << "[遅延実行でまとめる場合の比較]\n";
	std::cout << "出力配列数  : " << batched.outputCount << "\n";
	std::cout << "SIMDなしの通常ループ              : " << batched.scalarMs << " ms\n";
	std::cout << "手書きSIMD(通常実行)              : " << batched.manualSimdMs << " ms\n";
	std::cout << "DirectXMath(通常実行)             : " << batched.directXMathMs << " ms\n";
	std::cout << "式API(1つずつexecute)             : " << batched.sequentialExpressionMs << " ms\n";
	std::cout << "式API(まとめて1回execute)         : " << batched.batchedExpressionMs << " ms\n";
	std::cout << "数値誤差(通常ループ と SIMD)      : " << batched.manualSimdFirstOutputError << "\n";
	std::cout << "数値誤差(通常ループ と DXMath)    : " << batched.directXMathFirstOutputError << "\n";
	std::cout << "数値誤差(通常ループ と 1つずつ)    : " << batched.sequentialFirstOutputError << "\n";
	std::cout << "数値誤差(通常ループ と まとめ実行) : " << batched.batchedFirstOutputError << "\n";
	std::cout << "\n";

	std::cout << "[重い式の比較(今回は(a * b + c) * (d * e + f) + (a + d) * (b + e))]\n";
	std::cout << "SIMDなしの通常ループ         : " << heavy.scalarMs << " ms\n";
	std::cout << "手書きSIMD                   : " << heavy.manualSimdMs << " ms\n";
	std::cout << "DirectXMath                  : " << heavy.directXMathMs << " ms\n";
	std::cout << "式API                        : " << heavy.expressionMs << " ms\n";
	std::cout << "数値誤差(通常ループ と SIMD) : " << heavy.manualSimdError << "\n";
	std::cout << "数値誤差(通常ループ と DXMath): " << heavy.directXMathError << "\n";
	std::cout << "数値誤差(通常ループ と 式API): " << heavy.expressionError << "\n";

	return 0;
}
