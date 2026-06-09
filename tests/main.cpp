#include "benchmark.h"

#include <cstddef>
#include <iostream>

namespace
{
constexpr std::size_t ELEMENT_COUNT{static_cast<std::size_t>(1) << 24};
constexpr int REPEAT_COUNT{100};
constexpr float DELTA_TIME{0.016f};
}

int main()
{
	// ここを変更すると、すべてのベンチマーク条件をまとめて調整できます。
	// a * b を共有する3つの式で、スカラー実装、手書きSIMD、通常式API、
	// コンパイル済みプランの実行時間と誤差を比較します。
	const simdbench::FusedExpressionBenchmarkResult fused{
	    simdbench::runFusedExpressionBenchmark(ELEMENT_COUNT, REPEAT_COUNT)};

	// 3成分データの位置・速度更新で、汎用的な式合成ではなく専用更新コードにした場合の
	// 効果を測ります。
	const simdbench::ThreeComponentUpdateBenchmarkResult componentUpdate{
	    simdbench::runThreeComponentUpdateBenchmark(ELEMENT_COUNT, REPEAT_COUNT, DELTA_TIME)};

	// 入力サイズと繰り返し回数を先に表示して、実行条件を結果と一緒に確認できるようにします。
	std::cout << "要素数      : " << fused.elementCount << "\n";
	std::cout << "繰り返し回数: " << fused.repeatCount << "\n\n";

	std::cout << "[共有部分式 a * b + c/d/e の比較]\n";
	std::cout << "SIMDなしの通常ループ      : " << fused.scalarMs << " ms\n";
	std::cout << "手書きSIMD                 : " << fused.manualSimdMs << " ms\n";
	std::cout << "式APIを毎回コンパイル      : " << fused.normalExpressionMs << " ms\n";
	std::cout << "プラン作成のみ             : " << fused.compileMs << " ms\n";
	std::cout << "コンパイル済みプランの実行 : " << fused.scheduledPlanMs << " ms\n";
	std::cout << "数値誤差(SIMDなし)         : " << fused.scalarXError << "\n";
	std::cout << "数値誤差(式API)            : " << fused.expressionXError << "\n";
	std::cout << "数値誤差(コンパイル済み)   : " << fused.planXError << "\n\n";

	std::cout << "[3成分の位置・速度更新の比較]\n";
	std::cout << "SIMDなしの通常ループ       : " << componentUpdate.scalarMs << " ms\n";
	std::cout << "手書きSIMD                 : " << componentUpdate.manualSimdMs << " ms\n";
	std::cout << "専用更新コード             : " << componentUpdate.specializedUpdateMs
	          << " ms\n";
	std::cout << "数値誤差(SIMDなし)         : " << componentUpdate.scalarPositionXError
	          << "\n";
	std::cout << "数値誤差(専用更新コード)   : " << componentUpdate.specializedPositionXError
	          << "\n";

	return 0;
}
