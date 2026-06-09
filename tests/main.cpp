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
	std::cout << "Element count: " << fused.elementCount << "\n";
	std::cout << "Repeat count : " << fused.repeatCount << "\n\n";

	std::cout << "[Fused expressions]\n";
	std::cout << "Scalar no SIMD    : " << fused.scalarMs << " ms\n";
	std::cout << "Manual fused SIMD : " << fused.manualSimdMs << " ms\n";
	std::cout << "Normal expression : " << fused.normalExpressionMs << " ms\n";
	std::cout << "Compile once      : " << fused.compileMs << " ms\n";
	std::cout << "ScheduledPlan     : " << fused.scheduledPlanMs << " ms\n";
	std::cout << "Stages            : " << fused.stageCount << "\n";
	std::cout << "Instructions      : " << fused.instructionCount << "\n";
	std::cout << "Max registers     : " << fused.maxRegisterCount << "\n";
	std::cout << "scalar x error    : " << fused.scalarXError << "\n";
	std::cout << "expression x error: " << fused.expressionXError << "\n";
	std::cout << "plan x error      : " << fused.planXError << "\n\n";

	std::cout << "[Three-component update specialized]\n";
	std::cout << "Scalar no SIMD             : " << componentUpdate.scalarMs << " ms\n";
	std::cout << "Manual 3-component SIMD    : " << componentUpdate.manualSimdMs << " ms\n";
	std::cout << "Specialized 3-component    : " << componentUpdate.specializedUpdateMs
	          << " ms\n";
	std::cout << "scalar position.x error     : " << componentUpdate.scalarPositionXError
	          << "\n";
	std::cout << "specialized position.x error: " << componentUpdate.specializedPositionXError
	          << "\n";

	return 0;
}
