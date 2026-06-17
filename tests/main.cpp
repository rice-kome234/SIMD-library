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
constexpr std::size_t ELEMENT_COUNT{static_cast<std::size_t>(1) << 24};
constexpr int REPEAT_COUNT{100};
constexpr float DELTA_TIME{0.016f};

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
	
	// ここを変更すると、すべてのベンチマーク条件をまとめて調整できます。
	// 通常式でのベンチマーク
	const simdbench::FusedExpressionBenchmarkResult fused{
	    simdbench::runFusedExpressionBenchmark(ELEMENT_COUNT, REPEAT_COUNT)};

	// 位置と速度をまとめて更新するベンチマーク
	const simdbench::ThreeComponentUpdateBenchmarkResult motionUpdate{
	    simdbench::runThreeComponentUpdateBenchmark(ELEMENT_COUNT, REPEAT_COUNT, DELTA_TIME)};

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
	std::cout << "手書きSIMD                            : " << motionUpdate.manualSimdMs<< " ms\n";
	std::cout << "DirectXMath                           : " << motionUpdate.directXMathMs<< " ms\n";
	std::cout << "式API                                 : " << motionUpdate.specializedUpdateMs<< " ms\n";
	std::cout << "数値誤差(通常ループ と SIMD)          : " << motionUpdate.scalarPositionXError<< "\n";
	std::cout << "数値誤差(通常ループ と DXMath)        : "<< motionUpdate.directXMathPositionXError << "\n";
	std::cout << "数値誤差(通常ループ と 式API)         : "<< motionUpdate.specializedPositionXError << "\n";

	return 0;
}
