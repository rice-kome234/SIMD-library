#pragma once

#include "simd_internal.h"

#include <iostream>
#include <ostream>

namespace rice::simd::internal
{
struct DebugAccess {
	static const ScheduledPlanData &data(const ScheduledPlan &plan) noexcept
	{
		return plan.data_;
	}
};
}

namespace rice::simd::debug
{
/*!
 *   @brief 内部実行手順の診断情報を指定ストリームへ出力
 *   @param[in] plan 表示する内部実行手順
 *   @param[out] os 出力先ストリーム
 *   @details 開発やベンチマーク確認用の補助APIです。通常利用では不要です。
 */
inline void printStages(const internal::ScheduledPlan &plan, std::ostream &os)
{
	const internal::ScheduledPlanData &data{internal::DebugAccess::data(plan)};

	os << "InternalPlan\n";
	os << "  stage count       : " << plan.stageCount() << "\n";
	os << "  instruction count : " << plan.instructionCount() << "\n";
	os << "  max register count: " << plan.maxRegisterCount() << "\n";

	for (std::size_t i{}; i < data.stages.size(); ++i) {
		os << "  Stage " << i << " | instructions: " << data.stages[i].instructionCount()
		   << " | registers: " << data.stages[i].registerCount() << "\n";
	}
}

/*!
 *   @brief 内部実行手順の診断情報を標準出力へ出力
 *   @param[in] plan 表示する内部実行手順
 */
inline void printStages(const internal::ScheduledPlan &plan) { printStages(plan, std::cout); }
}
