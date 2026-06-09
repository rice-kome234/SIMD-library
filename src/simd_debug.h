#pragma once

#include "simd_internal.h"

#include <iostream>
#include <ostream>

namespace rice::simd::internal
{
struct DebugAccess {
	static const ScheduledPlanData &data(const ScheduledPlan &plan) noexcept
	{
		return *plan.impl_;
	}
};
}

namespace rice::simd::debug
{
/*!
 *   @brief コンパイル済みプランの診断情報を指定ストリームへ出力
 *   @param[in] plan 表示するコンパイル済みプラン
 *   @param[out] os 出力先ストリーム
 *   @details 開発やベンチマーク確認用の補助APIです。通常利用では不要です。
 */
inline void printStages(const ScheduledPlan &plan, std::ostream &os)
{
	const internal::ScheduledPlanData &data{internal::DebugAccess::data(plan)};

	os << "ScheduledPlan\n";
	os << "  stage count       : " << plan.stageCount() << "\n";
	os << "  instruction count : " << plan.instructionCount() << "\n";
	os << "  max register count: " << plan.maxRegisterCount() << "\n";

	for (std::size_t i{}; i < data.stages.size(); ++i) {
		os << "  Stage " << i << " | instructions: " << data.stages[i].instructionCount()
		   << " | registers: " << data.stages[i].registerCount() << "\n";
	}
}

/*!
 *   @brief コンパイル済みプランの診断情報を標準出力へ出力
 *   @param[in] plan 表示するコンパイル済みプラン
 */
inline void printStages(const ScheduledPlan &plan) { printStages(plan, std::cout); }
}
