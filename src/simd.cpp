#include "simd_internal.h"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <stdexcept>
#include <utility>

namespace rice::simd
{
using internal::Assignment;
using internal::CompileContext;
using internal::CompileGroup;
using internal::ExprNode;
using internal::FloatArray;
using internal::FusionPlan;
using internal::KeyIndex;
using internal::MAX_REGISTERS;
using internal::NodeKind;
using internal::SIMD_WIDTH;
using internal::ValueKind;
using internal::ValueRef;
using internal::Variable;

internal::FloatArray::FloatArray(std::size_t elementCount) { resize(elementCount); }

void internal::FloatArray::resize(std::size_t elementCount)
{
	elementCount_ = elementCount;
	blocks_.assign((elementCount + SIMD_WIDTH - 1) / SIMD_WIDTH, _mm256_setzero_ps());
}

std::size_t internal::FloatArray::elementCount() const noexcept { return elementCount_; }

std::size_t internal::FloatArray::blockCount() const noexcept { return blocks_.size(); }

__m256 &internal::FloatArray::block(std::size_t index) noexcept { return blocks_[index]; }

const __m256 &internal::FloatArray::block(std::size_t index) const noexcept
{
	return blocks_[index];
}

__m256 *internal::FloatArray::data() noexcept { return blocks_.data(); }

const __m256 *internal::FloatArray::data() const noexcept { return blocks_.data(); }

void internal::FloatArray::fill(float value) noexcept
{
	const __m256 block{_mm256_set1_ps(value)};

	for (__m256 &stored : blocks_) {
		stored = block;
	}
}

void internal::FloatArray::assign(std::size_t elementCount, float value)
{
	resize(elementCount);
	fill(value);
}

void internal::FloatArray::copyFrom(const std::vector<float> &values)
{
	copyFrom(values.data(), values.size());
}

void internal::FloatArray::copyFrom(std::initializer_list<float> values)
{
	copyFrom(values.begin(), values.size());
}

void internal::FloatArray::push_back(float value)
{
	const std::size_t index{elementCount_};
	if (index % SIMD_WIDTH == 0) {
		blocks_.push_back(_mm256_setzero_ps());
	}

	++elementCount_;
	setElement(index, value);
}

void internal::FloatArray::copyTo(std::vector<float> &out) const
{
	out.resize(elementCount_);
	if (elementCount_ == 0) {
		return;
	}

	const std::size_t fullBlocks{elementCount_ / SIMD_WIDTH};
	for (std::size_t blockIndex{}; blockIndex < fullBlocks; ++blockIndex) {
		_mm256_storeu_ps(out.data() + blockIndex * SIMD_WIDTH, blocks_[blockIndex]);
	}

	const std::size_t tailCount{elementCount_ % SIMD_WIDTH};
	if (tailCount == 0) {
		return;
	}

	alignas(32) float temp[SIMD_WIDTH]{};
	_mm256_store_ps(temp, blocks_[fullBlocks]);

	const std::size_t offset{fullBlocks * SIMD_WIDTH};
	for (std::size_t lane{}; lane < tailCount; ++lane) {
		out[offset + lane] = temp[lane];
	}
}

void internal::FloatArray::copyFrom(const float *values, std::size_t count)
{
	resize(count);

	const std::size_t fullBlocks{count / SIMD_WIDTH};
	for (std::size_t blockIndex{}; blockIndex < fullBlocks; ++blockIndex) {
		blocks_[blockIndex] = _mm256_loadu_ps(values + blockIndex * SIMD_WIDTH);
	}

	const std::size_t tailCount{count % SIMD_WIDTH};
	if (tailCount != 0) {
		alignas(32) float temp[SIMD_WIDTH]{};
		const std::size_t offset{fullBlocks * SIMD_WIDTH};

		for (std::size_t lane{}; lane < tailCount; ++lane) {
			temp[lane] = values[offset + lane];
		}

		blocks_[fullBlocks] = _mm256_load_ps(temp);
	}
}

void internal::FloatArray::setElement(std::size_t index, float value) noexcept
{
	alignas(32) float temp[SIMD_WIDTH]{};
	const std::size_t blockIndex{index / SIMD_WIDTH};
	const std::size_t lane{index % SIMD_WIDTH};

	_mm256_store_ps(temp, blocks_[blockIndex]);
	temp[lane] = value;
	blocks_[blockIndex] = _mm256_load_ps(temp);
}

internal::Variable::Variable(Engine *engine, FloatArray *array) noexcept
    : engine_{engine}, array_{array}
{
}

FloatArray &internal::Variable::array() noexcept
{
	assert(array_ != nullptr);
	return *array_;
}

const FloatArray &internal::Variable::array() const noexcept
{
	assert(array_ != nullptr);
	return *array_;
}

Engine *internal::Variable::engine() const noexcept { return engine_; }

internal::ArrayData::ArrayData(Engine &engine, std::size_t elementCount)
    : storage{elementCount}, variable{&engine, &storage}
{
}

Array::Array(Engine &engine, std::size_t elementCount)
    : impl_{std::make_unique<internal::ArrayData>(engine, elementCount)}
{
}

Array::~Array() noexcept = default;
Array::Array(Array &&) noexcept = default;
Array &Array::operator=(Array &&) noexcept = default;

Array &Array::operator=(const Array &value)
{
	assert(impl_ != nullptr);
	return operator=(impl_->variable.engine()->makeVariable(value));
}

Array &Array::operator=(const Expression &expr)
{
	assert(impl_ != nullptr);
	impl_->variable.engine()->deferAssign(*this, expr);
	return *this;
}

Array &Array::operator=(const std::vector<float> &values)
{
	copyFrom(values);
	return *this;
}

Array &Array::operator=(std::initializer_list<float> values)
{
	copyFrom(values);
	return *this;
}

void Array::resize(std::size_t elementCount)
{
	assert(impl_ != nullptr);
	impl_->storage.resize(elementCount);
}

void Array::resizeLike(const Array &source)
{
	assert(source.impl_ != nullptr);
	resize(source.elementCount());
}

void Array::fill(float value) noexcept
{
	assert(impl_ != nullptr);
	impl_->storage.fill(value);
}

void Array::assign(std::size_t elementCount, float value)
{
	assert(impl_ != nullptr);
	impl_->storage.assign(elementCount, value);
}

void Array::copyFrom(const Array &source)
{
	assert(impl_ != nullptr);
	assert(source.impl_ != nullptr);
	impl_->storage = source.impl_->storage;
}

void Array::copyFrom(const std::vector<float> &values)
{
	assert(impl_ != nullptr);
	impl_->storage.copyFrom(values);
}

void Array::copyFrom(std::initializer_list<float> values)
{
	assert(impl_ != nullptr);
	impl_->storage.copyFrom(values);
}

void Array::push_back(float value)
{
	assert(impl_ != nullptr);
	impl_->storage.push_back(value);
}

std::size_t Array::elementCount() const noexcept
{
	assert(impl_ != nullptr);
	return impl_->storage.elementCount();
}

std::size_t Array::size() const noexcept { return elementCount(); }

void Array::copyTo(std::vector<float> &out) const
{
	assert(impl_ != nullptr);
	impl_->storage.copyTo(out);
}

std::vector<float> Array::toVector() const
{
	std::vector<float> result{};
	copyTo(result);
	return result;
}

Engine *Array::engine() const noexcept
{
	assert(impl_ != nullptr);
	return impl_->variable.engine();
}

Expression::Expression(Engine *engine, std::size_t nodeId) noexcept
    : engine_{engine}, nodeId_{nodeId}
{
}

Engine *Expression::engine() const noexcept { return engine_; }

std::size_t Expression::nodeId() const noexcept { return nodeId_; }

namespace
{
void executeAddRR(const internal::FusionPlanData &plan, std::size_t, __m256 *regs,
                  std::size_t index) noexcept
{
	const internal::AddRR &op{plan.addRR[index]};
	regs[op.dst] = _mm256_add_ps(regs[op.a], regs[op.b]);
}

void executeAddRA(const internal::FusionPlanData &plan, std::size_t blockIndex, __m256 *regs,
                  std::size_t index) noexcept
{
	const internal::AddRA &op{plan.addRA[index]};
	regs[op.dst] = _mm256_add_ps(regs[op.a], op.b[blockIndex]);
}

void executeAddAR(const internal::FusionPlanData &plan, std::size_t blockIndex, __m256 *regs,
                  std::size_t index) noexcept
{
	const internal::AddAR &op{plan.addAR[index]};
	regs[op.dst] = _mm256_add_ps(op.a[blockIndex], regs[op.b]);
}

void executeAddAA(const internal::FusionPlanData &plan, std::size_t blockIndex, __m256 *regs,
                  std::size_t index) noexcept
{
	const internal::AddAA &op{plan.addAA[index]};
	regs[op.dst] = _mm256_add_ps(op.a[blockIndex], op.b[blockIndex]);
}

void executeAddRS(const internal::FusionPlanData &plan, std::size_t, __m256 *regs,
                  std::size_t index) noexcept
{
	const internal::AddRS &op{plan.addRS[index]};
	regs[op.dst] = _mm256_add_ps(regs[op.a], op.b);
}

void executeAddAS(const internal::FusionPlanData &plan, std::size_t blockIndex, __m256 *regs,
                  std::size_t index) noexcept
{
	const internal::AddAS &op{plan.addAS[index]};
	regs[op.dst] = _mm256_add_ps(op.a[blockIndex], op.b);
}

void executeSubRR(const internal::FusionPlanData &plan, std::size_t, __m256 *regs,
                  std::size_t index) noexcept
{
	const internal::SubRR &op{plan.subRR[index]};
	regs[op.dst] = _mm256_sub_ps(regs[op.a], regs[op.b]);
}

void executeSubRA(const internal::FusionPlanData &plan, std::size_t blockIndex, __m256 *regs,
                  std::size_t index) noexcept
{
	const internal::SubRA &op{plan.subRA[index]};
	regs[op.dst] = _mm256_sub_ps(regs[op.a], op.b[blockIndex]);
}

void executeSubAR(const internal::FusionPlanData &plan, std::size_t blockIndex, __m256 *regs,
                  std::size_t index) noexcept
{
	const internal::SubAR &op{plan.subAR[index]};
	regs[op.dst] = _mm256_sub_ps(op.a[blockIndex], regs[op.b]);
}

void executeSubAA(const internal::FusionPlanData &plan, std::size_t blockIndex, __m256 *regs,
                  std::size_t index) noexcept
{
	const internal::SubAA &op{plan.subAA[index]};
	regs[op.dst] = _mm256_sub_ps(op.a[blockIndex], op.b[blockIndex]);
}

void executeSubRS(const internal::FusionPlanData &plan, std::size_t, __m256 *regs,
                  std::size_t index) noexcept
{
	const internal::SubRS &op{plan.subRS[index]};
	regs[op.dst] = _mm256_sub_ps(regs[op.a], op.b);
}

void executeSubAS(const internal::FusionPlanData &plan, std::size_t blockIndex, __m256 *regs,
                  std::size_t index) noexcept
{
	const internal::SubAS &op{plan.subAS[index]};
	regs[op.dst] = _mm256_sub_ps(op.a[blockIndex], op.b);
}

void executeSubSR(const internal::FusionPlanData &plan, std::size_t, __m256 *regs,
                  std::size_t index) noexcept
{
	const internal::SubSR &op{plan.subSR[index]};
	regs[op.dst] = _mm256_sub_ps(op.a, regs[op.b]);
}

void executeSubSA(const internal::FusionPlanData &plan, std::size_t blockIndex, __m256 *regs,
                  std::size_t index) noexcept
{
	const internal::SubSA &op{plan.subSA[index]};
	regs[op.dst] = _mm256_sub_ps(op.a, op.b[blockIndex]);
}

void executeMulRR(const internal::FusionPlanData &plan, std::size_t, __m256 *regs,
                  std::size_t index) noexcept
{
	const internal::MulRR &op{plan.mulRR[index]};
	regs[op.dst] = _mm256_mul_ps(regs[op.a], regs[op.b]);
}

void executeMulRA(const internal::FusionPlanData &plan, std::size_t blockIndex, __m256 *regs,
                  std::size_t index) noexcept
{
	const internal::MulRA &op{plan.mulRA[index]};
	regs[op.dst] = _mm256_mul_ps(regs[op.a], op.b[blockIndex]);
}

void executeMulAR(const internal::FusionPlanData &plan, std::size_t blockIndex, __m256 *regs,
                  std::size_t index) noexcept
{
	const internal::MulAR &op{plan.mulAR[index]};
	regs[op.dst] = _mm256_mul_ps(op.a[blockIndex], regs[op.b]);
}

void executeMulAA(const internal::FusionPlanData &plan, std::size_t blockIndex, __m256 *regs,
                  std::size_t index) noexcept
{
	const internal::MulAA &op{plan.mulAA[index]};
	regs[op.dst] = _mm256_mul_ps(op.a[blockIndex], op.b[blockIndex]);
}

void executeMulRS(const internal::FusionPlanData &plan, std::size_t, __m256 *regs,
                  std::size_t index) noexcept
{
	const internal::MulRS &op{plan.mulRS[index]};
	regs[op.dst] = _mm256_mul_ps(regs[op.a], op.b);
}

void executeMulAS(const internal::FusionPlanData &plan, std::size_t blockIndex, __m256 *regs,
                  std::size_t index) noexcept
{
	const internal::MulAS &op{plan.mulAS[index]};
	regs[op.dst] = _mm256_mul_ps(op.a[blockIndex], op.b);
}

void executeDivRR(const internal::FusionPlanData &plan, std::size_t, __m256 *regs,
                  std::size_t index) noexcept
{
	const internal::DivRR &op{plan.divRR[index]};
	regs[op.dst] = _mm256_div_ps(regs[op.a], regs[op.b]);
}

void executeDivRA(const internal::FusionPlanData &plan, std::size_t blockIndex, __m256 *regs,
                  std::size_t index) noexcept
{
	const internal::DivRA &op{plan.divRA[index]};
	regs[op.dst] = _mm256_div_ps(regs[op.a], op.b[blockIndex]);
}

void executeDivAR(const internal::FusionPlanData &plan, std::size_t blockIndex, __m256 *regs,
                  std::size_t index) noexcept
{
	const internal::DivAR &op{plan.divAR[index]};
	regs[op.dst] = _mm256_div_ps(op.a[blockIndex], regs[op.b]);
}

void executeDivAA(const internal::FusionPlanData &plan, std::size_t blockIndex, __m256 *regs,
                  std::size_t index) noexcept
{
	const internal::DivAA &op{plan.divAA[index]};
	regs[op.dst] = _mm256_div_ps(op.a[blockIndex], op.b[blockIndex]);
}

void executeDivRS(const internal::FusionPlanData &plan, std::size_t, __m256 *regs,
                  std::size_t index) noexcept
{
	const internal::DivRS &op{plan.divRS[index]};
	regs[op.dst] = _mm256_div_ps(regs[op.a], op.b);
}

void executeDivAS(const internal::FusionPlanData &plan, std::size_t blockIndex, __m256 *regs,
                  std::size_t index) noexcept
{
	const internal::DivAS &op{plan.divAS[index]};
	regs[op.dst] = _mm256_div_ps(op.a[blockIndex], op.b);
}

void executeDivSR(const internal::FusionPlanData &plan, std::size_t, __m256 *regs,
                  std::size_t index) noexcept
{
	const internal::DivSR &op{plan.divSR[index]};
	regs[op.dst] = _mm256_div_ps(op.a, regs[op.b]);
}

void executeDivSA(const internal::FusionPlanData &plan, std::size_t blockIndex, __m256 *regs,
                  std::size_t index) noexcept
{
	const internal::DivSA &op{plan.divSA[index]};
	regs[op.dst] = _mm256_div_ps(op.a, op.b[blockIndex]);
}

void executeFmaRRR(const internal::FusionPlanData &plan, std::size_t, __m256 *regs,
                   std::size_t index) noexcept
{
	const internal::FmaRRR &op{plan.fmaRRR[index]};
	regs[op.dst] = _mm256_fmadd_ps(regs[op.a], regs[op.b], regs[op.c]);
}

void executeFmaRRA(const internal::FusionPlanData &plan, std::size_t blockIndex, __m256 *regs,
                   std::size_t index) noexcept
{
	const internal::FmaRRA &op{plan.fmaRRA[index]};
	regs[op.dst] = _mm256_fmadd_ps(regs[op.a], regs[op.b], op.c[blockIndex]);
}

void executeFmaRAA(const internal::FusionPlanData &plan, std::size_t blockIndex, __m256 *regs,
                   std::size_t index) noexcept
{
	const internal::FmaRAA &op{plan.fmaRAA[index]};
	regs[op.dst] = _mm256_fmadd_ps(regs[op.a], op.b[blockIndex], op.c[blockIndex]);
}

void executeFmaAAA(const internal::FusionPlanData &plan, std::size_t blockIndex, __m256 *regs,
                   std::size_t index) noexcept
{
	const internal::FmaAAA &op{plan.fmaAAA[index]};
	regs[op.dst] = _mm256_fmadd_ps(op.a[blockIndex], op.b[blockIndex], op.c[blockIndex]);
}

void executeFmaASA(const internal::FusionPlanData &plan, std::size_t blockIndex, __m256 *regs,
                   std::size_t index) noexcept
{
	const internal::FmaASA &op{plan.fmaASA[index]};
	regs[op.dst] = _mm256_fmadd_ps(op.a[blockIndex], op.b, op.c[blockIndex]);
}

void executeFmaRAS(const internal::FusionPlanData &plan, std::size_t blockIndex, __m256 *regs,
                   std::size_t index) noexcept
{
	const internal::FmaRAS &op{plan.fmaRAS[index]};
	regs[op.dst] = _mm256_fmadd_ps(regs[op.a], op.b[blockIndex], op.c);
}

void executeFmaRSA(const internal::FusionPlanData &plan, std::size_t blockIndex, __m256 *regs,
                   std::size_t index) noexcept
{
	const internal::FmaRSA &op{plan.fmaRSA[index]};
	regs[op.dst] = _mm256_fmadd_ps(regs[op.a], op.b, op.c[blockIndex]);
}

void executeSetS(const internal::FusionPlanData &plan, std::size_t, __m256 *regs,
                 std::size_t index) noexcept
{
	const internal::SetS &op{plan.setS[index]};
	regs[op.dst] = op.a;
}

void executeStoreR(const internal::FusionPlanData &plan, std::size_t blockIndex, __m256 *regs,
                   std::size_t index) noexcept
{
	const internal::StoreR &op{plan.storeR[index]};
	op.out[blockIndex] = regs[op.a];
}

void executeStoreA(const internal::FusionPlanData &plan, std::size_t blockIndex, __m256 *,
                   std::size_t index) noexcept
{
	const internal::StoreA &op{plan.storeA[index]};
	op.out[blockIndex] = op.a[blockIndex];
}

void executeStoreS(const internal::FusionPlanData &plan, std::size_t blockIndex, __m256 *,
                   std::size_t index) noexcept
{
	const internal::StoreS &op{plan.storeS[index]};
	op.out[blockIndex] = op.a;
}
}

internal::FusionPlan::FusionPlan() : impl_{std::make_shared<internal::FusionPlanData>()} {}

internal::FusionPlan::~FusionPlan() noexcept = default;

void internal::FusionPlan::execute() const noexcept
{
	__m256 regs[MAX_REGISTERS];

	for (std::size_t blockIndex{}; blockIndex < impl_->blockCount; ++blockIndex) {
		executeBlock(blockIndex, regs);
	}
}

void internal::FusionPlan::executeBlock(std::size_t blockIndex, __m256 *regs) const noexcept
{
	for (const internal::OpRef &ref : impl_->ops) {
		assert(ref.execute != nullptr);
		ref.execute(*impl_, blockIndex, regs, ref.index);
	}
}

std::size_t internal::FusionPlan::instructionCount() const noexcept { return impl_->ops.size(); }

int internal::FusionPlan::registerCount() const noexcept { return impl_->maxRegisterCount; }

int internal::FusionPlan::allocateRegister() noexcept
{
	int result{-1};

	if (!impl_->freeRegisters.empty()) {
		result = impl_->freeRegisters.back();
		impl_->freeRegisters.pop_back();
	} else {
		result = impl_->nextRegister;
		++impl_->nextRegister;
		impl_->maxRegisterCount = std::max(impl_->maxRegisterCount, impl_->nextRegister);
	}

	assert(result < MAX_REGISTERS);
	return result;
}

void internal::FusionPlan::releaseRegister(int reg)
{
	if (reg >= 0) {
		impl_->freeRegisters.push_back(reg);
	}
}

ScheduledPlan::ScheduledPlan() : impl_{std::make_shared<internal::ScheduledPlanData>()} {}

ScheduledPlan::~ScheduledPlan() noexcept = default;

void ScheduledPlan::execute() const noexcept
{
	if (impl_->stages.empty())
		return;

	if (impl_->stages.size() == 1) {
		impl_->stages.front().execute();
		return;
	}

	__m256 regs[MAX_REGISTERS];

	for (std::size_t blockIndex{}; blockIndex < impl_->blockCount; ++blockIndex) {
		for (const FusionPlan &stage : impl_->stages) {
			stage.executeBlock(blockIndex, regs);
		}
	}
}

std::size_t ScheduledPlan::stageCount() const noexcept { return impl_->stages.size(); }

std::size_t ScheduledPlan::instructionCount() const noexcept
{
	std::size_t total{};

	for (const FusionPlan &stage : impl_->stages) {
		total += stage.instructionCount();
	}

	return total;
}

int ScheduledPlan::maxRegisterCount() const noexcept
{
	int result{};

	for (const FusionPlan &stage : impl_->stages) {
		result = std::max(result, stage.registerCount());
	}

	return result;
}

namespace internal
{
struct Compiler {
	static std::uint32_t floatBits(float value) noexcept
	{
		std::uint32_t bits{};
		std::memcpy(&bits, &value, sizeof(float));
		return bits;
	}

	static std::uint64_t mix(std::uint64_t x) noexcept
	{
		constexpr std::uint64_t firstMultiplier{0xff51afd7ed558ccd};
		constexpr std::uint64_t secondMultiplier{0xc4ceb9fe1a85ec53};

		x ^= x >> 33;
		x *= firstMultiplier;
		x ^= x >> 33;
		x *= secondMultiplier;
		x ^= x >> 33;
		return x;
	}

	static std::uint64_t combineHash(std::uint64_t a, std::uint64_t b, std::uint64_t c) noexcept
	{
		constexpr std::uint64_t goldenRatio{0x9e3779b97f4a7c15};

		return mix(a ^ (b + goldenRatio + (a << 6) + (a >> 2)) ^ c);
	}

	static void requireSameShape(const FloatArray &expected, const FloatArray &actual,
	                             const char *message)
	{
		if (expected.elementCount() != actual.elementCount()) {
			throw std::invalid_argument{message};
		}

		if (expected.blockCount() != actual.blockCount()) {
			throw std::invalid_argument{message};
		}
	}

	static void requireVarOwner(const Engine &engine, const Variable &value,
	                            const char *message)
	{
		if (value.engine() != &engine) {
			throw std::invalid_argument{message};
		}
	}

	static void requireExprOwner(const Engine &engine, const Expression &expr,
	                             const char *message)
	{
		if (expr.engine() != &engine) {
			throw std::invalid_argument{message};
		}

		if (expr.nodeId() >= engine.impl_->nodes.size()) {
			throw std::invalid_argument{message};
		}
	}

	static bool containsArray(const std::vector<const FloatArray *> &values,
	                          const FloatArray *value) noexcept
	{
		for (const FloatArray *stored : values) {
			if (stored == value) {
				return true;
			}
		}

		return false;
	}

	static void appendUniqueArray(std::vector<const FloatArray *> &values,
	                              const FloatArray *value)
	{
		if (!containsArray(values, value)) {
			values.push_back(value);
		}
	}

	static void reserveNodesFor(Engine &engine, std::size_t additionalCount)
	{
		const std::size_t required{engine.impl_->nodes.size() + additionalCount};
		if (required <= engine.impl_->nodes.capacity()) {
			return;
		}

		std::size_t newCapacity{engine.impl_->nodes.capacity()};
		if (newCapacity == 0) {
			newCapacity = 8;
		}

		while (newCapacity < required) {
			newCapacity *= 2;
		}

		if (newCapacity < engine.impl_->expressionReserveHint) {
			newCapacity = engine.impl_->expressionReserveHint;
		}

		engine.impl_->nodes.reserve(newCapacity);
	}

	static void rememberExpressionReserve(Engine &engine) noexcept
	{
		if (engine.impl_->expressionReserveHint < engine.impl_->nodes.size()) {
			engine.impl_->expressionReserveHint = engine.impl_->nodes.size();
		}
	}

	static bool isCommutative(NodeKind kind) noexcept
	{
		if (kind == NodeKind::Add) {
			return true;
		}

		if (kind == NodeKind::Mul) {
			return true;
		}

		return false;
	}

	static bool sameNode(const ExprNode &lhs, const ExprNode &rhs) noexcept
	{
		if (lhs.key != rhs.key) {
			return false;
		}

		if (lhs.kind != rhs.kind) {
			return false;
		}

		if (lhs.kind == NodeKind::Variable) {
			return lhs.array == rhs.array;
		}

		if (lhs.kind == NodeKind::Scalar) {
			return floatBits(lhs.scalar) == floatBits(rhs.scalar);
		}

		if (lhs.lhs == rhs.lhs && lhs.rhs == rhs.rhs) {
			return true;
		}

		if (isCommutative(lhs.kind) && lhs.lhs == rhs.rhs && lhs.rhs == rhs.lhs) {
			return true;
		}

		return false;
	}

	static Expression appendNode(Engine &engine, const ExprNode &node)
	{
		const std::vector<ExprNode> &nodes{engine.impl_->nodes};
		for (std::size_t index{nodes.size()}; index > 0; --index) {
			const std::size_t nodeId{index - 1};
			if (sameNode(nodes[nodeId], node)) {
				return Expression{&engine, nodeId};
			}
		}

		reserveNodesFor(engine, 1);
		const std::size_t nodeId{engine.impl_->nodes.size()};
		engine.impl_->nodes.push_back(node);
		return Expression{&engine, nodeId};
	}

	static Expression makeBinary(Engine &engine, NodeKind kind, const Expression &lhs,
	                             const Expression &rhs)
	{
		requireExprOwner(engine, lhs, "左辺式が別のEngineに紐づいているか、無効です。");
		requireExprOwner(engine, rhs, "右辺式が別のEngineに紐づいているか、無効です。");

		const ExprNode &lhsNode{engine.impl_->nodes[lhs.nodeId()]};
		const ExprNode &rhsNode{engine.impl_->nodes[rhs.nodeId()]};

		if (lhsNode.kind == NodeKind::Scalar && rhsNode.kind == NodeKind::Scalar) {
			float folded{};

			if (kind == NodeKind::Add) {
				folded = lhsNode.scalar + rhsNode.scalar;
			} else if (kind == NodeKind::Sub) {
				folded = lhsNode.scalar - rhsNode.scalar;
			} else if (kind == NodeKind::Mul) {
				folded = lhsNode.scalar * rhsNode.scalar;
			} else {
				folded = lhsNode.scalar / rhsNode.scalar;
			}

			return engine.makeScalar(folded);
		}

		std::uint64_t lhsKey{lhsNode.key};
		std::uint64_t rhsKey{rhsNode.key};

		if ((kind == NodeKind::Add || kind == NodeKind::Mul) && rhsKey < lhsKey) {
			std::swap(lhsKey, rhsKey);
		}

		ExprNode node{};
		node.kind = kind;
		node.lhs = lhs.nodeId();
		node.rhs = rhs.nodeId();
		if (kind == NodeKind::Add) {
			node.key = combineHash(3, lhsKey, rhsKey);
		} else if (kind == NodeKind::Sub) {
			node.key = combineHash(4, lhsKey, rhsKey);
		} else if (kind == NodeKind::Mul) {
			node.key = combineHash(5, lhsKey, rhsKey);
		} else {
			node.key = combineHash(6, lhsKey, rhsKey);
		}

		return appendNode(engine, node);
	}

	static void collectReads(const Engine &engine, std::size_t nodeId,
	                         std::vector<const FloatArray *> &reads)
	{
		const ExprNode &n{engine.impl_->nodes[nodeId]};

		if (n.kind == NodeKind::Variable) {
			appendUniqueArray(reads, n.array);
			return;
		}

		if (n.lhs != INVALID_NODE)
			collectReads(engine, n.lhs, reads);
		if (n.rhs != INVALID_NODE)
			collectReads(engine, n.rhs, reads);
	}

	static std::vector<std::vector<Assignment>>
	buildStages(const Engine &engine, const std::vector<Assignment> &assignments)
	{
		std::vector<std::vector<Assignment>> stages{};
		std::vector<Assignment> currentStage{};
		std::vector<const FloatArray *> currentWrites{};
		std::vector<const FloatArray *> reads{};
		reads.reserve(8);

		for (const Assignment &assignment : assignments) {
			reads.clear();
			collectReads(engine, assignment.expr_.nodeId(), reads);

			const FloatArray *write{&assignment.out_->variable.array()};
			bool newStage{containsArray(currentWrites, write)};

			for (const FloatArray *read : reads) {
				if (containsArray(currentWrites, read)) {
					newStage = true;
					break;
				}
			}

			if (newStage && !currentStage.empty()) {
				stages.push_back(currentStage);
				currentStage.clear();
				currentWrites.clear();
			}

			currentStage.push_back(assignment);
			appendUniqueArray(currentWrites, write);
		}

		if (!currentStage.empty()) {
			stages.push_back(currentStage);
		}

		return stages;
	}

	static void validateAssignments(const Engine &engine,
	                                const std::vector<Assignment> &assignments)
	{
		if (assignments.empty()) {
			return;
		}

		if (assignments.front().out_ == nullptr) {
			throw std::invalid_argument{"代入先が空です。"};
		}

		requireVarOwner(engine, assignments.front().out_->variable,
		                "代入先が別のEngineに紐づいています。");
		const FloatArray &expected{assignments.front().out_->variable.array()};
		std::vector<const FloatArray *> reads{};
		reads.reserve(8);

		for (const Assignment &assignment : assignments) {
			if (assignment.out_ == nullptr) {
				throw std::invalid_argument{"代入先が空です。"};
			}

			requireVarOwner(engine, assignment.out_->variable,
			                "代入先が別のEngineに紐づいています。");
			requireExprOwner(engine, assignment.expr_,
			                 "代入式が別のEngineに紐づいているか、無効です。");
			requireSameShape(expected, assignment.out_->variable.array(),
			                 "代入先配列の要素数が一致しません。");

			reads.clear();
			collectReads(engine, assignment.expr_.nodeId(), reads);

			for (const FloatArray *read : reads) {
				requireSameShape(expected, *read,
				                 "式に含まれる配列の要素数が一致しません。");
			}
		}
	}

	static CompileContext makeCompileContext(const Engine &engine)
	{
		CompileContext context{};
		const std::vector<ExprNode> &nodes{engine.impl_->nodes};

		context.groupByNode.assign(nodes.size(), INVALID_NODE);
		if (nodes.empty()) {
			return context;
		}

		std::vector<KeyIndex> keys{};
		keys.reserve(nodes.size());
		context.groups.reserve(nodes.size());

		for (std::size_t i{}; i < nodes.size(); ++i) {
			keys.push_back(KeyIndex{nodes[i].key, i});
		}

		std::sort(keys.begin(), keys.end(), [](const KeyIndex &lhs, const KeyIndex &rhs) {
			if (lhs.key != rhs.key) {
				return lhs.key < rhs.key;
			}

			return lhs.node < rhs.node;
		});

		for (const KeyIndex &entry : keys) {
			if (context.groups.empty() || context.groups.back().key != entry.key ||
			    !sameNode(nodes[context.groups.back().node], nodes[entry.node])) {
				CompileGroup group{};
				group.key = entry.key;
				group.node = entry.node;
				context.groups.push_back(group);
			}

			context.groupByNode[entry.node] = context.groups.size() - 1;
		}

		return context;
	}

	static void countUses(const Engine &engine, std::size_t nodeId,
	                      CompileContext &context) noexcept
	{
		const ExprNode &n{engine.impl_->nodes[nodeId]};
		++context.groups[context.groupByNode[nodeId]].useCount;

		if (n.lhs != INVALID_NODE)
			countUses(engine, n.lhs, context);
		if (n.rhs != INVALID_NODE)
			countUses(engine, n.rhs, context);
	}

	static ValueRef makeLeafValue(const Engine &engine, std::size_t nodeId,
	                              const CompileContext &context) noexcept
	{
		const ExprNode &n{engine.impl_->nodes[nodeId]};
		const std::size_t group{context.groupByNode[nodeId]};

		if (n.kind == NodeKind::Variable) {
			ValueRef value{};
			value.kind = ValueKind::Array;
			value.array = n.array->data();
			value.group = group;
			value.hasGroup = true;
			return value;
		}

		ValueRef value{};
		value.kind = ValueKind::Scalar;
		value.scalar = _mm256_set1_ps(n.scalar);
		value.group = group;
		value.hasGroup = true;
		return value;
	}

	static void releaseIfLastUse(FusionPlan &plan, const ValueRef &value,
	                             CompileContext &context)
	{
		if (!value.hasGroup)
			return;

		if (value.group >= context.groups.size())
			return;

		CompileGroup &group{context.groups[value.group]};
		--group.useCount;

		if (group.useCount <= 0) {
			group.hasCachedValue = false;

			if (value.kind == ValueKind::Reg) {
				plan.releaseRegister(value.reg);
			}
		}
	}

	static bool canUseFma(const Engine &engine, std::size_t nodeId,
	                      const CompileContext &context) noexcept
	{
		const ExprNode &n{engine.impl_->nodes[nodeId]};

		if (n.kind != NodeKind::Add)
			return false;

		if (engine.impl_->nodes[n.lhs].kind == NodeKind::Mul) {
			const std::size_t group{context.groupByNode[n.lhs]};
			return group < context.groups.size() && context.groups[group].useCount <= 1;
		}

		if (engine.impl_->nodes[n.rhs].kind == NodeKind::Mul) {
			const std::size_t group{context.groupByNode[n.rhs]};
			return group < context.groups.size() && context.groups[group].useCount <= 1;
		}

		return false;
	}

	static ValueRef makeRegValue(int reg, std::size_t group) noexcept
	{
		ValueRef value{};
		value.kind = ValueKind::Reg;
		value.reg = reg;
		value.group = group;
		value.hasGroup = true;
		return value;
	}

	static ValueRef compileNode(const Engine &engine, FusionPlan &plan, std::size_t nodeId,
	                            CompileContext &context)
	{
		const ExprNode &n{engine.impl_->nodes[nodeId]};
		const std::size_t groupId{context.groupByNode[nodeId]};

		if (n.kind == NodeKind::Variable || n.kind == NodeKind::Scalar) {
			return makeLeafValue(engine, nodeId, context);
		}

		CompileGroup &group{context.groups[groupId]};
		if (group.hasCachedValue) {
			return group.cachedValue;
		}

		ValueRef result{};

		if (canUseFma(engine, nodeId, context)) {
			result = compileFma(engine, plan, nodeId, context);
		} else if (n.kind == NodeKind::Add) {
			const ValueRef lhs{compileNode(engine, plan, n.lhs, context)};
			const ValueRef rhs{compileNode(engine, plan, n.rhs, context)};

			const int dst{plan.allocateRegister()};
			emitAdd(plan, dst, lhs, rhs);

			releaseIfLastUse(plan, lhs, context);
			releaseIfLastUse(plan, rhs, context);

			result = makeRegValue(dst, groupId);
		} else if (n.kind == NodeKind::Sub) {
			const ValueRef lhs{compileNode(engine, plan, n.lhs, context)};
			const ValueRef rhs{compileNode(engine, plan, n.rhs, context)};

			const int dst{plan.allocateRegister()};
			emitSub(plan, dst, lhs, rhs);

			releaseIfLastUse(plan, lhs, context);
			releaseIfLastUse(plan, rhs, context);

			result = makeRegValue(dst, groupId);
		} else if (n.kind == NodeKind::Mul) {
			const ValueRef lhs{compileNode(engine, plan, n.lhs, context)};
			const ValueRef rhs{compileNode(engine, plan, n.rhs, context)};

			const int dst{plan.allocateRegister()};
			emitMul(plan, dst, lhs, rhs);

			releaseIfLastUse(plan, lhs, context);
			releaseIfLastUse(plan, rhs, context);

			result = makeRegValue(dst, groupId);
		} else {
			const ValueRef lhs{compileNode(engine, plan, n.lhs, context)};
			const ValueRef rhs{compileNode(engine, plan, n.rhs, context)};

			const int dst{plan.allocateRegister()};
			emitDiv(plan, dst, lhs, rhs);

			releaseIfLastUse(plan, lhs, context);
			releaseIfLastUse(plan, rhs, context);

			result = makeRegValue(dst, groupId);
		}

		group.cachedValue = result;
		group.hasCachedValue = true;
		return result;
	}

	static ValueRef compileFma(const Engine &engine, FusionPlan &plan, std::size_t nodeId,
	                           CompileContext &context)
	{
		const ExprNode &n{engine.impl_->nodes[nodeId]};

		std::size_t mulId{INVALID_NODE};
		std::size_t addId{INVALID_NODE};

		if (engine.impl_->nodes[n.lhs].kind == NodeKind::Mul) {
			mulId = n.lhs;
			addId = n.rhs;
		} else {
			mulId = n.rhs;
			addId = n.lhs;
		}

		const ExprNode &mul{engine.impl_->nodes[mulId]};

		const ValueRef a{compileNode(engine, plan, mul.lhs, context)};
		const ValueRef b{compileNode(engine, plan, mul.rhs, context)};
		const ValueRef c{compileNode(engine, plan, addId, context)};

		const int dst{plan.allocateRegister()};
		emitFmaOrFallback(plan, dst, a, b, c);

		releaseIfLastUse(plan, a, context);
		releaseIfLastUse(plan, b, context);
		releaseIfLastUse(plan, c, context);

		return makeRegValue(dst, context.groupByNode[nodeId]);
	}

	template <typename Op>
	static void appendOp(FusionPlan &plan, std::vector<Op> &bucket,
	                     internal::OpExecutor execute, const Op &op)
	{
		const std::size_t index{bucket.size()};
		bucket.push_back(op);
		plan.impl_->ops.push_back(internal::OpRef{execute, index});
	}

	static void emitStore(FusionPlan &plan, __m256 *out, const ValueRef &value,
	                      CompileContext &context)
	{
		if (value.kind == ValueKind::Reg) {
			appendOp(plan, plan.impl_->storeR, executeStoreR,
			         internal::StoreR{out, value.reg});
		} else if (value.kind == ValueKind::Array) {
			appendOp(plan, plan.impl_->storeA, executeStoreA,
			         internal::StoreA{out, value.array});
		} else {
			appendOp(plan, plan.impl_->storeS, executeStoreS,
			         internal::StoreS{out, value.scalar});
		}

		releaseIfLastUse(plan, value, context);
	}

	static void emitAdd(FusionPlan &plan, int dst, const ValueRef &a, const ValueRef &b)
	{
		if (a.kind == ValueKind::Reg && b.kind == ValueKind::Reg)
			appendOp(plan, plan.impl_->addRR, executeAddRR,
			         internal::AddRR{dst, a.reg, b.reg});
		else if (a.kind == ValueKind::Reg && b.kind == ValueKind::Array)
			appendOp(plan, plan.impl_->addRA, executeAddRA,
			         internal::AddRA{dst, a.reg, b.array});
		else if (a.kind == ValueKind::Array && b.kind == ValueKind::Reg)
			appendOp(plan, plan.impl_->addAR, executeAddAR,
			         internal::AddAR{dst, a.array, b.reg});
		else if (a.kind == ValueKind::Array && b.kind == ValueKind::Array)
			appendOp(plan, plan.impl_->addAA, executeAddAA,
			         internal::AddAA{dst, a.array, b.array});
		else if (a.kind == ValueKind::Reg && b.kind == ValueKind::Scalar)
			appendOp(plan, plan.impl_->addRS, executeAddRS,
			         internal::AddRS{dst, a.reg, b.scalar});
		else if (a.kind == ValueKind::Array && b.kind == ValueKind::Scalar)
			appendOp(plan, plan.impl_->addAS, executeAddAS,
			         internal::AddAS{dst, a.array, b.scalar});
		else if (a.kind == ValueKind::Scalar && b.kind == ValueKind::Reg)
			appendOp(plan, plan.impl_->addRS, executeAddRS,
			         internal::AddRS{dst, b.reg, a.scalar});
		else if (a.kind == ValueKind::Scalar && b.kind == ValueKind::Array)
			appendOp(plan, plan.impl_->addAS, executeAddAS,
			         internal::AddAS{dst, b.array, a.scalar});
		else
			appendOp(plan, plan.impl_->setS, executeSetS,
			         internal::SetS{dst, _mm256_add_ps(a.scalar, b.scalar)});
	}

	static void emitSub(FusionPlan &plan, int dst, const ValueRef &a, const ValueRef &b)
	{
		if (a.kind == ValueKind::Reg && b.kind == ValueKind::Reg)
			appendOp(plan, plan.impl_->subRR, executeSubRR,
			         internal::SubRR{dst, a.reg, b.reg});
		else if (a.kind == ValueKind::Reg && b.kind == ValueKind::Array)
			appendOp(plan, plan.impl_->subRA, executeSubRA,
			         internal::SubRA{dst, a.reg, b.array});
		else if (a.kind == ValueKind::Array && b.kind == ValueKind::Reg)
			appendOp(plan, plan.impl_->subAR, executeSubAR,
			         internal::SubAR{dst, a.array, b.reg});
		else if (a.kind == ValueKind::Array && b.kind == ValueKind::Array)
			appendOp(plan, plan.impl_->subAA, executeSubAA,
			         internal::SubAA{dst, a.array, b.array});
		else if (a.kind == ValueKind::Reg && b.kind == ValueKind::Scalar)
			appendOp(plan, plan.impl_->subRS, executeSubRS,
			         internal::SubRS{dst, a.reg, b.scalar});
		else if (a.kind == ValueKind::Array && b.kind == ValueKind::Scalar)
			appendOp(plan, plan.impl_->subAS, executeSubAS,
			         internal::SubAS{dst, a.array, b.scalar});
		else if (a.kind == ValueKind::Scalar && b.kind == ValueKind::Reg)
			appendOp(plan, plan.impl_->subSR, executeSubSR,
			         internal::SubSR{dst, a.scalar, b.reg});
		else if (a.kind == ValueKind::Scalar && b.kind == ValueKind::Array)
			appendOp(plan, plan.impl_->subSA, executeSubSA,
			         internal::SubSA{dst, a.scalar, b.array});
		else
			appendOp(plan, plan.impl_->setS, executeSetS,
			         internal::SetS{dst, _mm256_sub_ps(a.scalar, b.scalar)});
	}

	static void emitMul(FusionPlan &plan, int dst, const ValueRef &a, const ValueRef &b)
	{
		if (a.kind == ValueKind::Reg && b.kind == ValueKind::Reg)
			appendOp(plan, plan.impl_->mulRR, executeMulRR,
			         internal::MulRR{dst, a.reg, b.reg});
		else if (a.kind == ValueKind::Reg && b.kind == ValueKind::Array)
			appendOp(plan, plan.impl_->mulRA, executeMulRA,
			         internal::MulRA{dst, a.reg, b.array});
		else if (a.kind == ValueKind::Array && b.kind == ValueKind::Reg)
			appendOp(plan, plan.impl_->mulAR, executeMulAR,
			         internal::MulAR{dst, a.array, b.reg});
		else if (a.kind == ValueKind::Array && b.kind == ValueKind::Array)
			appendOp(plan, plan.impl_->mulAA, executeMulAA,
			         internal::MulAA{dst, a.array, b.array});
		else if (a.kind == ValueKind::Reg && b.kind == ValueKind::Scalar)
			appendOp(plan, plan.impl_->mulRS, executeMulRS,
			         internal::MulRS{dst, a.reg, b.scalar});
		else if (a.kind == ValueKind::Array && b.kind == ValueKind::Scalar)
			appendOp(plan, plan.impl_->mulAS, executeMulAS,
			         internal::MulAS{dst, a.array, b.scalar});
		else if (a.kind == ValueKind::Scalar && b.kind == ValueKind::Reg)
			appendOp(plan, plan.impl_->mulRS, executeMulRS,
			         internal::MulRS{dst, b.reg, a.scalar});
		else if (a.kind == ValueKind::Scalar && b.kind == ValueKind::Array)
			appendOp(plan, plan.impl_->mulAS, executeMulAS,
			         internal::MulAS{dst, b.array, a.scalar});
		else
			appendOp(plan, plan.impl_->setS, executeSetS,
			         internal::SetS{dst, _mm256_mul_ps(a.scalar, b.scalar)});
	}

	static void emitDiv(FusionPlan &plan, int dst, const ValueRef &a, const ValueRef &b)
	{
		if (a.kind == ValueKind::Reg && b.kind == ValueKind::Reg)
			appendOp(plan, plan.impl_->divRR, executeDivRR,
			         internal::DivRR{dst, a.reg, b.reg});
		else if (a.kind == ValueKind::Reg && b.kind == ValueKind::Array)
			appendOp(plan, plan.impl_->divRA, executeDivRA,
			         internal::DivRA{dst, a.reg, b.array});
		else if (a.kind == ValueKind::Array && b.kind == ValueKind::Reg)
			appendOp(plan, plan.impl_->divAR, executeDivAR,
			         internal::DivAR{dst, a.array, b.reg});
		else if (a.kind == ValueKind::Array && b.kind == ValueKind::Array)
			appendOp(plan, plan.impl_->divAA, executeDivAA,
			         internal::DivAA{dst, a.array, b.array});
		else if (a.kind == ValueKind::Reg && b.kind == ValueKind::Scalar)
			appendOp(plan, plan.impl_->divRS, executeDivRS,
			         internal::DivRS{dst, a.reg, b.scalar});
		else if (a.kind == ValueKind::Array && b.kind == ValueKind::Scalar)
			appendOp(plan, plan.impl_->divAS, executeDivAS,
			         internal::DivAS{dst, a.array, b.scalar});
		else if (a.kind == ValueKind::Scalar && b.kind == ValueKind::Reg)
			appendOp(plan, plan.impl_->divSR, executeDivSR,
			         internal::DivSR{dst, a.scalar, b.reg});
		else if (a.kind == ValueKind::Scalar && b.kind == ValueKind::Array)
			appendOp(plan, plan.impl_->divSA, executeDivSA,
			         internal::DivSA{dst, a.scalar, b.array});
		else
			appendOp(plan, plan.impl_->setS, executeSetS,
			         internal::SetS{dst, _mm256_div_ps(a.scalar, b.scalar)});
	}

	static void emitFmaOrFallback(FusionPlan &plan, int dst, const ValueRef &a,
	                              const ValueRef &b, const ValueRef &c)
	{
		if (a.kind == ValueKind::Reg && b.kind == ValueKind::Reg &&
		    c.kind == ValueKind::Reg)
			appendOp(plan, plan.impl_->fmaRRR, executeFmaRRR,
			         internal::FmaRRR{dst, a.reg, b.reg, c.reg});
		else if (a.kind == ValueKind::Reg && b.kind == ValueKind::Reg &&
		         c.kind == ValueKind::Array)
			appendOp(plan, plan.impl_->fmaRRA, executeFmaRRA,
			         internal::FmaRRA{dst, a.reg, b.reg, c.array});
		else if (a.kind == ValueKind::Reg && b.kind == ValueKind::Array &&
		         c.kind == ValueKind::Array)
			appendOp(plan, plan.impl_->fmaRAA, executeFmaRAA,
			         internal::FmaRAA{dst, a.reg, b.array, c.array});
		else if (a.kind == ValueKind::Array && b.kind == ValueKind::Array &&
		         c.kind == ValueKind::Array)
			appendOp(plan, plan.impl_->fmaAAA, executeFmaAAA,
			         internal::FmaAAA{dst, a.array, b.array, c.array});
		else if (a.kind == ValueKind::Array && b.kind == ValueKind::Scalar &&
		         c.kind == ValueKind::Array)
			appendOp(plan, plan.impl_->fmaASA, executeFmaASA,
			         internal::FmaASA{dst, a.array, b.scalar, c.array});
		else if (a.kind == ValueKind::Scalar && b.kind == ValueKind::Array &&
		         c.kind == ValueKind::Array)
			appendOp(plan, plan.impl_->fmaASA, executeFmaASA,
			         internal::FmaASA{dst, b.array, a.scalar, c.array});
		else if (a.kind == ValueKind::Reg && b.kind == ValueKind::Array &&
		         c.kind == ValueKind::Scalar)
			appendOp(plan, plan.impl_->fmaRAS, executeFmaRAS,
			         internal::FmaRAS{dst, a.reg, b.array, c.scalar});
		else if (a.kind == ValueKind::Reg && b.kind == ValueKind::Scalar &&
		         c.kind == ValueKind::Array)
			appendOp(plan, plan.impl_->fmaRSA, executeFmaRSA,
			         internal::FmaRSA{dst, a.reg, b.scalar, c.array});
		else {
			const int temp{plan.allocateRegister()};
			emitMul(plan, temp, a, b);
			ValueRef tempValue{};
			tempValue.kind = ValueKind::Reg;
			tempValue.reg = temp;
			emitAdd(plan, dst, tempValue, c);
			plan.releaseRegister(temp);
		}
	}

	static FusionPlan compileFusion(const Engine &engine,
	                                const std::vector<Assignment> &assignments)
	{
		assert(!assignments.empty());

		FusionPlan plan{};
		plan.impl_->blockCount = assignments.front().out_->variable.array().blockCount();
		plan.impl_->ops.reserve(assignments.size() * 8);

		CompileContext context{makeCompileContext(engine)};

		for (const Assignment &assignment : assignments) {
			countUses(engine, assignment.expr_.nodeId(), context);
		}

		std::vector<std::pair<__m256 *, ValueRef>> stores{};
		stores.reserve(assignments.size());

		for (const Assignment &assignment : assignments) {
			ValueRef value{
			    compileNode(engine, plan, assignment.expr_.nodeId(), context)};
			stores.push_back({assignment.out_->variable.array().data(), value});
		}

		for (const std::pair<__m256 *, ValueRef> &store : stores) {
			emitStore(plan, store.first, store.second, context);
		}

		return plan;
	}

	static ScheduledPlan compileScheduled(const Engine &engine,
	                                      const std::vector<Assignment> &assignments)
	{
		ScheduledPlan scheduled{};
		if (assignments.empty())
			return scheduled;

		validateAssignments(engine, assignments);

		const std::vector<std::vector<Assignment>> stages{buildStages(engine, assignments)};
		scheduled.impl_->blockCount =
		    assignments.front().out_->variable.array().blockCount();
		scheduled.impl_->stages.reserve(stages.size());

		for (const std::vector<Assignment> &stage : stages) {
			scheduled.impl_->stages.push_back(compileFusion(engine, stage));
		}

		return scheduled;
	}
};
}

Engine::Engine() : impl_{std::make_unique<internal::EngineData>()} {}

Engine::~Engine() noexcept = default;

Array Engine::createArray(std::size_t elementCount) { return Array{*this, elementCount}; }

Array Engine::createArray(const std::vector<float> &values)
{
	Array result{createArray()};
	result.copyFrom(values);
	return result;
}

Array Engine::createArray(std::initializer_list<float> values)
{
	Array result{createArray()};
	result.copyFrom(values);
	return result;
}

Array Engine::createArray(std::size_t elementCount, float value)
{
	Array result{createArray()};
	result.assign(elementCount, value);
	return result;
}

Expression Engine::makeVariable(const Array &value)
{
	assert(value.impl_ != nullptr);
	const Variable &variable{value.impl_->variable};
	internal::Compiler::requireVarOwner(*this, variable, "変数が別のEngineに紐づいています。");

	ExprNode node{};
	node.kind = NodeKind::Variable;
	node.array = &variable.array();
	node.key =
	    internal::Compiler::combineHash(1, reinterpret_cast<std::uintptr_t>(node.array), 0);
	return internal::Compiler::appendNode(*this, node);
}

Expression Engine::makeScalar(float value)
{
	ExprNode node{};
	node.kind = NodeKind::Scalar;
	node.scalar = value;
	node.key = internal::Compiler::combineHash(2, internal::Compiler::floatBits(value), 0);
	return internal::Compiler::appendNode(*this, node);
}

Expression Engine::makeAdd(const Expression &lhs, const Expression &rhs)
{
	internal::Compiler::requireExprOwner(*this, lhs,
	                                     "左辺式が別のEngineに紐づいているか、無効です。");
	internal::Compiler::requireExprOwner(*this, rhs,
	                                     "右辺式が別のEngineに紐づいているか、無効です。");

	return internal::Compiler::makeBinary(*this, NodeKind::Add, lhs, rhs);
}

Expression Engine::makeSub(const Expression &lhs, const Expression &rhs)
{
	internal::Compiler::requireExprOwner(*this, lhs,
	                                     "左辺式が別のEngineに紐づいているか、無効です。");
	internal::Compiler::requireExprOwner(*this, rhs,
	                                     "右辺式が別のEngineに紐づいているか、無効です。");

	return internal::Compiler::makeBinary(*this, NodeKind::Sub, lhs, rhs);
}

Expression Engine::makeMul(const Expression &lhs, const Expression &rhs)
{
	internal::Compiler::requireExprOwner(*this, lhs,
	                                     "左辺式が別のEngineに紐づいているか、無効です。");
	internal::Compiler::requireExprOwner(*this, rhs,
	                                     "右辺式が別のEngineに紐づいているか、無効です。");

	return internal::Compiler::makeBinary(*this, NodeKind::Mul, lhs, rhs);
}

Expression Engine::makeDiv(const Expression &lhs, const Expression &rhs)
{
	internal::Compiler::requireExprOwner(*this, lhs,
	                                     "左辺式が別のEngineに紐づいているか、無効です。");
	internal::Compiler::requireExprOwner(*this, rhs,
	                                     "右辺式が別のEngineに紐づいているか、無効です。");

	return internal::Compiler::makeBinary(*this, NodeKind::Div, lhs, rhs);
}

void Engine::deferAssign(Array &out, const Expression &expr)
{
	assert(out.impl_ != nullptr);
	Variable &variable{out.impl_->variable};
	internal::Compiler::requireVarOwner(*this, variable,
	                                    "代入先が別のEngineに紐づいています。");
	internal::Compiler::requireExprOwner(*this, expr,
	                                     "代入式が別のEngineに紐づいているか、無効です。");

	impl_->pendingAssignments.push_back(Assignment{out.impl_.get(), expr});
}

void Engine::execute()
{
	ScheduledPlan plan{compile()};
	plan.execute();
}

ScheduledPlan Engine::compile()
{
	ScheduledPlan plan{internal::Compiler::compileScheduled(*this, impl_->pendingAssignments)};
	impl_->pendingAssignments.clear();
	internal::Compiler::rememberExpressionReserve(*this);
	impl_->nodes.clear();
	return plan;
}

Expression operator+(const Expression &lhs, const Expression &rhs)
{
	return lhs.engine()->makeAdd(lhs, rhs);
}
Expression operator+(const Array &lhs, const Array &rhs)
{
	Expression lhsExpr{lhs.engine()->makeVariable(lhs)};
	return lhsExpr.engine()->makeAdd(lhsExpr, lhsExpr.engine()->makeVariable(rhs));
}
Expression operator+(const Array &lhs, const Expression &rhs)
{
	Expression lhsExpr{lhs.engine()->makeVariable(lhs)};
	return lhsExpr.engine()->makeAdd(lhsExpr, rhs);
}
Expression operator+(const Expression &lhs, const Array &rhs)
{
	return lhs.engine()->makeAdd(lhs, lhs.engine()->makeVariable(rhs));
}
Expression operator+(const Expression &lhs, float rhs)
{
	return lhs.engine()->makeAdd(lhs, lhs.engine()->makeScalar(rhs));
}
Expression operator+(float lhs, const Expression &rhs)
{
	return rhs.engine()->makeAdd(rhs.engine()->makeScalar(lhs), rhs);
}
Expression operator+(const Array &lhs, float rhs)
{
	Expression lhsExpr{lhs.engine()->makeVariable(lhs)};
	return lhsExpr.engine()->makeAdd(lhsExpr, lhsExpr.engine()->makeScalar(rhs));
}
Expression operator+(float lhs, const Array &rhs)
{
	Expression rhsExpr{rhs.engine()->makeVariable(rhs)};
	return rhsExpr.engine()->makeAdd(rhsExpr.engine()->makeScalar(lhs), rhsExpr);
}

Expression operator-(const Expression &value)
{
	return value.engine()->makeMul(value.engine()->makeScalar(-1.0f), value);
}
Expression operator-(const Array &value) { return -value.engine()->makeVariable(value); }
Expression operator-(const Expression &lhs, const Expression &rhs)
{
	return lhs.engine()->makeSub(lhs, rhs);
}
Expression operator-(const Array &lhs, const Array &rhs)
{
	Expression lhsExpr{lhs.engine()->makeVariable(lhs)};
	return lhsExpr.engine()->makeSub(lhsExpr, lhsExpr.engine()->makeVariable(rhs));
}
Expression operator-(const Array &lhs, const Expression &rhs)
{
	Expression lhsExpr{lhs.engine()->makeVariable(lhs)};
	return lhsExpr.engine()->makeSub(lhsExpr, rhs);
}
Expression operator-(const Expression &lhs, const Array &rhs)
{
	return lhs.engine()->makeSub(lhs, lhs.engine()->makeVariable(rhs));
}
Expression operator-(const Expression &lhs, float rhs)
{
	return lhs.engine()->makeSub(lhs, lhs.engine()->makeScalar(rhs));
}
Expression operator-(float lhs, const Expression &rhs)
{
	return rhs.engine()->makeSub(rhs.engine()->makeScalar(lhs), rhs);
}
Expression operator-(const Array &lhs, float rhs)
{
	Expression lhsExpr{lhs.engine()->makeVariable(lhs)};
	return lhsExpr.engine()->makeSub(lhsExpr, lhsExpr.engine()->makeScalar(rhs));
}
Expression operator-(float lhs, const Array &rhs)
{
	Expression rhsExpr{rhs.engine()->makeVariable(rhs)};
	return rhsExpr.engine()->makeSub(rhsExpr.engine()->makeScalar(lhs), rhsExpr);
}

Expression operator*(const Expression &lhs, const Expression &rhs)
{
	return lhs.engine()->makeMul(lhs, rhs);
}
Expression operator*(const Array &lhs, const Array &rhs)
{
	Expression lhsExpr{lhs.engine()->makeVariable(lhs)};
	return lhsExpr.engine()->makeMul(lhsExpr, lhsExpr.engine()->makeVariable(rhs));
}
Expression operator*(const Array &lhs, const Expression &rhs)
{
	Expression lhsExpr{lhs.engine()->makeVariable(lhs)};
	return lhsExpr.engine()->makeMul(lhsExpr, rhs);
}
Expression operator*(const Expression &lhs, const Array &rhs)
{
	return lhs.engine()->makeMul(lhs, lhs.engine()->makeVariable(rhs));
}
Expression operator*(const Expression &lhs, float rhs)
{
	return lhs.engine()->makeMul(lhs, lhs.engine()->makeScalar(rhs));
}
Expression operator*(float lhs, const Expression &rhs)
{
	return rhs.engine()->makeMul(rhs.engine()->makeScalar(lhs), rhs);
}
Expression operator*(const Array &lhs, float rhs)
{
	Expression lhsExpr{lhs.engine()->makeVariable(lhs)};
	return lhsExpr.engine()->makeMul(lhsExpr, lhsExpr.engine()->makeScalar(rhs));
}
Expression operator*(float lhs, const Array &rhs)
{
	Expression rhsExpr{rhs.engine()->makeVariable(rhs)};
	return rhsExpr.engine()->makeMul(rhsExpr.engine()->makeScalar(lhs), rhsExpr);
}

Expression operator/(const Expression &lhs, const Expression &rhs)
{
	return lhs.engine()->makeDiv(lhs, rhs);
}
Expression operator/(const Array &lhs, const Array &rhs)
{
	Expression lhsExpr{lhs.engine()->makeVariable(lhs)};
	return lhsExpr.engine()->makeDiv(lhsExpr, lhsExpr.engine()->makeVariable(rhs));
}
Expression operator/(const Array &lhs, const Expression &rhs)
{
	Expression lhsExpr{lhs.engine()->makeVariable(lhs)};
	return lhsExpr.engine()->makeDiv(lhsExpr, rhs);
}
Expression operator/(const Expression &lhs, const Array &rhs)
{
	return lhs.engine()->makeDiv(lhs, lhs.engine()->makeVariable(rhs));
}
Expression operator/(const Expression &lhs, float rhs)
{
	return lhs.engine()->makeDiv(lhs, lhs.engine()->makeScalar(rhs));
}
Expression operator/(float lhs, const Expression &rhs)
{
	return rhs.engine()->makeDiv(rhs.engine()->makeScalar(lhs), rhs);
}
Expression operator/(const Array &lhs, float rhs)
{
	Expression lhsExpr{lhs.engine()->makeVariable(lhs)};
	return lhsExpr.engine()->makeDiv(lhsExpr, lhsExpr.engine()->makeScalar(rhs));
}
Expression operator/(float lhs, const Array &rhs)
{
	Expression rhsExpr{rhs.engine()->makeVariable(rhs)};
	return rhsExpr.engine()->makeDiv(rhsExpr.engine()->makeScalar(lhs), rhsExpr);
}

}
