#include "simd_internal.h"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <stdexcept>
#include <utility>

namespace rice::simd
{
using internal::addBlock;
using internal::Assignment;
using internal::AssignmentKind;
using internal::CompileContext;
using internal::CompileGroup;
using internal::divBlock;
using internal::ExprNode;
using internal::FloatArray;
using internal::FusionPlan;
using internal::KeyIndex;
using internal::loadAlignedBlock;
using internal::loadUnalignedBlock;
using internal::MAX_REGISTERS;
using internal::mulBlock;
using internal::multiplyAddBlock;
using internal::negativeMultiplyAddBlock;
using internal::NodeKind;
using internal::PendingStore;
using internal::PendingStoreKind;
using internal::PlanCacheKey;
using internal::set1Block;
using internal::SIMD_ALIGNMENT;
using internal::SIMD_WIDTH;
using internal::SimdBlock;
using internal::storeAlignedBlock;
using internal::storeUnalignedBlock;
using internal::subBlock;
using internal::ValueKind;
using internal::ValueRef;
using internal::Variable;
using internal::zeroBlock;

internal::FloatArray::FloatArray(std::size_t elementCount) { resize(elementCount); }

void internal::FloatArray::resize(std::size_t elementCount)
{
	elementCount_ = elementCount;
	blocks_.assign((elementCount + SIMD_WIDTH - 1) / SIMD_WIDTH, zeroBlock());
}

std::size_t internal::FloatArray::elementCount() const noexcept { return elementCount_; }

std::size_t internal::FloatArray::blockCount() const noexcept { return blocks_.size(); }

SimdBlock &internal::FloatArray::block(std::size_t index) noexcept { return blocks_[index]; }

const SimdBlock &internal::FloatArray::block(std::size_t index) const noexcept
{
	return blocks_[index];
}

SimdBlock *internal::FloatArray::data() noexcept { return blocks_.data(); }

const SimdBlock *internal::FloatArray::data() const noexcept { return blocks_.data(); }

void internal::FloatArray::fill(float value) noexcept
{
	const SimdBlock block{set1Block(value)};

	for (SimdBlock &stored : blocks_) {
		stored = block;
	}
}

void internal::FloatArray::assign(std::size_t elementCount, float value)
{
	if (elementCount_ != elementCount) {
		resize(elementCount);
	}

	fill(value);
}

void internal::FloatArray::copyFrom(const FloatArray &source)
{
	if (elementCount_ != source.elementCount_) {
		*this = source;
		return;
	}

	std::copy(source.blocks_.begin(), source.blocks_.end(), blocks_.begin());
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
		blocks_.push_back(zeroBlock());
	}

	++elementCount_;
	setElement(index, value);
}

float internal::FloatArray::element(std::size_t index) const noexcept
{
	alignas(SIMD_ALIGNMENT) float temp[SIMD_WIDTH]{};
	const std::size_t blockIndex{index / SIMD_WIDTH};
	const std::size_t lane{index % SIMD_WIDTH};

	storeAlignedBlock(temp, blocks_[blockIndex]);
	return temp[lane];
}

void internal::FloatArray::copyTo(std::vector<float> &out) const
{
	out.resize(elementCount_);
	if (elementCount_ == 0) {
		return;
	}

	const std::size_t fullBlocks{elementCount_ / SIMD_WIDTH};
	for (std::size_t blockIndex{}; blockIndex < fullBlocks; ++blockIndex) {
		storeUnalignedBlock(out.data() + blockIndex * SIMD_WIDTH, blocks_[blockIndex]);
	}

	const std::size_t tailCount{elementCount_ % SIMD_WIDTH};
	if (tailCount == 0) {
		return;
	}

	alignas(SIMD_ALIGNMENT) float temp[SIMD_WIDTH]{};
	storeAlignedBlock(temp, blocks_[fullBlocks]);

	const std::size_t offset{fullBlocks * SIMD_WIDTH};
	for (std::size_t lane{}; lane < tailCount; ++lane) {
		out[offset + lane] = temp[lane];
	}
}

void internal::FloatArray::copyFrom(const float *values, std::size_t count)
{
	if (elementCount_ != count) {
		resize(count);
	}

	const std::size_t fullBlocks{count / SIMD_WIDTH};
	for (std::size_t blockIndex{}; blockIndex < fullBlocks; ++blockIndex) {
		blocks_[blockIndex] = loadUnalignedBlock(values + blockIndex * SIMD_WIDTH);
	}

	const std::size_t tailCount{count % SIMD_WIDTH};
	if (tailCount != 0) {
		alignas(SIMD_ALIGNMENT) float temp[SIMD_WIDTH]{};
		const std::size_t offset{fullBlocks * SIMD_WIDTH};

		for (std::size_t lane{}; lane < tailCount; ++lane) {
			temp[lane] = values[offset + lane];
		}

		blocks_[fullBlocks] = loadAlignedBlock(temp);
	}
}

void internal::FloatArray::setElement(std::size_t index, float value) noexcept
{
	alignas(SIMD_ALIGNMENT) float temp[SIMD_WIDTH]{};
	const std::size_t blockIndex{index / SIMD_WIDTH};
	const std::size_t lane{index % SIMD_WIDTH};

	storeAlignedBlock(temp, blocks_[blockIndex]);
	temp[lane] = value;
	blocks_[blockIndex] = loadAlignedBlock(temp);
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

Array::Array(Engine &engine, const std::vector<float> &values) : Array{engine} { copyFrom(values); }

Array::Array(Engine &engine, std::initializer_list<float> values) : Array{engine}
{
	copyFrom(values);
}

Array::Array(Engine &engine, std::size_t elementCount, float value) : Array{engine}
{
	assign(elementCount, value);
}

Array::~Array() noexcept
{
	if (impl_ != nullptr && impl_->variable.engine() != nullptr) {
		impl_->variable.engine()->impl_->hasCachedPlan = false;
	}
}
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

namespace
{
void requireCompoundAssignmentTarget(const Array &target)
{
	if (target.elementCount() == 0) {
		throw std::invalid_argument{"複合代入の左辺配列に要素がありません。resize()"
		                            "や代入でサイズを設定してください。"};
	}
}

void requireElementIndex(const Array &array, std::size_t index)
{
	if (index >= array.size()) {
		throw std::out_of_range{"Arrayの要素番号が範囲外です。"};
	}
}
}

Array &Array::operator+=(const Expression &expr)
{
	requireCompoundAssignmentTarget(*this);
	assert(impl_ != nullptr);
	impl_->variable.engine()->deferCompoundAssign(*this, expr, AssignmentKind::AddAssign);
	return *this;
}

Array &Array::operator+=(const Array &value)
{
	requireCompoundAssignmentTarget(*this);
	assert(impl_ != nullptr);
	Engine *engine{impl_->variable.engine()};
	engine->deferCompoundAssign(*this, engine->makeVariable(value), AssignmentKind::AddAssign);
	return *this;
}

Array &Array::operator+=(float value)
{
	requireCompoundAssignmentTarget(*this);
	assert(impl_ != nullptr);
	Engine *engine{impl_->variable.engine()};
	engine->deferCompoundAssign(*this, engine->makeScalar(value), AssignmentKind::AddAssign);
	return *this;
}

Array &Array::operator-=(const Expression &expr)
{
	requireCompoundAssignmentTarget(*this);
	assert(impl_ != nullptr);
	impl_->variable.engine()->deferCompoundAssign(*this, expr, AssignmentKind::SubAssign);
	return *this;
}

Array &Array::operator-=(const Array &value)
{
	requireCompoundAssignmentTarget(*this);
	assert(impl_ != nullptr);
	Engine *engine{impl_->variable.engine()};
	engine->deferCompoundAssign(*this, engine->makeVariable(value), AssignmentKind::SubAssign);
	return *this;
}

Array &Array::operator-=(float value)
{
	requireCompoundAssignmentTarget(*this);
	assert(impl_ != nullptr);
	Engine *engine{impl_->variable.engine()};
	engine->deferCompoundAssign(*this, engine->makeScalar(value), AssignmentKind::SubAssign);
	return *this;
}

Array &Array::operator*=(const Expression &expr)
{
	requireCompoundAssignmentTarget(*this);
	assert(impl_ != nullptr);
	impl_->variable.engine()->deferCompoundAssign(*this, expr, AssignmentKind::MulAssign);
	return *this;
}

Array &Array::operator*=(const Array &value)
{
	requireCompoundAssignmentTarget(*this);
	assert(impl_ != nullptr);
	Engine *engine{impl_->variable.engine()};
	engine->deferCompoundAssign(*this, engine->makeVariable(value), AssignmentKind::MulAssign);
	return *this;
}

Array &Array::operator*=(float value)
{
	requireCompoundAssignmentTarget(*this);
	assert(impl_ != nullptr);
	Engine *engine{impl_->variable.engine()};
	engine->deferCompoundAssign(*this, engine->makeScalar(value), AssignmentKind::MulAssign);
	return *this;
}

Array &Array::operator/=(const Expression &expr)
{
	requireCompoundAssignmentTarget(*this);
	assert(impl_ != nullptr);
	impl_->variable.engine()->deferCompoundAssign(*this, expr, AssignmentKind::DivAssign);
	return *this;
}

Array &Array::operator/=(const Array &value)
{
	requireCompoundAssignmentTarget(*this);
	assert(impl_ != nullptr);
	Engine *engine{impl_->variable.engine()};
	engine->deferCompoundAssign(*this, engine->makeVariable(value), AssignmentKind::DivAssign);
	return *this;
}

Array &Array::operator/=(float value)
{
	requireCompoundAssignmentTarget(*this);
	assert(impl_ != nullptr);
	Engine *engine{impl_->variable.engine()};
	engine->deferCompoundAssign(*this, engine->makeScalar(value), AssignmentKind::DivAssign);
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
	impl_->storage.copyFrom(source.impl_->storage);
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

Array::ElementProxy::ElementProxy(Array &array, std::size_t index) noexcept
    : array_{&array}, index_{index}
{
}

Array::ElementProxy &Array::ElementProxy::operator=(float value)
{
	assert(array_ != nullptr);
	array_->set(index_, value);
	return *this;
}

Array::ElementProxy &Array::ElementProxy::operator=(const ElementProxy &value)
{
	return *this = static_cast<float>(value);
}

Array::ElementProxy::operator float() const
{
	assert(array_ != nullptr);
	return array_->get(index_);
}

float Array::get(std::size_t index) const
{
	assert(impl_ != nullptr);
	requireElementIndex(*this, index);
	return impl_->storage.element(index);
}

void Array::set(std::size_t index, float value)
{
	assert(impl_ != nullptr);
	requireElementIndex(*this, index);
	impl_->storage.setElement(index, value);
}

Array::ElementProxy Array::operator[](std::size_t index)
{
	requireElementIndex(*this, index);
	return ElementProxy{*this, index};
}

float Array::operator[](std::size_t index) const
{
	return get(index);
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
void executeAddRR(const internal::FusionPlanData &plan, std::size_t, SimdBlock *regs,
                  std::size_t index) noexcept
{
	const internal::AddRR &op{plan.addRR[index]};
	regs[op.dst] = addBlock(regs[op.a], regs[op.b]);
}

void executeAddRA(const internal::FusionPlanData &plan, std::size_t blockIndex, SimdBlock *regs,
                  std::size_t index) noexcept
{
	const internal::AddRA &op{plan.addRA[index]};
	regs[op.dst] = addBlock(regs[op.a], op.b[blockIndex]);
}

void executeAddAR(const internal::FusionPlanData &plan, std::size_t blockIndex, SimdBlock *regs,
                  std::size_t index) noexcept
{
	const internal::AddAR &op{plan.addAR[index]};
	regs[op.dst] = addBlock(op.a[blockIndex], regs[op.b]);
}

void executeAddAA(const internal::FusionPlanData &plan, std::size_t blockIndex, SimdBlock *regs,
                  std::size_t index) noexcept
{
	const internal::AddAA &op{plan.addAA[index]};
	regs[op.dst] = addBlock(op.a[blockIndex], op.b[blockIndex]);
}

void executeAddRS(const internal::FusionPlanData &plan, std::size_t, SimdBlock *regs,
                  std::size_t index) noexcept
{
	const internal::AddRS &op{plan.addRS[index]};
	regs[op.dst] = addBlock(regs[op.a], op.b);
}

void executeAddAS(const internal::FusionPlanData &plan, std::size_t blockIndex, SimdBlock *regs,
                  std::size_t index) noexcept
{
	const internal::AddAS &op{plan.addAS[index]};
	regs[op.dst] = addBlock(op.a[blockIndex], op.b);
}

void executeSubRR(const internal::FusionPlanData &plan, std::size_t, SimdBlock *regs,
                  std::size_t index) noexcept
{
	const internal::SubRR &op{plan.subRR[index]};
	regs[op.dst] = subBlock(regs[op.a], regs[op.b]);
}

void executeSubRA(const internal::FusionPlanData &plan, std::size_t blockIndex, SimdBlock *regs,
                  std::size_t index) noexcept
{
	const internal::SubRA &op{plan.subRA[index]};
	regs[op.dst] = subBlock(regs[op.a], op.b[blockIndex]);
}

void executeSubAR(const internal::FusionPlanData &plan, std::size_t blockIndex, SimdBlock *regs,
                  std::size_t index) noexcept
{
	const internal::SubAR &op{plan.subAR[index]};
	regs[op.dst] = subBlock(op.a[blockIndex], regs[op.b]);
}

void executeSubAA(const internal::FusionPlanData &plan, std::size_t blockIndex, SimdBlock *regs,
                  std::size_t index) noexcept
{
	const internal::SubAA &op{plan.subAA[index]};
	regs[op.dst] = subBlock(op.a[blockIndex], op.b[blockIndex]);
}

void executeSubRS(const internal::FusionPlanData &plan, std::size_t, SimdBlock *regs,
                  std::size_t index) noexcept
{
	const internal::SubRS &op{plan.subRS[index]};
	regs[op.dst] = subBlock(regs[op.a], op.b);
}

void executeSubAS(const internal::FusionPlanData &plan, std::size_t blockIndex, SimdBlock *regs,
                  std::size_t index) noexcept
{
	const internal::SubAS &op{plan.subAS[index]};
	regs[op.dst] = subBlock(op.a[blockIndex], op.b);
}

void executeSubSR(const internal::FusionPlanData &plan, std::size_t, SimdBlock *regs,
                  std::size_t index) noexcept
{
	const internal::SubSR &op{plan.subSR[index]};
	regs[op.dst] = subBlock(op.a, regs[op.b]);
}

void executeSubSA(const internal::FusionPlanData &plan, std::size_t blockIndex, SimdBlock *regs,
                  std::size_t index) noexcept
{
	const internal::SubSA &op{plan.subSA[index]};
	regs[op.dst] = subBlock(op.a, op.b[blockIndex]);
}

void executeMulRR(const internal::FusionPlanData &plan, std::size_t, SimdBlock *regs,
                  std::size_t index) noexcept
{
	const internal::MulRR &op{plan.mulRR[index]};
	regs[op.dst] = mulBlock(regs[op.a], regs[op.b]);
}

void executeMulRA(const internal::FusionPlanData &plan, std::size_t blockIndex, SimdBlock *regs,
                  std::size_t index) noexcept
{
	const internal::MulRA &op{plan.mulRA[index]};
	regs[op.dst] = mulBlock(regs[op.a], op.b[blockIndex]);
}

void executeMulAR(const internal::FusionPlanData &plan, std::size_t blockIndex, SimdBlock *regs,
                  std::size_t index) noexcept
{
	const internal::MulAR &op{plan.mulAR[index]};
	regs[op.dst] = mulBlock(op.a[blockIndex], regs[op.b]);
}

void executeMulAA(const internal::FusionPlanData &plan, std::size_t blockIndex, SimdBlock *regs,
                  std::size_t index) noexcept
{
	const internal::MulAA &op{plan.mulAA[index]};
	regs[op.dst] = mulBlock(op.a[blockIndex], op.b[blockIndex]);
}

void executeMulRS(const internal::FusionPlanData &plan, std::size_t, SimdBlock *regs,
                  std::size_t index) noexcept
{
	const internal::MulRS &op{plan.mulRS[index]};
	regs[op.dst] = mulBlock(regs[op.a], op.b);
}

void executeMulAS(const internal::FusionPlanData &plan, std::size_t blockIndex, SimdBlock *regs,
                  std::size_t index) noexcept
{
	const internal::MulAS &op{plan.mulAS[index]};
	regs[op.dst] = mulBlock(op.a[blockIndex], op.b);
}

void executeDivRR(const internal::FusionPlanData &plan, std::size_t, SimdBlock *regs,
                  std::size_t index) noexcept
{
	const internal::DivRR &op{plan.divRR[index]};
	regs[op.dst] = divBlock(regs[op.a], regs[op.b]);
}

void executeDivRA(const internal::FusionPlanData &plan, std::size_t blockIndex, SimdBlock *regs,
                  std::size_t index) noexcept
{
	const internal::DivRA &op{plan.divRA[index]};
	regs[op.dst] = divBlock(regs[op.a], op.b[blockIndex]);
}

void executeDivAR(const internal::FusionPlanData &plan, std::size_t blockIndex, SimdBlock *regs,
                  std::size_t index) noexcept
{
	const internal::DivAR &op{plan.divAR[index]};
	regs[op.dst] = divBlock(op.a[blockIndex], regs[op.b]);
}

void executeDivAA(const internal::FusionPlanData &plan, std::size_t blockIndex, SimdBlock *regs,
                  std::size_t index) noexcept
{
	const internal::DivAA &op{plan.divAA[index]};
	regs[op.dst] = divBlock(op.a[blockIndex], op.b[blockIndex]);
}

void executeDivRS(const internal::FusionPlanData &plan, std::size_t, SimdBlock *regs,
                  std::size_t index) noexcept
{
	const internal::DivRS &op{plan.divRS[index]};
	regs[op.dst] = divBlock(regs[op.a], op.b);
}

void executeDivAS(const internal::FusionPlanData &plan, std::size_t blockIndex, SimdBlock *regs,
                  std::size_t index) noexcept
{
	const internal::DivAS &op{plan.divAS[index]};
	regs[op.dst] = divBlock(op.a[blockIndex], op.b);
}

void executeDivSR(const internal::FusionPlanData &plan, std::size_t, SimdBlock *regs,
                  std::size_t index) noexcept
{
	const internal::DivSR &op{plan.divSR[index]};
	regs[op.dst] = divBlock(op.a, regs[op.b]);
}

void executeDivSA(const internal::FusionPlanData &plan, std::size_t blockIndex, SimdBlock *regs,
                  std::size_t index) noexcept
{
	const internal::DivSA &op{plan.divSA[index]};
	regs[op.dst] = divBlock(op.a, op.b[blockIndex]);
}

void executeFmaRRR(const internal::FusionPlanData &plan, std::size_t, SimdBlock *regs,
                   std::size_t index) noexcept
{
	const internal::FmaRRR &op{plan.fmaRRR[index]};
	regs[op.dst] = multiplyAddBlock(regs[op.a], regs[op.b], regs[op.c]);
}

void executeFmaRRA(const internal::FusionPlanData &plan, std::size_t blockIndex, SimdBlock *regs,
                   std::size_t index) noexcept
{
	const internal::FmaRRA &op{plan.fmaRRA[index]};
	regs[op.dst] = multiplyAddBlock(regs[op.a], regs[op.b], op.c[blockIndex]);
}

void executeFmaRAA(const internal::FusionPlanData &plan, std::size_t blockIndex, SimdBlock *regs,
                   std::size_t index) noexcept
{
	const internal::FmaRAA &op{plan.fmaRAA[index]};
	regs[op.dst] = multiplyAddBlock(regs[op.a], op.b[blockIndex], op.c[blockIndex]);
}

void executeFmaAAA(const internal::FusionPlanData &plan, std::size_t blockIndex, SimdBlock *regs,
                   std::size_t index) noexcept
{
	const internal::FmaAAA &op{plan.fmaAAA[index]};
	regs[op.dst] = multiplyAddBlock(op.a[blockIndex], op.b[blockIndex], op.c[blockIndex]);
}

void executeFmaASA(const internal::FusionPlanData &plan, std::size_t blockIndex, SimdBlock *regs,
                   std::size_t index) noexcept
{
	const internal::FmaASA &op{plan.fmaASA[index]};
	regs[op.dst] = multiplyAddBlock(op.a[blockIndex], op.b, op.c[blockIndex]);
}

void executeFmaRAS(const internal::FusionPlanData &plan, std::size_t blockIndex, SimdBlock *regs,
                   std::size_t index) noexcept
{
	const internal::FmaRAS &op{plan.fmaRAS[index]};
	regs[op.dst] = multiplyAddBlock(regs[op.a], op.b[blockIndex], op.c);
}

void executeFmaRSA(const internal::FusionPlanData &plan, std::size_t blockIndex, SimdBlock *regs,
                   std::size_t index) noexcept
{
	const internal::FmaRSA &op{plan.fmaRSA[index]};
	regs[op.dst] = multiplyAddBlock(regs[op.a], op.b, op.c[blockIndex]);
}

void executeStoreFmaAAA(const internal::FusionPlanData &plan, std::size_t blockIndex, SimdBlock *,
                        std::size_t index) noexcept
{
	const internal::StoreFmaAAA &op{plan.storeFmaAAA[index]};
	op.out[blockIndex] = multiplyAddBlock(op.a[blockIndex], op.b[blockIndex], op.c[blockIndex]);
}

void executeStoreFmaASA(const internal::FusionPlanData &plan, std::size_t blockIndex, SimdBlock *,
                        std::size_t index) noexcept
{
	const internal::StoreFmaASA &op{plan.storeFmaASA[index]};
	op.out[blockIndex] = multiplyAddBlock(op.a[blockIndex], op.b, op.c[blockIndex]);
}

void executeStoreFmaRRR(const internal::FusionPlanData &plan, std::size_t blockIndex,
                        SimdBlock *regs, std::size_t index) noexcept
{
	const internal::StoreFmaRRR &op{plan.storeFmaRRR[index]};
	op.out[blockIndex] = multiplyAddBlock(regs[op.a], regs[op.b], regs[op.c]);
}

void executeStoreFmaRRA(const internal::FusionPlanData &plan, std::size_t blockIndex,
                        SimdBlock *regs, std::size_t index) noexcept
{
	const internal::StoreFmaRRA &op{plan.storeFmaRRA[index]};
	op.out[blockIndex] = multiplyAddBlock(regs[op.a], regs[op.b], op.c[blockIndex]);
}

void executeStoreFmaRAA(const internal::FusionPlanData &plan, std::size_t blockIndex,
                        SimdBlock *regs, std::size_t index) noexcept
{
	const internal::StoreFmaRAA &op{plan.storeFmaRAA[index]};
	op.out[blockIndex] = multiplyAddBlock(regs[op.a], op.b[blockIndex], op.c[blockIndex]);
}

void executeStoreFmaRAS(const internal::FusionPlanData &plan, std::size_t blockIndex,
                        SimdBlock *regs, std::size_t index) noexcept
{
	const internal::StoreFmaRAS &op{plan.storeFmaRAS[index]};
	op.out[blockIndex] = multiplyAddBlock(regs[op.a], op.b[blockIndex], op.c);
}

void executeStoreFmaRSA(const internal::FusionPlanData &plan, std::size_t blockIndex,
                        SimdBlock *regs, std::size_t index) noexcept
{
	const internal::StoreFmaRSA &op{plan.storeFmaRSA[index]};
	op.out[blockIndex] = multiplyAddBlock(regs[op.a], op.b, op.c[blockIndex]);
}

void executeStoreNegFmaAAA(const internal::FusionPlanData &plan, std::size_t blockIndex,
                           SimdBlock *, std::size_t index) noexcept
{
	const internal::StoreNegFmaAAA &op{plan.storeNegFmaAAA[index]};
	op.out[blockIndex] =
	    negativeMultiplyAddBlock(op.a[blockIndex], op.b[blockIndex], op.c[blockIndex]);
}

void executeStoreNegFmaASA(const internal::FusionPlanData &plan, std::size_t blockIndex,
                           SimdBlock *, std::size_t index) noexcept
{
	const internal::StoreNegFmaASA &op{plan.storeNegFmaASA[index]};
	op.out[blockIndex] = negativeMultiplyAddBlock(op.a[blockIndex], op.b, op.c[blockIndex]);
}

void executeStoreAddAA(const internal::FusionPlanData &plan, std::size_t blockIndex, SimdBlock *,
                       std::size_t index) noexcept
{
	const internal::StoreBinaryAA &op{plan.storeAddAA[index]};
	op.out[blockIndex] = addBlock(op.a[blockIndex], op.b[blockIndex]);
}

void executeStoreAddAS(const internal::FusionPlanData &plan, std::size_t blockIndex, SimdBlock *,
                       std::size_t index) noexcept
{
	const internal::StoreBinaryAS &op{plan.storeAddAS[index]};
	op.out[blockIndex] = addBlock(op.a[blockIndex], op.b);
}

void executeStoreSubAA(const internal::FusionPlanData &plan, std::size_t blockIndex, SimdBlock *,
                       std::size_t index) noexcept
{
	const internal::StoreBinaryAA &op{plan.storeSubAA[index]};
	op.out[blockIndex] = subBlock(op.a[blockIndex], op.b[blockIndex]);
}

void executeStoreSubAS(const internal::FusionPlanData &plan, std::size_t blockIndex, SimdBlock *,
                       std::size_t index) noexcept
{
	const internal::StoreBinaryAS &op{plan.storeSubAS[index]};
	op.out[blockIndex] = subBlock(op.a[blockIndex], op.b);
}

void executeStoreSubSA(const internal::FusionPlanData &plan, std::size_t blockIndex, SimdBlock *,
                       std::size_t index) noexcept
{
	const internal::StoreBinarySA &op{plan.storeSubSA[index]};
	op.out[blockIndex] = subBlock(op.a, op.b[blockIndex]);
}

void executeStoreMulAA(const internal::FusionPlanData &plan, std::size_t blockIndex, SimdBlock *,
                       std::size_t index) noexcept
{
	const internal::StoreBinaryAA &op{plan.storeMulAA[index]};
	op.out[blockIndex] = mulBlock(op.a[blockIndex], op.b[blockIndex]);
}

void executeStoreMulAS(const internal::FusionPlanData &plan, std::size_t blockIndex, SimdBlock *,
                       std::size_t index) noexcept
{
	const internal::StoreBinaryAS &op{plan.storeMulAS[index]};
	op.out[blockIndex] = mulBlock(op.a[blockIndex], op.b);
}

void executeStoreDivAA(const internal::FusionPlanData &plan, std::size_t blockIndex, SimdBlock *,
                       std::size_t index) noexcept
{
	const internal::StoreBinaryAA &op{plan.storeDivAA[index]};
	op.out[blockIndex] = divBlock(op.a[blockIndex], op.b[blockIndex]);
}

void executeStoreDivAS(const internal::FusionPlanData &plan, std::size_t blockIndex, SimdBlock *,
                       std::size_t index) noexcept
{
	const internal::StoreBinaryAS &op{plan.storeDivAS[index]};
	op.out[blockIndex] = divBlock(op.a[blockIndex], op.b);
}

void executeStoreDivSA(const internal::FusionPlanData &plan, std::size_t blockIndex, SimdBlock *,
                       std::size_t index) noexcept
{
	const internal::StoreBinarySA &op{plan.storeDivSA[index]};
	op.out[blockIndex] = divBlock(op.a, op.b[blockIndex]);
}

void executeSetS(const internal::FusionPlanData &plan, std::size_t, SimdBlock *regs,
                 std::size_t index) noexcept
{
	const internal::SetS &op{plan.setS[index]};
	regs[op.dst] = op.a;
}

void executeStoreR(const internal::FusionPlanData &plan, std::size_t blockIndex, SimdBlock *regs,
                   std::size_t index) noexcept
{
	const internal::StoreR &op{plan.storeR[index]};
	op.out[blockIndex] = regs[op.a];
}

void executeStoreA(const internal::FusionPlanData &plan, std::size_t blockIndex, SimdBlock *,
                   std::size_t index) noexcept
{
	const internal::StoreA &op{plan.storeA[index]};
	op.out[blockIndex] = op.a[blockIndex];
}

void executeStoreS(const internal::FusionPlanData &plan, std::size_t blockIndex, SimdBlock *,
                   std::size_t index) noexcept
{
	const internal::StoreS &op{plan.storeS[index]};
	op.out[blockIndex] = op.a;
}

void executeDirectStoresBlock(const internal::FusionPlanData &plan, std::size_t blockIndex) noexcept
{
	for (const internal::StoreFmaAAA &op : plan.storeFmaAAA) {
		op.out[blockIndex] =
		    multiplyAddBlock(op.a[blockIndex], op.b[blockIndex], op.c[blockIndex]);
	}

	for (const internal::StoreFmaASA &op : plan.storeFmaASA) {
		op.out[blockIndex] = multiplyAddBlock(op.a[blockIndex], op.b, op.c[blockIndex]);
	}

	for (const internal::StoreNegFmaAAA &op : plan.storeNegFmaAAA) {
		op.out[blockIndex] =
		    negativeMultiplyAddBlock(op.a[blockIndex], op.b[blockIndex], op.c[blockIndex]);
	}

	for (const internal::StoreNegFmaASA &op : plan.storeNegFmaASA) {
		op.out[blockIndex] =
		    negativeMultiplyAddBlock(op.a[blockIndex], op.b, op.c[blockIndex]);
	}

	for (const internal::StoreBinaryAA &op : plan.storeAddAA) {
		op.out[blockIndex] = addBlock(op.a[blockIndex], op.b[blockIndex]);
	}

	for (const internal::StoreBinaryAS &op : plan.storeAddAS) {
		op.out[blockIndex] = addBlock(op.a[blockIndex], op.b);
	}

	for (const internal::StoreBinaryAA &op : plan.storeSubAA) {
		op.out[blockIndex] = subBlock(op.a[blockIndex], op.b[blockIndex]);
	}

	for (const internal::StoreBinaryAS &op : plan.storeSubAS) {
		op.out[blockIndex] = subBlock(op.a[blockIndex], op.b);
	}

	for (const internal::StoreBinarySA &op : plan.storeSubSA) {
		op.out[blockIndex] = subBlock(op.a, op.b[blockIndex]);
	}

	for (const internal::StoreBinaryAA &op : plan.storeMulAA) {
		op.out[blockIndex] = mulBlock(op.a[blockIndex], op.b[blockIndex]);
	}

	for (const internal::StoreBinaryAS &op : plan.storeMulAS) {
		op.out[blockIndex] = mulBlock(op.a[blockIndex], op.b);
	}

	for (const internal::StoreBinaryAA &op : plan.storeDivAA) {
		op.out[blockIndex] = divBlock(op.a[blockIndex], op.b[blockIndex]);
	}

	for (const internal::StoreBinaryAS &op : plan.storeDivAS) {
		op.out[blockIndex] = divBlock(op.a[blockIndex], op.b);
	}

	for (const internal::StoreBinarySA &op : plan.storeDivSA) {
		op.out[blockIndex] = divBlock(op.a, op.b[blockIndex]);
	}
}

}

internal::FusionPlan::FusionPlan() = default;

internal::FusionPlan::~FusionPlan() noexcept = default;

void internal::FusionPlan::execute() const noexcept
{
	if (data_.directOnly) {
		for (std::size_t blockIndex{}; blockIndex < data_.blockCount; ++blockIndex) {
			executeDirectStoresBlock(data_, blockIndex);
		}

		return;
	}

	SimdBlock regs[MAX_REGISTERS];

	for (std::size_t blockIndex{}; blockIndex < data_.blockCount; ++blockIndex) {
		executeBlock(blockIndex, regs);
	}
}

void internal::FusionPlan::executeBlock(std::size_t blockIndex, SimdBlock *regs) const noexcept
{
	if (data_.directOnly) {
		executeDirectStoresBlock(data_, blockIndex);
		return;
	}

	for (const internal::OpRef &ref : data_.ops) {
		assert(ref.execute != nullptr);
		ref.execute(data_, blockIndex, regs, ref.index);
	}
}

std::size_t internal::FusionPlan::instructionCount() const noexcept
{
	if (data_.directOnly) {
		return data_.directInstructionCount;
	}

	return data_.ops.size();
}

int internal::FusionPlan::registerCount() const noexcept { return data_.maxRegisterCount; }

int internal::FusionPlan::allocateRegister() noexcept
{
	int result{-1};

	if (!data_.freeRegisters.empty()) {
		result = data_.freeRegisters.back();
		data_.freeRegisters.pop_back();
	} else {
		result = data_.nextRegister;
		++data_.nextRegister;
		data_.maxRegisterCount = std::max(data_.maxRegisterCount, data_.nextRegister);
	}

	assert(result < MAX_REGISTERS);
	return result;
}

void internal::FusionPlan::releaseRegister(int reg)
{
	if (reg >= 0) {
		data_.freeRegisters.push_back(reg);
	}
}

internal::ScheduledPlan::ScheduledPlan() = default;

internal::ScheduledPlan::~ScheduledPlan() noexcept = default;

void internal::ScheduledPlan::execute() const noexcept
{
	if (data_.stages.empty())
		return;

	if (data_.stages.size() == 1) {
		data_.stages.front().execute();
		return;
	}

	SimdBlock regs[MAX_REGISTERS];

	for (std::size_t blockIndex{}; blockIndex < data_.blockCount; ++blockIndex) {
		for (const FusionPlan &stage : data_.stages) {
			stage.executeBlock(blockIndex, regs);
		}
	}
}

std::size_t internal::ScheduledPlan::stageCount() const noexcept { return data_.stages.size(); }

std::size_t internal::ScheduledPlan::instructionCount() const noexcept
{
	std::size_t total{};

	for (const FusionPlan &stage : data_.stages) {
		total += stage.instructionCount();
	}

	return total;
}

int internal::ScheduledPlan::maxRegisterCount() const noexcept
{
	int result{};

	for (const FusionPlan &stage : data_.stages) {
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
		engine.impl_->nodeKeyIndex.reserve(newCapacity);
	}

	static void rememberExpressionReserve(Engine &engine) noexcept
	{
		if (engine.impl_->expressionReserveHint < engine.impl_->nodes.size()) {
			engine.impl_->expressionReserveHint = engine.impl_->nodes.size();
		}
	}

	static void reserveAssignmentsFor(Engine &engine, std::size_t additionalCount)
	{
		const std::size_t required{engine.impl_->pendingAssignments.size() +
		                           additionalCount};
		if (required <= engine.impl_->pendingAssignments.capacity()) {
			return;
		}

		std::size_t newCapacity{engine.impl_->pendingAssignments.capacity()};
		if (newCapacity == 0) {
			newCapacity = 4;
		}

		while (newCapacity < required) {
			newCapacity *= 2;
		}

		if (newCapacity < engine.impl_->assignmentReserveHint) {
			newCapacity = engine.impl_->assignmentReserveHint;
		}

		engine.impl_->pendingAssignments.reserve(newCapacity);
	}

	static void rememberAssignmentReserve(Engine &engine) noexcept
	{
		if (engine.impl_->assignmentReserveHint < engine.impl_->pendingAssignments.size()) {
			engine.impl_->assignmentReserveHint =
			    engine.impl_->pendingAssignments.size();
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
		constexpr std::size_t fastLookbackCount{16};
		std::size_t fastBegin{};
		if (nodes.size() > fastLookbackCount) {
			fastBegin = nodes.size() - fastLookbackCount;
		}

		for (std::size_t index{nodes.size()}; index > 0; --index) {
			const std::size_t nodeId{index - 1};
			if (nodeId < fastBegin) {
				break;
			}

			if (sameNode(nodes[nodeId], node)) {
				return Expression{&engine, nodeId};
			}
		}

		std::vector<KeyIndex> &nodeIndex{engine.impl_->nodeKeyIndex};
		const KeyIndex lookup{node.key, 0};
		const auto first{std::lower_bound(
		    nodeIndex.begin(), nodeIndex.end(), lookup,
		    [](const KeyIndex &lhs, const KeyIndex &rhs) {
			    if (lhs.key != rhs.key) {
				    return lhs.key < rhs.key;
			    }

			    return lhs.node < rhs.node;
		    })};

		for (auto it{first}; it != nodeIndex.end() && it->key == node.key; ++it) {
			if (sameNode(nodes[it->node], node)) {
				return Expression{&engine, it->node};
			}
		}

		reserveNodesFor(engine, 1);
		const std::size_t nodeId{engine.impl_->nodes.size()};
		engine.impl_->nodes.push_back(node);
		const KeyIndex insertLookup{node.key, nodeId};
		const auto insertPosition{std::lower_bound(
		    nodeIndex.begin(), nodeIndex.end(), insertLookup,
		    [](const KeyIndex &lhs, const KeyIndex &rhs) {
			    if (lhs.key != rhs.key) {
				    return lhs.key < rhs.key;
			    }

			    return lhs.node < rhs.node;
		    })};
		nodeIndex.insert(insertPosition, insertLookup);
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

	static bool isCompoundAssignment(AssignmentKind kind) noexcept
	{
		return kind != AssignmentKind::Assign;
	}

	static std::uint64_t assignmentKindCode(AssignmentKind kind) noexcept
	{
		if (kind == AssignmentKind::Assign) {
			return 1;
		}

		if (kind == AssignmentKind::AddAssign) {
			return 2;
		}

		if (kind == AssignmentKind::SubAssign) {
			return 3;
		}

		if (kind == AssignmentKind::MulAssign) {
			return 4;
		}

		return 5;
	}

	static NodeKind compoundNodeKind(AssignmentKind kind) noexcept
	{
		if (kind == AssignmentKind::AddAssign) {
			return NodeKind::Add;
		}

		if (kind == AssignmentKind::SubAssign) {
			return NodeKind::Sub;
		}

		if (kind == AssignmentKind::MulAssign) {
			return NodeKind::Mul;
		}

		return NodeKind::Div;
	}

	static void collectAssignmentReads(const Engine &engine, const Assignment &assignment,
	                                   std::vector<const FloatArray *> &reads)
	{
		if (isCompoundAssignment(assignment.kind)) {
			appendUniqueArray(reads, &assignment.out_->variable.array());
		}

		collectReads(engine, assignment.expr_.nodeId(), reads);
	}

	static void appendCacheArray(std::uint64_t &hash, const FloatArray &array) noexcept
	{
		hash = combineHash(hash, reinterpret_cast<std::uintptr_t>(&array),
		                   reinterpret_cast<std::uintptr_t>(array.data()));
		hash = combineHash(hash, array.elementCount(), array.blockCount());
	}

	static bool samePlanCacheKey(const PlanCacheKey &lhs, const PlanCacheKey &rhs) noexcept
	{
		return lhs.hash == rhs.hash && lhs.assignmentCount == rhs.assignmentCount;
	}

	static const FloatArray *findFirstRead(const Engine &engine, std::size_t nodeId) noexcept
	{
		const ExprNode &n{engine.impl_->nodes[nodeId]};

		if (n.kind == NodeKind::Variable) {
			return n.array;
		}

		if (n.lhs != INVALID_NODE) {
			const FloatArray *lhs{findFirstRead(engine, n.lhs)};
			if (lhs != nullptr) {
				return lhs;
			}
		}

		if (n.rhs != INVALID_NODE) {
			return findFirstRead(engine, n.rhs);
		}

		return nullptr;
	}

	static void resizeOutputIfEmpty(const Engine &engine, Variable &output,
	                                const Expression &expr)
	{
		FloatArray &outputArray{output.array()};
		if (outputArray.elementCount() != 0) {
			return;
		}

		const FloatArray *source{findFirstRead(engine, expr.nodeId())};
		if (source != nullptr) {
			outputArray.resize(source->elementCount());
		}
	}

	static const std::vector<internal::AssignmentRange> &
	buildStages(Engine &engine, const std::vector<Assignment> &assignments)
	{
		std::vector<internal::AssignmentRange> &ranges{engine.impl_->stageRanges};
		std::vector<const FloatArray *> &currentWrites{engine.impl_->writeScratch};
		std::vector<const FloatArray *> &reads{engine.impl_->readScratch};

		ranges.clear();
		currentWrites.clear();
		reads.clear();
		reads.reserve(8);
		ranges.reserve(assignments.size());

		std::size_t stageBegin{};
		for (std::size_t i{}; i < assignments.size(); ++i) {
			const Assignment &assignment{assignments[i]};
			reads.clear();
			collectAssignmentReads(engine, assignment, reads);

			const FloatArray *write{&assignment.out_->variable.array()};
			bool newStage{containsArray(currentWrites, write)};

			for (const FloatArray *read : reads) {
				if (containsArray(currentWrites, read)) {
					newStage = true;
					break;
				}
			}

			if (newStage && stageBegin < i) {
				ranges.push_back(internal::AssignmentRange{stageBegin, i});
				currentWrites.clear();
				stageBegin = i;
			}

			appendUniqueArray(currentWrites, write);
		}

		if (stageBegin < assignments.size()) {
			ranges.push_back(internal::AssignmentRange{stageBegin, assignments.size()});
		}

		return ranges;
	}

	static PlanCacheKey
	validateAssignmentsAndBuildCacheKey(Engine &engine,
	                                    const std::vector<Assignment> &assignments)
	{
		PlanCacheKey key{};
		key.assignmentCount = assignments.size();
		key.hash = combineHash(0x706c616e, assignments.size(), 0);

		if (assignments.empty()) {
			return key;
		}

		if (assignments.front().out_ == nullptr) {
			throw std::invalid_argument{"代入先が空です。"};
		}

		requireVarOwner(engine, assignments.front().out_->variable,
		                "代入先が別のEngineに紐づいています。");
		const FloatArray &expected{assignments.front().out_->variable.array()};
		std::vector<const FloatArray *> &reads{engine.impl_->readScratch};
		reads.clear();
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

			const FloatArray &output{assignment.out_->variable.array()};
			const ExprNode &root{engine.impl_->nodes[assignment.expr_.nodeId()]};
			appendCacheArray(key.hash, output);
			key.hash =
			    combineHash(key.hash, root.key, assignmentKindCode(assignment.kind));

			reads.clear();
			collectAssignmentReads(engine, assignment, reads);

			for (const FloatArray *read : reads) {
				requireSameShape(expected, *read,
				                 "式に含まれる配列の要素数が一致しません。");
				appendCacheArray(key.hash, *read);
			}
		}

		return key;
	}

	static CompileContext &makeCompileContext(Engine &engine)
	{
		CompileContext &context{engine.impl_->compileContext};
		const std::vector<ExprNode> &nodes{engine.impl_->nodes};

		context.groupByNode.assign(nodes.size(), INVALID_NODE);
		context.groups.clear();
		if (nodes.empty()) {
			return context;
		}

		const std::vector<KeyIndex> &keys{engine.impl_->nodeKeyIndex};
		context.groups.reserve(nodes.size());

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
		value.scalar = set1Block(n.scalar);
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
		plan.data_.ops.push_back(internal::OpRef{execute, index});
		plan.data_.directOnly = false;
	}

	template <typename Op>
	static void appendDirectStore(FusionPlan &plan, std::vector<Op> &bucket, const Op &op)
	{
		bucket.push_back(op);
		++plan.data_.directInstructionCount;
	}

	static bool findFmaParts(const Engine &engine, std::size_t nodeId, std::size_t &mulId,
	                         std::size_t &addId) noexcept
	{
		const ExprNode &node{engine.impl_->nodes[nodeId]};
		if (node.kind != NodeKind::Add) {
			return false;
		}

		if (engine.impl_->nodes[node.lhs].kind == NodeKind::Mul) {
			mulId = node.lhs;
			addId = node.rhs;
			return true;
		}

		if (engine.impl_->nodes[node.rhs].kind == NodeKind::Mul) {
			mulId = node.rhs;
			addId = node.lhs;
			return true;
		}

		return false;
	}

	static bool makeDirectFmaStore(const Engine &engine, SimdBlock *out, std::size_t nodeId,
	                               PendingStore &store) noexcept
	{
		std::size_t mulId{INVALID_NODE};
		std::size_t addId{INVALID_NODE};
		if (!findFmaParts(engine, nodeId, mulId, addId)) {
			return false;
		}

		const ExprNode &add{engine.impl_->nodes[addId]};
		if (add.kind != NodeKind::Variable) {
			return false;
		}

		const ExprNode &mul{engine.impl_->nodes[mulId]};
		const ExprNode &lhs{engine.impl_->nodes[mul.lhs]};
		const ExprNode &rhs{engine.impl_->nodes[mul.rhs]};

		store = PendingStore{};
		store.out = out;
		store.c = add.array->data();

		if (lhs.kind == NodeKind::Variable && rhs.kind == NodeKind::Variable) {
			store.kind = PendingStoreKind::FmaAAA;
			store.a = lhs.array->data();
			store.bArray = rhs.array->data();
			return true;
		}

		if (lhs.kind == NodeKind::Variable && rhs.kind == NodeKind::Scalar) {
			store.kind = PendingStoreKind::FmaASA;
			store.a = lhs.array->data();
			store.bScalar = set1Block(rhs.scalar);
			return true;
		}

		if (lhs.kind == NodeKind::Scalar && rhs.kind == NodeKind::Variable) {
			store.kind = PendingStoreKind::FmaASA;
			store.a = rhs.array->data();
			store.bScalar = set1Block(lhs.scalar);
			return true;
		}

		return false;
	}

	static bool makeDirectBinaryStore(const Engine &engine, SimdBlock *out, std::size_t nodeId,
	                                  PendingStore &store) noexcept
	{
		const ExprNode &node{engine.impl_->nodes[nodeId]};
		if (node.kind == NodeKind::Variable || node.kind == NodeKind::Scalar) {
			return false;
		}

		const ExprNode &lhs{engine.impl_->nodes[node.lhs]};
		const ExprNode &rhs{engine.impl_->nodes[node.rhs]};
		const bool lhsArray{lhs.kind == NodeKind::Variable};
		const bool rhsArray{rhs.kind == NodeKind::Variable};
		const bool lhsScalar{lhs.kind == NodeKind::Scalar};
		const bool rhsScalar{rhs.kind == NodeKind::Scalar};

		if ((!lhsArray && !lhsScalar) || (!rhsArray && !rhsScalar)) {
			return false;
		}

		store = PendingStore{};
		store.out = out;

		if (node.kind == NodeKind::Add) {
			if (lhsArray && rhsArray) {
				store.kind = PendingStoreKind::AddAA;
				store.a = lhs.array->data();
				store.bArray = rhs.array->data();
				return true;
			}

			if (lhsArray && rhsScalar) {
				store.kind = PendingStoreKind::AddAS;
				store.a = lhs.array->data();
				store.bScalar = set1Block(rhs.scalar);
				return true;
			}

			if (lhsScalar && rhsArray) {
				store.kind = PendingStoreKind::AddAS;
				store.a = rhs.array->data();
				store.bScalar = set1Block(lhs.scalar);
				return true;
			}
		}

		if (node.kind == NodeKind::Sub) {
			if (lhsArray && rhsArray) {
				store.kind = PendingStoreKind::SubAA;
				store.a = lhs.array->data();
				store.bArray = rhs.array->data();
				return true;
			}

			if (lhsArray && rhsScalar) {
				store.kind = PendingStoreKind::SubAS;
				store.a = lhs.array->data();
				store.bScalar = set1Block(rhs.scalar);
				return true;
			}

			if (lhsScalar && rhsArray) {
				store.kind = PendingStoreKind::SubSA;
				store.bScalar = set1Block(lhs.scalar);
				store.bArray = rhs.array->data();
				return true;
			}
		}

		if (node.kind == NodeKind::Mul) {
			if (lhsArray && rhsArray) {
				store.kind = PendingStoreKind::MulAA;
				store.a = lhs.array->data();
				store.bArray = rhs.array->data();
				return true;
			}

			if (lhsArray && rhsScalar) {
				store.kind = PendingStoreKind::MulAS;
				store.a = lhs.array->data();
				store.bScalar = set1Block(rhs.scalar);
				return true;
			}

			if (lhsScalar && rhsArray) {
				store.kind = PendingStoreKind::MulAS;
				store.a = rhs.array->data();
				store.bScalar = set1Block(lhs.scalar);
				return true;
			}
		}

		if (node.kind == NodeKind::Div) {
			if (lhsArray && rhsArray) {
				store.kind = PendingStoreKind::DivAA;
				store.a = lhs.array->data();
				store.bArray = rhs.array->data();
				return true;
			}

			if (lhsArray && rhsScalar) {
				store.kind = PendingStoreKind::DivAS;
				store.a = lhs.array->data();
				store.bScalar = set1Block(rhs.scalar);
				return true;
			}

			if (lhsScalar && rhsArray) {
				store.kind = PendingStoreKind::DivSA;
				store.bScalar = set1Block(lhs.scalar);
				store.bArray = rhs.array->data();
				return true;
			}
		}

		return false;
	}

	static bool makeDirectStore(const Engine &engine, SimdBlock *out, std::size_t nodeId,
	                            PendingStore &store) noexcept
	{
		if (makeDirectFmaStore(engine, out, nodeId, store)) {
			return true;
		}

		return makeDirectBinaryStore(engine, out, nodeId, store);
	}

	static bool makeDirectAddAssignFmaStore(const Engine &engine, SimdBlock *out,
	                                        const FloatArray &outputArray, std::size_t nodeId,
	                                        PendingStore &store) noexcept
	{
		const ExprNode &node{engine.impl_->nodes[nodeId]};
		if (node.kind != NodeKind::Mul) {
			return false;
		}

		const ExprNode &lhs{engine.impl_->nodes[node.lhs]};
		const ExprNode &rhs{engine.impl_->nodes[node.rhs]};

		store = PendingStore{};
		store.out = out;
		store.c = outputArray.data();

		if (lhs.kind == NodeKind::Variable && rhs.kind == NodeKind::Variable) {
			store.kind = PendingStoreKind::FmaAAA;
			store.a = lhs.array->data();
			store.bArray = rhs.array->data();
			return true;
		}

		if (lhs.kind == NodeKind::Variable && rhs.kind == NodeKind::Scalar) {
			store.kind = PendingStoreKind::FmaASA;
			store.a = lhs.array->data();
			store.bScalar = set1Block(rhs.scalar);
			return true;
		}

		if (lhs.kind == NodeKind::Scalar && rhs.kind == NodeKind::Variable) {
			store.kind = PendingStoreKind::FmaASA;
			store.a = rhs.array->data();
			store.bScalar = set1Block(lhs.scalar);
			return true;
		}

		return false;
	}

	static bool makeDirectSubAssignFmaStore(const Engine &engine, SimdBlock *out,
	                                        const FloatArray &outputArray, std::size_t nodeId,
	                                        PendingStore &store) noexcept
	{
		const ExprNode &node{engine.impl_->nodes[nodeId]};
		if (node.kind != NodeKind::Mul) {
			return false;
		}

		const ExprNode &lhs{engine.impl_->nodes[node.lhs]};
		const ExprNode &rhs{engine.impl_->nodes[node.rhs]};

		store = PendingStore{};
		store.out = out;
		store.c = outputArray.data();

		if (lhs.kind == NodeKind::Variable && rhs.kind == NodeKind::Variable) {
			store.kind = PendingStoreKind::NegFmaAAA;
			store.a = lhs.array->data();
			store.bArray = rhs.array->data();
			return true;
		}

		if (lhs.kind == NodeKind::Variable && rhs.kind == NodeKind::Scalar) {
			store.kind = PendingStoreKind::NegFmaASA;
			store.a = lhs.array->data();
			store.bScalar = set1Block(rhs.scalar);
			return true;
		}

		if (lhs.kind == NodeKind::Scalar && rhs.kind == NodeKind::Variable) {
			store.kind = PendingStoreKind::NegFmaASA;
			store.a = rhs.array->data();
			store.bScalar = set1Block(lhs.scalar);
			return true;
		}

		return false;
	}

	static bool makeDirectCompoundBinaryStore(const Engine &engine, SimdBlock *out,
	                                          const FloatArray &outputArray,
	                                          const Assignment &assignment,
	                                          PendingStore &store) noexcept
	{
		const ExprNode &rhs{engine.impl_->nodes[assignment.expr_.nodeId()]};
		const bool rhsArray{rhs.kind == NodeKind::Variable};
		const bool rhsScalar{rhs.kind == NodeKind::Scalar};

		if (!rhsArray && !rhsScalar) {
			return false;
		}

		store = PendingStore{};
		store.out = out;
		store.a = outputArray.data();

		if (rhsArray) {
			store.bArray = rhs.array->data();
		} else {
			store.bScalar = set1Block(rhs.scalar);
		}

		if (assignment.kind == AssignmentKind::AddAssign) {
			if (rhsArray) {
				store.kind = PendingStoreKind::AddAA;
			} else {
				store.kind = PendingStoreKind::AddAS;
			}
			return true;
		}

		if (assignment.kind == AssignmentKind::SubAssign) {
			if (rhsArray) {
				store.kind = PendingStoreKind::SubAA;
			} else {
				store.kind = PendingStoreKind::SubAS;
			}
			return true;
		}

		if (assignment.kind == AssignmentKind::MulAssign) {
			if (rhsArray) {
				store.kind = PendingStoreKind::MulAA;
			} else {
				store.kind = PendingStoreKind::MulAS;
			}
			return true;
		}

		if (assignment.kind == AssignmentKind::DivAssign) {
			if (rhsArray) {
				store.kind = PendingStoreKind::DivAA;
			} else {
				store.kind = PendingStoreKind::DivAS;
			}
			return true;
		}

		return false;
	}

	static bool makeDirectAssignmentStore(const Engine &engine, const Assignment &assignment,
	                                      PendingStore &store) noexcept
	{
		FloatArray &outputArray{assignment.out_->variable.array()};
		SimdBlock *out{outputArray.data()};

		if (assignment.kind == AssignmentKind::Assign) {
			return makeDirectStore(engine, out, assignment.expr_.nodeId(), store);
		}

		if (assignment.kind == AssignmentKind::AddAssign &&
		    makeDirectAddAssignFmaStore(engine, out, outputArray, assignment.expr_.nodeId(),
		                                store)) {
			return true;
		}

		if (assignment.kind == AssignmentKind::SubAssign &&
		    makeDirectSubAssignFmaStore(engine, out, outputArray, assignment.expr_.nodeId(),
		                                store)) {
			return true;
		}

		return makeDirectCompoundBinaryStore(engine, out, outputArray, assignment, store);
	}

	static void emitStore(FusionPlan &plan, SimdBlock *out, const ValueRef &value,
	                      CompileContext &context)
	{
		if (value.kind == ValueKind::Reg) {
			appendOp(plan, plan.data_.storeR, executeStoreR,
			         internal::StoreR{out, value.reg});
		} else if (value.kind == ValueKind::Array) {
			appendOp(plan, plan.data_.storeA, executeStoreA,
			         internal::StoreA{out, value.array});
		} else {
			appendOp(plan, plan.data_.storeS, executeStoreS,
			         internal::StoreS{out, value.scalar});
		}

		releaseIfLastUse(plan, value, context);
		if (value.kind == ValueKind::Reg && !value.hasGroup) {
			plan.releaseRegister(value.reg);
		}
	}

	static void emitPendingStore(FusionPlan &plan, const PendingStore &store,
	                             CompileContext &context)
	{
		if (store.kind == PendingStoreKind::FmaAAA) {
			appendOp(plan, plan.data_.storeFmaAAA, executeStoreFmaAAA,
			         internal::StoreFmaAAA{store.out, store.a, store.bArray, store.c});
			return;
		}

		if (store.kind == PendingStoreKind::FmaASA) {
			appendOp(plan, plan.data_.storeFmaASA, executeStoreFmaASA,
			         internal::StoreFmaASA{store.out, store.a, store.bScalar, store.c});
			return;
		}

		if (store.kind == PendingStoreKind::NegFmaAAA) {
			appendOp(
			    plan, plan.data_.storeNegFmaAAA, executeStoreNegFmaAAA,
			    internal::StoreNegFmaAAA{store.out, store.a, store.bArray, store.c});
			return;
		}

		if (store.kind == PendingStoreKind::NegFmaASA) {
			appendOp(
			    plan, plan.data_.storeNegFmaASA, executeStoreNegFmaASA,
			    internal::StoreNegFmaASA{store.out, store.a, store.bScalar, store.c});
			return;
		}

		if (store.kind == PendingStoreKind::AddAA) {
			appendOp(plan, plan.data_.storeAddAA, executeStoreAddAA,
			         internal::StoreBinaryAA{store.out, store.a, store.bArray});
			return;
		}

		if (store.kind == PendingStoreKind::AddAS) {
			appendOp(plan, plan.data_.storeAddAS, executeStoreAddAS,
			         internal::StoreBinaryAS{store.out, store.a, store.bScalar});
			return;
		}

		if (store.kind == PendingStoreKind::SubAA) {
			appendOp(plan, plan.data_.storeSubAA, executeStoreSubAA,
			         internal::StoreBinaryAA{store.out, store.a, store.bArray});
			return;
		}

		if (store.kind == PendingStoreKind::SubAS) {
			appendOp(plan, plan.data_.storeSubAS, executeStoreSubAS,
			         internal::StoreBinaryAS{store.out, store.a, store.bScalar});
			return;
		}

		if (store.kind == PendingStoreKind::SubSA) {
			appendOp(plan, plan.data_.storeSubSA, executeStoreSubSA,
			         internal::StoreBinarySA{store.out, store.bScalar, store.bArray});
			return;
		}

		if (store.kind == PendingStoreKind::MulAA) {
			appendOp(plan, plan.data_.storeMulAA, executeStoreMulAA,
			         internal::StoreBinaryAA{store.out, store.a, store.bArray});
			return;
		}

		if (store.kind == PendingStoreKind::MulAS) {
			appendOp(plan, plan.data_.storeMulAS, executeStoreMulAS,
			         internal::StoreBinaryAS{store.out, store.a, store.bScalar});
			return;
		}

		if (store.kind == PendingStoreKind::DivAA) {
			appendOp(plan, plan.data_.storeDivAA, executeStoreDivAA,
			         internal::StoreBinaryAA{store.out, store.a, store.bArray});
			return;
		}

		if (store.kind == PendingStoreKind::DivAS) {
			appendOp(plan, plan.data_.storeDivAS, executeStoreDivAS,
			         internal::StoreBinaryAS{store.out, store.a, store.bScalar});
			return;
		}

		if (store.kind == PendingStoreKind::DivSA) {
			appendOp(plan, plan.data_.storeDivSA, executeStoreDivSA,
			         internal::StoreBinarySA{store.out, store.bScalar, store.bArray});
			return;
		}

		emitStore(plan, store.out, store.value, context);
	}

	static void emitDirectPendingStore(FusionPlan &plan, const PendingStore &store)
	{
		if (store.kind == PendingStoreKind::FmaAAA) {
			appendDirectStore(
			    plan, plan.data_.storeFmaAAA,
			    internal::StoreFmaAAA{store.out, store.a, store.bArray, store.c});
			return;
		}

		if (store.kind == PendingStoreKind::FmaASA) {
			appendDirectStore(
			    plan, plan.data_.storeFmaASA,
			    internal::StoreFmaASA{store.out, store.a, store.bScalar, store.c});
			return;
		}

		if (store.kind == PendingStoreKind::NegFmaAAA) {
			appendDirectStore(
			    plan, plan.data_.storeNegFmaAAA,
			    internal::StoreNegFmaAAA{store.out, store.a, store.bArray, store.c});
			return;
		}

		if (store.kind == PendingStoreKind::NegFmaASA) {
			appendDirectStore(
			    plan, plan.data_.storeNegFmaASA,
			    internal::StoreNegFmaASA{store.out, store.a, store.bScalar, store.c});
			return;
		}

		if (store.kind == PendingStoreKind::AddAA) {
			appendDirectStore(
			    plan, plan.data_.storeAddAA,
			    internal::StoreBinaryAA{store.out, store.a, store.bArray});
			return;
		}

		if (store.kind == PendingStoreKind::AddAS) {
			appendDirectStore(
			    plan, plan.data_.storeAddAS,
			    internal::StoreBinaryAS{store.out, store.a, store.bScalar});
			return;
		}

		if (store.kind == PendingStoreKind::SubAA) {
			appendDirectStore(
			    plan, plan.data_.storeSubAA,
			    internal::StoreBinaryAA{store.out, store.a, store.bArray});
			return;
		}

		if (store.kind == PendingStoreKind::SubAS) {
			appendDirectStore(
			    plan, plan.data_.storeSubAS,
			    internal::StoreBinaryAS{store.out, store.a, store.bScalar});
			return;
		}

		if (store.kind == PendingStoreKind::SubSA) {
			appendDirectStore(
			    plan, plan.data_.storeSubSA,
			    internal::StoreBinarySA{store.out, store.bScalar, store.bArray});
			return;
		}

		if (store.kind == PendingStoreKind::MulAA) {
			appendDirectStore(
			    plan, plan.data_.storeMulAA,
			    internal::StoreBinaryAA{store.out, store.a, store.bArray});
			return;
		}

		if (store.kind == PendingStoreKind::MulAS) {
			appendDirectStore(
			    plan, plan.data_.storeMulAS,
			    internal::StoreBinaryAS{store.out, store.a, store.bScalar});
			return;
		}

		if (store.kind == PendingStoreKind::DivAA) {
			appendDirectStore(
			    plan, plan.data_.storeDivAA,
			    internal::StoreBinaryAA{store.out, store.a, store.bArray});
			return;
		}

		if (store.kind == PendingStoreKind::DivAS) {
			appendDirectStore(
			    plan, plan.data_.storeDivAS,
			    internal::StoreBinaryAS{store.out, store.a, store.bScalar});
			return;
		}

		if (store.kind == PendingStoreKind::DivSA) {
			appendDirectStore(
			    plan, plan.data_.storeDivSA,
			    internal::StoreBinarySA{store.out, store.bScalar, store.bArray});
		}
	}

	static void emitAdd(FusionPlan &plan, int dst, const ValueRef &a, const ValueRef &b)
	{
		if (a.kind == ValueKind::Reg && b.kind == ValueKind::Reg)
			appendOp(plan, plan.data_.addRR, executeAddRR,
			         internal::AddRR{dst, a.reg, b.reg});
		else if (a.kind == ValueKind::Reg && b.kind == ValueKind::Array)
			appendOp(plan, plan.data_.addRA, executeAddRA,
			         internal::AddRA{dst, a.reg, b.array});
		else if (a.kind == ValueKind::Array && b.kind == ValueKind::Reg)
			appendOp(plan, plan.data_.addAR, executeAddAR,
			         internal::AddAR{dst, a.array, b.reg});
		else if (a.kind == ValueKind::Array && b.kind == ValueKind::Array)
			appendOp(plan, plan.data_.addAA, executeAddAA,
			         internal::AddAA{dst, a.array, b.array});
		else if (a.kind == ValueKind::Reg && b.kind == ValueKind::Scalar)
			appendOp(plan, plan.data_.addRS, executeAddRS,
			         internal::AddRS{dst, a.reg, b.scalar});
		else if (a.kind == ValueKind::Array && b.kind == ValueKind::Scalar)
			appendOp(plan, plan.data_.addAS, executeAddAS,
			         internal::AddAS{dst, a.array, b.scalar});
		else if (a.kind == ValueKind::Scalar && b.kind == ValueKind::Reg)
			appendOp(plan, plan.data_.addRS, executeAddRS,
			         internal::AddRS{dst, b.reg, a.scalar});
		else if (a.kind == ValueKind::Scalar && b.kind == ValueKind::Array)
			appendOp(plan, plan.data_.addAS, executeAddAS,
			         internal::AddAS{dst, b.array, a.scalar});
		else
			appendOp(plan, plan.data_.setS, executeSetS,
			         internal::SetS{dst, addBlock(a.scalar, b.scalar)});
	}

	static void emitSub(FusionPlan &plan, int dst, const ValueRef &a, const ValueRef &b)
	{
		if (a.kind == ValueKind::Reg && b.kind == ValueKind::Reg)
			appendOp(plan, plan.data_.subRR, executeSubRR,
			         internal::SubRR{dst, a.reg, b.reg});
		else if (a.kind == ValueKind::Reg && b.kind == ValueKind::Array)
			appendOp(plan, plan.data_.subRA, executeSubRA,
			         internal::SubRA{dst, a.reg, b.array});
		else if (a.kind == ValueKind::Array && b.kind == ValueKind::Reg)
			appendOp(plan, plan.data_.subAR, executeSubAR,
			         internal::SubAR{dst, a.array, b.reg});
		else if (a.kind == ValueKind::Array && b.kind == ValueKind::Array)
			appendOp(plan, plan.data_.subAA, executeSubAA,
			         internal::SubAA{dst, a.array, b.array});
		else if (a.kind == ValueKind::Reg && b.kind == ValueKind::Scalar)
			appendOp(plan, plan.data_.subRS, executeSubRS,
			         internal::SubRS{dst, a.reg, b.scalar});
		else if (a.kind == ValueKind::Array && b.kind == ValueKind::Scalar)
			appendOp(plan, plan.data_.subAS, executeSubAS,
			         internal::SubAS{dst, a.array, b.scalar});
		else if (a.kind == ValueKind::Scalar && b.kind == ValueKind::Reg)
			appendOp(plan, plan.data_.subSR, executeSubSR,
			         internal::SubSR{dst, a.scalar, b.reg});
		else if (a.kind == ValueKind::Scalar && b.kind == ValueKind::Array)
			appendOp(plan, plan.data_.subSA, executeSubSA,
			         internal::SubSA{dst, a.scalar, b.array});
		else
			appendOp(plan, plan.data_.setS, executeSetS,
			         internal::SetS{dst, subBlock(a.scalar, b.scalar)});
	}

	static void emitMul(FusionPlan &plan, int dst, const ValueRef &a, const ValueRef &b)
	{
		if (a.kind == ValueKind::Reg && b.kind == ValueKind::Reg)
			appendOp(plan, plan.data_.mulRR, executeMulRR,
			         internal::MulRR{dst, a.reg, b.reg});
		else if (a.kind == ValueKind::Reg && b.kind == ValueKind::Array)
			appendOp(plan, plan.data_.mulRA, executeMulRA,
			         internal::MulRA{dst, a.reg, b.array});
		else if (a.kind == ValueKind::Array && b.kind == ValueKind::Reg)
			appendOp(plan, plan.data_.mulAR, executeMulAR,
			         internal::MulAR{dst, a.array, b.reg});
		else if (a.kind == ValueKind::Array && b.kind == ValueKind::Array)
			appendOp(plan, plan.data_.mulAA, executeMulAA,
			         internal::MulAA{dst, a.array, b.array});
		else if (a.kind == ValueKind::Reg && b.kind == ValueKind::Scalar)
			appendOp(plan, plan.data_.mulRS, executeMulRS,
			         internal::MulRS{dst, a.reg, b.scalar});
		else if (a.kind == ValueKind::Array && b.kind == ValueKind::Scalar)
			appendOp(plan, plan.data_.mulAS, executeMulAS,
			         internal::MulAS{dst, a.array, b.scalar});
		else if (a.kind == ValueKind::Scalar && b.kind == ValueKind::Reg)
			appendOp(plan, plan.data_.mulRS, executeMulRS,
			         internal::MulRS{dst, b.reg, a.scalar});
		else if (a.kind == ValueKind::Scalar && b.kind == ValueKind::Array)
			appendOp(plan, plan.data_.mulAS, executeMulAS,
			         internal::MulAS{dst, b.array, a.scalar});
		else
			appendOp(plan, plan.data_.setS, executeSetS,
			         internal::SetS{dst, mulBlock(a.scalar, b.scalar)});
	}

	static void emitDiv(FusionPlan &plan, int dst, const ValueRef &a, const ValueRef &b)
	{
		if (a.kind == ValueKind::Reg && b.kind == ValueKind::Reg)
			appendOp(plan, plan.data_.divRR, executeDivRR,
			         internal::DivRR{dst, a.reg, b.reg});
		else if (a.kind == ValueKind::Reg && b.kind == ValueKind::Array)
			appendOp(plan, plan.data_.divRA, executeDivRA,
			         internal::DivRA{dst, a.reg, b.array});
		else if (a.kind == ValueKind::Array && b.kind == ValueKind::Reg)
			appendOp(plan, plan.data_.divAR, executeDivAR,
			         internal::DivAR{dst, a.array, b.reg});
		else if (a.kind == ValueKind::Array && b.kind == ValueKind::Array)
			appendOp(plan, plan.data_.divAA, executeDivAA,
			         internal::DivAA{dst, a.array, b.array});
		else if (a.kind == ValueKind::Reg && b.kind == ValueKind::Scalar)
			appendOp(plan, plan.data_.divRS, executeDivRS,
			         internal::DivRS{dst, a.reg, b.scalar});
		else if (a.kind == ValueKind::Array && b.kind == ValueKind::Scalar)
			appendOp(plan, plan.data_.divAS, executeDivAS,
			         internal::DivAS{dst, a.array, b.scalar});
		else if (a.kind == ValueKind::Scalar && b.kind == ValueKind::Reg)
			appendOp(plan, plan.data_.divSR, executeDivSR,
			         internal::DivSR{dst, a.scalar, b.reg});
		else if (a.kind == ValueKind::Scalar && b.kind == ValueKind::Array)
			appendOp(plan, plan.data_.divSA, executeDivSA,
			         internal::DivSA{dst, a.scalar, b.array});
		else
			appendOp(plan, plan.data_.setS, executeSetS,
			         internal::SetS{dst, divBlock(a.scalar, b.scalar)});
	}

	static void emitFmaOrFallback(FusionPlan &plan, int dst, const ValueRef &a,
	                              const ValueRef &b, const ValueRef &c)
	{
		if (a.kind == ValueKind::Reg && b.kind == ValueKind::Reg &&
		    c.kind == ValueKind::Reg)
			appendOp(plan, plan.data_.fmaRRR, executeFmaRRR,
			         internal::FmaRRR{dst, a.reg, b.reg, c.reg});
		else if (a.kind == ValueKind::Reg && b.kind == ValueKind::Reg &&
		         c.kind == ValueKind::Array)
			appendOp(plan, plan.data_.fmaRRA, executeFmaRRA,
			         internal::FmaRRA{dst, a.reg, b.reg, c.array});
		else if (a.kind == ValueKind::Reg && b.kind == ValueKind::Array &&
		         c.kind == ValueKind::Array)
			appendOp(plan, plan.data_.fmaRAA, executeFmaRAA,
			         internal::FmaRAA{dst, a.reg, b.array, c.array});
		else if (a.kind == ValueKind::Array && b.kind == ValueKind::Array &&
		         c.kind == ValueKind::Array)
			appendOp(plan, plan.data_.fmaAAA, executeFmaAAA,
			         internal::FmaAAA{dst, a.array, b.array, c.array});
		else if (a.kind == ValueKind::Array && b.kind == ValueKind::Scalar &&
		         c.kind == ValueKind::Array)
			appendOp(plan, plan.data_.fmaASA, executeFmaASA,
			         internal::FmaASA{dst, a.array, b.scalar, c.array});
		else if (a.kind == ValueKind::Scalar && b.kind == ValueKind::Array &&
		         c.kind == ValueKind::Array)
			appendOp(plan, plan.data_.fmaASA, executeFmaASA,
			         internal::FmaASA{dst, b.array, a.scalar, c.array});
		else if (a.kind == ValueKind::Reg && b.kind == ValueKind::Array &&
		         c.kind == ValueKind::Scalar)
			appendOp(plan, plan.data_.fmaRAS, executeFmaRAS,
			         internal::FmaRAS{dst, a.reg, b.array, c.scalar});
		else if (a.kind == ValueKind::Reg && b.kind == ValueKind::Scalar &&
		         c.kind == ValueKind::Array)
			appendOp(plan, plan.data_.fmaRSA, executeFmaRSA,
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

	static bool emitFmaStoreOrFallback(FusionPlan &plan, SimdBlock *out, const ValueRef &a,
	                                   const ValueRef &b, const ValueRef &c)
	{
		if (a.kind == ValueKind::Reg && b.kind == ValueKind::Reg &&
		    c.kind == ValueKind::Reg) {
			appendOp(plan, plan.data_.storeFmaRRR, executeStoreFmaRRR,
			         internal::StoreFmaRRR{out, a.reg, b.reg, c.reg});
			return true;
		}

		if (a.kind == ValueKind::Reg && b.kind == ValueKind::Reg &&
		    c.kind == ValueKind::Array) {
			appendOp(plan, plan.data_.storeFmaRRA, executeStoreFmaRRA,
			         internal::StoreFmaRRA{out, a.reg, b.reg, c.array});
			return true;
		}

		if (a.kind == ValueKind::Reg && b.kind == ValueKind::Array &&
		    c.kind == ValueKind::Array) {
			appendOp(plan, plan.data_.storeFmaRAA, executeStoreFmaRAA,
			         internal::StoreFmaRAA{out, a.reg, b.array, c.array});
			return true;
		}

		if (a.kind == ValueKind::Array && b.kind == ValueKind::Array &&
		    c.kind == ValueKind::Array) {
			appendOp(plan, plan.data_.storeFmaAAA, executeStoreFmaAAA,
			         internal::StoreFmaAAA{out, a.array, b.array, c.array});
			return true;
		}

		if (a.kind == ValueKind::Array && b.kind == ValueKind::Scalar &&
		    c.kind == ValueKind::Array) {
			appendOp(plan, plan.data_.storeFmaASA, executeStoreFmaASA,
			         internal::StoreFmaASA{out, a.array, b.scalar, c.array});
			return true;
		}

		if (a.kind == ValueKind::Scalar && b.kind == ValueKind::Array &&
		    c.kind == ValueKind::Array) {
			appendOp(plan, plan.data_.storeFmaASA, executeStoreFmaASA,
			         internal::StoreFmaASA{out, b.array, a.scalar, c.array});
			return true;
		}

		if (a.kind == ValueKind::Reg && b.kind == ValueKind::Array &&
		    c.kind == ValueKind::Scalar) {
			appendOp(plan, plan.data_.storeFmaRAS, executeStoreFmaRAS,
			         internal::StoreFmaRAS{out, a.reg, b.array, c.scalar});
			return true;
		}

		if (a.kind == ValueKind::Reg && b.kind == ValueKind::Scalar &&
		    c.kind == ValueKind::Array) {
			appendOp(plan, plan.data_.storeFmaRSA, executeStoreFmaRSA,
			         internal::StoreFmaRSA{out, a.reg, b.scalar, c.array});
			return true;
		}

		return false;
	}

	static bool emitRootFmaStore(const Engine &engine, FusionPlan &plan,
	                             const Assignment &assignment, CompileContext &context)
	{
		if (assignment.kind != AssignmentKind::Assign) {
			return false;
		}

		const std::size_t nodeId{assignment.expr_.nodeId()};
		if (!canUseFma(engine, nodeId, context)) {
			return false;
		}

		const std::size_t rootGroup{context.groupByNode[nodeId]};
		if (rootGroup >= context.groups.size() || context.groups[rootGroup].useCount != 1) {
			return false;
		}

		const ExprNode &node{engine.impl_->nodes[nodeId]};
		std::size_t mulId{INVALID_NODE};
		std::size_t addId{INVALID_NODE};
		if (engine.impl_->nodes[node.lhs].kind == NodeKind::Mul) {
			mulId = node.lhs;
			addId = node.rhs;
		} else {
			mulId = node.rhs;
			addId = node.lhs;
		}

		const ExprNode &mul{engine.impl_->nodes[mulId]};
		const ValueRef a{compileNode(engine, plan, mul.lhs, context)};
		const ValueRef b{compileNode(engine, plan, mul.rhs, context)};
		const ValueRef c{compileNode(engine, plan, addId, context)};
		SimdBlock *out{assignment.out_->variable.array().data()};

		const bool emitted{emitFmaStoreOrFallback(plan, out, a, b, c)};
		if (!emitted) {
			const int dst{plan.allocateRegister()};
			emitFmaOrFallback(plan, dst, a, b, c);
			emitStore(plan, out, makeTemporaryRegValue(dst), context);
		}

		releaseIfLastUse(plan, a, context);
		releaseIfLastUse(plan, b, context);
		releaseIfLastUse(plan, c, context);

		--context.groups[rootGroup].useCount;
		return true;
	}

	static ValueRef makeTemporaryRegValue(int reg) noexcept
	{
		ValueRef value{};
		value.kind = ValueKind::Reg;
		value.reg = reg;
		return value;
	}

	static ValueRef makeOutputArrayValue(const Assignment &assignment) noexcept
	{
		ValueRef value{};
		value.kind = ValueKind::Array;
		value.array = assignment.out_->variable.array().data();
		return value;
	}

	static ValueRef compileAssignmentValue(const Engine &engine, FusionPlan &plan,
	                                       const Assignment &assignment,
	                                       CompileContext &context)
	{
		if (assignment.kind == AssignmentKind::Assign) {
			return compileNode(engine, plan, assignment.expr_.nodeId(), context);
		}

		const ValueRef lhs{makeOutputArrayValue(assignment)};
		const ValueRef rhs{compileNode(engine, plan, assignment.expr_.nodeId(), context)};
		const int dst{plan.allocateRegister()};
		const NodeKind operation{compoundNodeKind(assignment.kind)};

		if (operation == NodeKind::Add) {
			emitAdd(plan, dst, lhs, rhs);
		} else if (operation == NodeKind::Sub) {
			emitSub(plan, dst, lhs, rhs);
		} else if (operation == NodeKind::Mul) {
			emitMul(plan, dst, lhs, rhs);
		} else {
			emitDiv(plan, dst, lhs, rhs);
		}

		releaseIfLastUse(plan, rhs, context);
		return makeTemporaryRegValue(dst);
	}

	struct PlanReserveEstimate {
		std::size_t instructionCount{};
		std::size_t storeR{};
		std::size_t storeA{};
		std::size_t storeS{};
		std::size_t storeFmaAAA{};
		std::size_t storeFmaASA{};
		std::size_t storeFmaRRR{};
		std::size_t storeFmaRRA{};
		std::size_t storeFmaRAA{};
		std::size_t storeFmaRAS{};
		std::size_t storeFmaRSA{};
		std::size_t storeNegFmaAAA{};
		std::size_t storeNegFmaASA{};
		std::size_t storeAddAA{};
		std::size_t storeAddAS{};
		std::size_t storeSubAA{};
		std::size_t storeSubAS{};
		std::size_t storeSubSA{};
		std::size_t storeMulAA{};
		std::size_t storeMulAS{};
		std::size_t storeDivAA{};
		std::size_t storeDivAS{};
		std::size_t storeDivSA{};
	};

	static std::size_t estimateNodeInstructionCount(const Engine &engine,
	                                                std::size_t nodeId) noexcept
	{
		const ExprNode &node{engine.impl_->nodes[nodeId]};
		if (node.kind == NodeKind::Variable || node.kind == NodeKind::Scalar) {
			return 0;
		}

		std::size_t mulId{INVALID_NODE};
		std::size_t addId{INVALID_NODE};
		if (findFmaParts(engine, nodeId, mulId, addId)) {
			const ExprNode &mul{engine.impl_->nodes[mulId]};
			return 1 + estimateNodeInstructionCount(engine, mul.lhs) +
			       estimateNodeInstructionCount(engine, mul.rhs) +
			       estimateNodeInstructionCount(engine, addId);
		}

		return 1 + estimateNodeInstructionCount(engine, node.lhs) +
		       estimateNodeInstructionCount(engine, node.rhs);
	}

	static std::size_t assignmentCount(internal::AssignmentRange range) noexcept
	{
		return range.end - range.begin;
	}

	static void countDirectStoreReserve(PlanReserveEstimate &estimate,
	                                    PendingStoreKind kind) noexcept
	{
		++estimate.instructionCount;

		if (kind == PendingStoreKind::FmaAAA) {
			++estimate.storeFmaAAA;
		} else if (kind == PendingStoreKind::FmaASA) {
			++estimate.storeFmaASA;
		} else if (kind == PendingStoreKind::NegFmaAAA) {
			++estimate.storeNegFmaAAA;
		} else if (kind == PendingStoreKind::NegFmaASA) {
			++estimate.storeNegFmaASA;
		} else if (kind == PendingStoreKind::AddAA) {
			++estimate.storeAddAA;
		} else if (kind == PendingStoreKind::AddAS) {
			++estimate.storeAddAS;
		} else if (kind == PendingStoreKind::SubAA) {
			++estimate.storeSubAA;
		} else if (kind == PendingStoreKind::SubAS) {
			++estimate.storeSubAS;
		} else if (kind == PendingStoreKind::SubSA) {
			++estimate.storeSubSA;
		} else if (kind == PendingStoreKind::MulAA) {
			++estimate.storeMulAA;
		} else if (kind == PendingStoreKind::MulAS) {
			++estimate.storeMulAS;
		} else if (kind == PendingStoreKind::DivAA) {
			++estimate.storeDivAA;
		} else if (kind == PendingStoreKind::DivAS) {
			++estimate.storeDivAS;
		} else if (kind == PendingStoreKind::DivSA) {
			++estimate.storeDivSA;
		}
	}

	static PlanReserveEstimate estimatePlanReserve(const Engine &engine,
	                                               const std::vector<Assignment> &assignments,
	                                               internal::AssignmentRange range) noexcept
	{
		PlanReserveEstimate estimate{};

		for (std::size_t i{range.begin}; i < range.end; ++i) {
			const Assignment &assignment{assignments[i]};
			PendingStore store{};
			if (makeDirectAssignmentStore(engine, assignment, store)) {
				countDirectStoreReserve(estimate, store.kind);
				continue;
			}

			const ExprNode &root{engine.impl_->nodes[assignment.expr_.nodeId()]};
			estimate.instructionCount +=
			    estimateNodeInstructionCount(engine, assignment.expr_.nodeId()) + 1;

			if (isCompoundAssignment(assignment.kind)) {
				++estimate.instructionCount;
				++estimate.storeR;
			} else if (root.kind == NodeKind::Add) {
				std::size_t mulId{INVALID_NODE};
				std::size_t addId{INVALID_NODE};
				if (findFmaParts(engine, assignment.expr_.nodeId(), mulId, addId)) {
					++estimate.storeFmaRRR;
				} else {
					++estimate.storeR;
				}
			} else if (root.kind == NodeKind::Variable) {
				++estimate.storeA;
			} else if (root.kind == NodeKind::Scalar) {
				++estimate.storeS;
			} else {
				++estimate.storeR;
			}
		}

		return estimate;
	}

	static void reservePlanStorage(FusionPlan &plan, const PlanReserveEstimate &estimate)
	{
		plan.data_.ops.reserve(estimate.instructionCount);
		plan.data_.storeR.reserve(estimate.storeR);
		plan.data_.storeA.reserve(estimate.storeA);
		plan.data_.storeS.reserve(estimate.storeS);
		plan.data_.storeFmaAAA.reserve(estimate.storeFmaAAA);
		plan.data_.storeFmaASA.reserve(estimate.storeFmaASA);
		plan.data_.storeFmaRRR.reserve(estimate.storeFmaRRR);
		plan.data_.storeFmaRRA.reserve(estimate.storeFmaRRA);
		plan.data_.storeFmaRAA.reserve(estimate.storeFmaRAA);
		plan.data_.storeFmaRAS.reserve(estimate.storeFmaRAS);
		plan.data_.storeFmaRSA.reserve(estimate.storeFmaRSA);
		plan.data_.storeNegFmaAAA.reserve(estimate.storeNegFmaAAA);
		plan.data_.storeNegFmaASA.reserve(estimate.storeNegFmaASA);
		plan.data_.storeAddAA.reserve(estimate.storeAddAA);
		plan.data_.storeAddAS.reserve(estimate.storeAddAS);
		plan.data_.storeSubAA.reserve(estimate.storeSubAA);
		plan.data_.storeSubAS.reserve(estimate.storeSubAS);
		plan.data_.storeSubSA.reserve(estimate.storeSubSA);
		plan.data_.storeMulAA.reserve(estimate.storeMulAA);
		plan.data_.storeMulAS.reserve(estimate.storeMulAS);
		plan.data_.storeDivAA.reserve(estimate.storeDivAA);
		plan.data_.storeDivAS.reserve(estimate.storeDivAS);
		plan.data_.storeDivSA.reserve(estimate.storeDivSA);
	}

	static void clearFusionPlanForReuse(FusionPlan &plan) noexcept
	{
		internal::FusionPlanData &data{plan.data_};

		data.blockCount = {};
		data.directInstructionCount = {};
		data.nextRegister = {};
		data.maxRegisterCount = {};
		data.directOnly = true;
		data.freeRegisters.clear();

		data.addRR.clear();
		data.addRA.clear();
		data.addAR.clear();
		data.addAA.clear();
		data.addRS.clear();
		data.addAS.clear();

		data.subRR.clear();
		data.subRA.clear();
		data.subAR.clear();
		data.subAA.clear();
		data.subRS.clear();
		data.subAS.clear();
		data.subSR.clear();
		data.subSA.clear();

		data.mulRR.clear();
		data.mulRA.clear();
		data.mulAR.clear();
		data.mulAA.clear();
		data.mulRS.clear();
		data.mulAS.clear();

		data.divRR.clear();
		data.divRA.clear();
		data.divAR.clear();
		data.divAA.clear();
		data.divRS.clear();
		data.divAS.clear();
		data.divSR.clear();
		data.divSA.clear();

		data.fmaRRR.clear();
		data.fmaRRA.clear();
		data.fmaRAA.clear();
		data.fmaAAA.clear();
		data.fmaASA.clear();
		data.fmaRAS.clear();
		data.fmaRSA.clear();

		data.storeFmaAAA.clear();
		data.storeFmaASA.clear();
		data.storeFmaRRR.clear();
		data.storeFmaRRA.clear();
		data.storeFmaRAA.clear();
		data.storeFmaRAS.clear();
		data.storeFmaRSA.clear();
		data.storeNegFmaAAA.clear();
		data.storeNegFmaASA.clear();
		data.storeAddAA.clear();
		data.storeAddAS.clear();
		data.storeSubAA.clear();
		data.storeSubAS.clear();
		data.storeSubSA.clear();
		data.storeMulAA.clear();
		data.storeMulAS.clear();
		data.storeDivAA.clear();
		data.storeDivAS.clear();
		data.storeDivSA.clear();

		data.setS.clear();
		data.storeR.clear();
		data.storeA.clear();
		data.storeS.clear();
		data.ops.clear();
	}

	static void compileFusionInto(Engine &engine, const std::vector<Assignment> &assignments,
	                              internal::AssignmentRange range, FusionPlan &plan)
	{
		assert(range.begin < range.end);

		clearFusionPlanForReuse(plan);
		plan.data_.blockCount =
		    assignments[range.begin].out_->variable.array().blockCount();
		const PlanReserveEstimate reserve{estimatePlanReserve(engine, assignments, range)};
		reservePlanStorage(plan, reserve);

		bool directOnly{true};

		for (std::size_t i{range.begin}; i < range.end; ++i) {
			const Assignment &assignment{assignments[i]};
			PendingStore store{};
			if (makeDirectAssignmentStore(engine, assignment, store)) {
				continue;
			}

			directOnly = false;
		}

		std::vector<PendingStore> &stores{engine.impl_->storeScratch};
		stores.clear();
		stores.reserve(assignmentCount(range));

		if (directOnly) {
			for (std::size_t i{range.begin}; i < range.end; ++i) {
				PendingStore store{};
				const bool direct{
				    makeDirectAssignmentStore(engine, assignments[i], store)};
				assert(direct);
				emitDirectPendingStore(plan, store);
			}

			return;
		}

		CompileContext &context{makeCompileContext(engine)};
		for (std::size_t i{range.begin}; i < range.end; ++i) {
			const Assignment &assignment{assignments[i]};
			PendingStore store{};
			if (makeDirectAssignmentStore(engine, assignment, store)) {
				continue;
			}

			countUses(engine, assignment.expr_.nodeId(), context);
		}

		for (std::size_t i{range.begin}; i < range.end; ++i) {
			const Assignment &assignment{assignments[i]};
			PendingStore store{};
			if (makeDirectAssignmentStore(engine, assignment, store)) {
				stores.push_back(store);
				continue;
			}

			if (emitRootFmaStore(engine, plan, assignment, context)) {
				continue;
			}

			ValueRef value{compileAssignmentValue(engine, plan, assignment, context)};
			store.kind = PendingStoreKind::Value;
			store.out = assignment.out_->variable.array().data();
			store.value = value;
			stores.push_back(store);
		}

		for (const PendingStore &store : stores) {
			emitPendingStore(plan, store, context);
		}
	}

	static void clearScheduledPlanForReuse(ScheduledPlan &scheduled) noexcept
	{
		scheduled.data_.blockCount = {};
		for (FusionPlan &stage : scheduled.data_.stages) {
			clearFusionPlanForReuse(stage);
		}
	}

	static void compileScheduledInto(Engine &engine, const std::vector<Assignment> &assignments,
	                                 ScheduledPlan &scheduled)
	{
		if (assignments.empty()) {
			clearScheduledPlanForReuse(scheduled);
			scheduled.data_.stages.clear();
			return;
		}

		const std::vector<internal::AssignmentRange> &stages{
		    buildStages(engine, assignments)};
		scheduled.data_.blockCount =
		    assignments.front().out_->variable.array().blockCount();
		scheduled.data_.stages.reserve(stages.size());
		scheduled.data_.stages.resize(stages.size());

		for (std::size_t i{}; i < stages.size(); ++i) {
			compileFusionInto(engine, assignments, stages[i],
			                  scheduled.data_.stages[i]);
		}
	}

	static const ScheduledPlan &
	compileOrReuseScheduled(Engine &engine, const std::vector<Assignment> &assignments)
	{
		const PlanCacheKey key{validateAssignmentsAndBuildCacheKey(engine, assignments)};

		if (engine.impl_->hasCachedPlan &&
		    samePlanCacheKey(key, engine.impl_->cachedPlanKey)) {
			return engine.impl_->cachedPlan;
		}

		engine.impl_->hasCachedPlan = false;
		compileScheduledInto(engine, assignments, engine.impl_->cachedPlan);
		engine.impl_->cachedPlanKey = key;
		engine.impl_->hasCachedPlan = true;
		return engine.impl_->cachedPlan;
	}
};
}

Engine::Engine() : impl_{std::make_unique<internal::EngineData>()} {}

Engine::~Engine() noexcept = default;

Array Engine::createArray(std::size_t elementCount) { return Array{*this, elementCount}; }

Array Engine::createArray(const std::vector<float> &values) { return Array{*this, values}; }

Array Engine::createArray(std::initializer_list<float> values) { return Array{*this, values}; }

Array Engine::createArray(std::size_t elementCount, float value)
{
	return Array{*this, elementCount, value};
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
	internal::Compiler::resizeOutputIfEmpty(*this, variable, expr);

	internal::Compiler::reserveAssignmentsFor(*this, 1);
	impl_->pendingAssignments.push_back(
	    Assignment{out.impl_.get(), expr, AssignmentKind::Assign});
}

void Engine::deferCompoundAssign(Array &out, const Expression &expr, internal::AssignmentKind kind)
{
	assert(out.impl_ != nullptr);
	Variable &variable{out.impl_->variable};
	internal::Compiler::requireVarOwner(*this, variable,
	                                    "代入先が別のEngineに紐づいています。");
	internal::Compiler::requireExprOwner(*this, expr,
	                                     "代入式が別のEngineに紐づいているか、無効です。");

	internal::Compiler::reserveAssignmentsFor(*this, 1);
	impl_->pendingAssignments.push_back(Assignment{out.impl_.get(), expr, kind});
}

void Engine::execute()
{
	if (impl_->pendingAssignments.empty()) {
		internal::Compiler::rememberExpressionReserve(*this);
		impl_->nodes.clear();
		impl_->nodeKeyIndex.clear();
		return;
	}

	const internal::ScheduledPlan &plan{
	    internal::Compiler::compileOrReuseScheduled(*this, impl_->pendingAssignments)};
	internal::Compiler::rememberAssignmentReserve(*this);
	impl_->pendingAssignments.clear();
	internal::Compiler::rememberExpressionReserve(*this);
	impl_->nodes.clear();
	impl_->nodeKeyIndex.clear();
	plan.execute();
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
