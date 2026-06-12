#pragma once

#include "simd.h"

#include <cstddef>
#include <cstdint>
#include <immintrin.h>
#include <initializer_list>
#include <limits>
#include <utility>
#include <vector>

namespace rice::simd::internal
{
inline constexpr std::size_t SIMD_WIDTH{8};
inline constexpr int MAX_REGISTERS{32};
inline constexpr std::size_t INVALID_NODE{std::numeric_limits<std::size_t>::max()};

class FloatArray
{
public:
	explicit FloatArray(std::size_t elementCount = {});

	FloatArray &operator=(const FloatArray &) = default;

	void resize(std::size_t elementCount);
	std::size_t elementCount() const noexcept;
	std::size_t blockCount() const noexcept;
	__m256 &block(std::size_t index) noexcept;
	const __m256 &block(std::size_t index) const noexcept;
	__m256 *data() noexcept;
	const __m256 *data() const noexcept;
	void fill(float value) noexcept;
	void assign(std::size_t elementCount, float value);
	void copyFrom(const std::vector<float> &values);
	void copyFrom(std::initializer_list<float> values);
	void push_back(float value);
	void copyTo(std::vector<float> &out) const;

private:
	void copyFrom(const float *values, std::size_t count);
	void setElement(std::size_t index, float value) noexcept;

	std::size_t elementCount_{};
	std::vector<__m256> blocks_;
};

class Variable
{
public:
	Variable() noexcept = default;
	Variable(Engine *engine, FloatArray *array) noexcept;

	FloatArray &array() noexcept;
	const FloatArray &array() const noexcept;
	Engine *engine() const noexcept;

private:
	Engine *engine_{};
	FloatArray *array_{};
};

struct ArrayData {
	ArrayData(Engine &engine, std::size_t elementCount);

	FloatArray storage;
	Variable variable;
};

struct Assignment {
	ArrayData *out_{};
	Expression expr_{};
};

enum class NodeKind { Variable, Scalar, Add, Sub, Mul, Div };

struct ExprNode {
	NodeKind kind{NodeKind::Scalar};
	const FloatArray *array{};
	float scalar{};
	std::size_t lhs{INVALID_NODE};
	std::size_t rhs{INVALID_NODE};
	std::uint64_t key{};
};

enum class ValueKind { Reg, Array, Scalar };

struct ValueRef {
	ValueKind kind{ValueKind::Scalar};
	int reg{-1};
	const __m256 *array{};
	__m256 scalar{_mm256_setzero_ps()};
	std::size_t group{INVALID_NODE};
	bool hasGroup{};
};

struct AddRR {
	int dst;
	int a;
	int b;
};
struct AddRA {
	int dst;
	int a;
	const __m256 *b;
};
struct AddAR {
	int dst;
	const __m256 *a;
	int b;
};
struct AddAA {
	int dst;
	const __m256 *a;
	const __m256 *b;
};
struct AddRS {
	int dst;
	int a;
	__m256 b;
};
struct AddAS {
	int dst;
	const __m256 *a;
	__m256 b;
};

struct SubRR {
	int dst;
	int a;
	int b;
};
struct SubRA {
	int dst;
	int a;
	const __m256 *b;
};
struct SubAR {
	int dst;
	const __m256 *a;
	int b;
};
struct SubAA {
	int dst;
	const __m256 *a;
	const __m256 *b;
};
struct SubRS {
	int dst;
	int a;
	__m256 b;
};
struct SubAS {
	int dst;
	const __m256 *a;
	__m256 b;
};
struct SubSR {
	int dst;
	__m256 a;
	int b;
};
struct SubSA {
	int dst;
	__m256 a;
	const __m256 *b;
};

struct MulRR {
	int dst;
	int a;
	int b;
};
struct MulRA {
	int dst;
	int a;
	const __m256 *b;
};
struct MulAR {
	int dst;
	const __m256 *a;
	int b;
};
struct MulAA {
	int dst;
	const __m256 *a;
	const __m256 *b;
};
struct MulRS {
	int dst;
	int a;
	__m256 b;
};
struct MulAS {
	int dst;
	const __m256 *a;
	__m256 b;
};

struct DivRR {
	int dst;
	int a;
	int b;
};
struct DivRA {
	int dst;
	int a;
	const __m256 *b;
};
struct DivAR {
	int dst;
	const __m256 *a;
	int b;
};
struct DivAA {
	int dst;
	const __m256 *a;
	const __m256 *b;
};
struct DivRS {
	int dst;
	int a;
	__m256 b;
};
struct DivAS {
	int dst;
	const __m256 *a;
	__m256 b;
};
struct DivSR {
	int dst;
	__m256 a;
	int b;
};
struct DivSA {
	int dst;
	__m256 a;
	const __m256 *b;
};

struct FmaRRR {
	int dst;
	int a;
	int b;
	int c;
};
struct FmaRRA {
	int dst;
	int a;
	int b;
	const __m256 *c;
};
struct FmaRAA {
	int dst;
	int a;
	const __m256 *b;
	const __m256 *c;
};
struct FmaAAA {
	int dst;
	const __m256 *a;
	const __m256 *b;
	const __m256 *c;
};
struct FmaASA {
	int dst;
	const __m256 *a;
	__m256 b;
	const __m256 *c;
};
struct FmaRAS {
	int dst;
	int a;
	const __m256 *b;
	__m256 c;
};
struct FmaRSA {
	int dst;
	int a;
	__m256 b;
	const __m256 *c;
};

struct StoreFmaAAA {
	__m256 *out;
	const __m256 *a;
	const __m256 *b;
	const __m256 *c;
};
struct StoreFmaASA {
	__m256 *out;
	const __m256 *a;
	__m256 b;
	const __m256 *c;
};

struct SetS {
	int dst;
	__m256 a;
};
struct StoreR {
	__m256 *out;
	int a;
};
struct StoreA {
	__m256 *out;
	const __m256 *a;
};
struct StoreS {
	__m256 *out;
	__m256 a;
};

struct FusionPlanData;

using OpExecutor = void (*)(const FusionPlanData &plan, std::size_t blockIndex, __m256 *regs,
                            std::size_t index) noexcept;

struct OpRef {
	OpExecutor execute{};
	std::size_t index{};
};

struct FusionPlanData {
	std::size_t blockCount{};
	int nextRegister{};
	int maxRegisterCount{};
	std::vector<int> freeRegisters;

	std::vector<AddRR> addRR;
	std::vector<AddRA> addRA;
	std::vector<AddAR> addAR;
	std::vector<AddAA> addAA;
	std::vector<AddRS> addRS;
	std::vector<AddAS> addAS;

	std::vector<SubRR> subRR;
	std::vector<SubRA> subRA;
	std::vector<SubAR> subAR;
	std::vector<SubAA> subAA;
	std::vector<SubRS> subRS;
	std::vector<SubAS> subAS;
	std::vector<SubSR> subSR;
	std::vector<SubSA> subSA;

	std::vector<MulRR> mulRR;
	std::vector<MulRA> mulRA;
	std::vector<MulAR> mulAR;
	std::vector<MulAA> mulAA;
	std::vector<MulRS> mulRS;
	std::vector<MulAS> mulAS;

	std::vector<DivRR> divRR;
	std::vector<DivRA> divRA;
	std::vector<DivAR> divAR;
	std::vector<DivAA> divAA;
	std::vector<DivRS> divRS;
	std::vector<DivAS> divAS;
	std::vector<DivSR> divSR;
	std::vector<DivSA> divSA;

	std::vector<FmaRRR> fmaRRR;
	std::vector<FmaRRA> fmaRRA;
	std::vector<FmaRAA> fmaRAA;
	std::vector<FmaAAA> fmaAAA;
	std::vector<FmaASA> fmaASA;
	std::vector<FmaRAS> fmaRAS;
	std::vector<FmaRSA> fmaRSA;

	std::vector<StoreFmaAAA> storeFmaAAA;
	std::vector<StoreFmaASA> storeFmaASA;

	std::vector<SetS> setS;
	std::vector<StoreR> storeR;
	std::vector<StoreA> storeA;
	std::vector<StoreS> storeS;
	std::vector<OpRef> ops;
};

class FusionPlan
{
public:
	FusionPlan();
	~FusionPlan() noexcept;

	FusionPlan(const FusionPlan &) = delete;
	FusionPlan &operator=(const FusionPlan &) = delete;
	FusionPlan(FusionPlan &&) noexcept = default;
	FusionPlan &operator=(FusionPlan &&) noexcept = default;

	void execute() const noexcept;
	std::size_t instructionCount() const noexcept;
	int registerCount() const noexcept;

private:
	friend class ScheduledPlan;
	friend struct Compiler;

	void executeBlock(std::size_t blockIndex, __m256 *regs) const noexcept;
	int allocateRegister() noexcept;
	void releaseRegister(int reg);

	FusionPlanData data_;
};

struct ScheduledPlanData {
	std::size_t blockCount{};
	std::vector<FusionPlan> stages;
};

class ScheduledPlan
{
public:
	ScheduledPlan();
	~ScheduledPlan() noexcept;

	ScheduledPlan(const ScheduledPlan &) = delete;
	ScheduledPlan &operator=(const ScheduledPlan &) = delete;
	ScheduledPlan(ScheduledPlan &&) noexcept = default;
	ScheduledPlan &operator=(ScheduledPlan &&) noexcept = default;

	void execute() const noexcept;
	std::size_t stageCount() const noexcept;
	std::size_t instructionCount() const noexcept;
	int maxRegisterCount() const noexcept;

private:
	friend struct Compiler;
	friend struct DebugAccess;

	ScheduledPlanData data_;
};

struct PlanCacheKey {
	std::uint64_t hash{};
	std::size_t assignmentCount{};
};

struct KeyIndex {
	std::uint64_t key{};
	std::size_t node{};
};

struct CompileGroup {
	std::uint64_t key{};
	std::size_t node{INVALID_NODE};
	int useCount{};
	ValueRef cachedValue{};
	bool hasCachedValue{};
};

struct AssignmentRange {
	std::size_t begin{};
	std::size_t end{};
};

enum class PendingStoreKind { Value, FmaAAA, FmaASA };

struct PendingStore {
	PendingStoreKind kind{PendingStoreKind::Value};
	__m256 *out{};
	ValueRef value{};
	const __m256 *a{};
	const __m256 *bArray{};
	__m256 bScalar{_mm256_setzero_ps()};
	const __m256 *c{};
};

struct CompileContext {
	std::vector<std::size_t> groupByNode;
	std::vector<CompileGroup> groups;
};

struct EngineData {
	std::vector<ExprNode> nodes;
	std::vector<Assignment> pendingAssignments;
	std::size_t expressionReserveHint{};
	std::size_t assignmentReserveHint{};

	ScheduledPlan cachedPlan;
	PlanCacheKey cachedPlanKey;
	bool hasCachedPlan{};

	std::vector<AssignmentRange> stageRanges;
	std::vector<const FloatArray *> readScratch;
	std::vector<const FloatArray *> writeScratch;
	std::vector<KeyIndex> keyScratch;
	std::vector<PendingStore> storeScratch;
	CompileContext compileContext;
};
}
