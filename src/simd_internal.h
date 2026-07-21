#pragma once

#include "simd.h"
#include "simd_backend.h"

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

namespace rice::simd::internal
{
inline constexpr int MAX_REGISTERS{32};
// 間接呼び出し1回で処理するブロック数。大きくし過ぎると作業領域がL1キャッシュを圧迫します。
inline constexpr std::size_t EXECUTION_BLOCK_TILE{4};

using NodeId = std::uint32_t;
inline constexpr NodeId INVALID_NODE{std::numeric_limits<NodeId>::max()};

class FloatArray
{
public:
	explicit FloatArray(std::size_t elementCount = {});

	FloatArray(const FloatArray &) = delete;
	FloatArray &operator=(const FloatArray &) = delete;

	void resize(std::size_t elementCount);
	std::size_t elementCount() const noexcept;
	std::size_t blockCount() const noexcept;
	std::uint32_t arrayId() const noexcept;
	std::uint64_t storageGeneration() const noexcept;
	void setArrayId(std::uint32_t arrayId) noexcept;
	SimdBlock &block(std::size_t index) noexcept;
	const SimdBlock &block(std::size_t index) const noexcept;
	SimdBlock *data() noexcept;
	const SimdBlock *data() const noexcept;
	void fill(float value) noexcept;
	void assign(std::size_t elementCount, float value);
	void copyFrom(const FloatArray &source);
	void copyFrom(const std::vector<float> &values);
	void copyFrom(std::initializer_list<float> values);
	void push_back(float value);
	float element(std::size_t index) const noexcept;
	void setElement(std::size_t index, float value) noexcept;
	void copyTo(std::vector<float> &out) const;

private:
	void copyFrom(const float *values, std::size_t count);
	void advanceStorageGeneration() noexcept;

	std::size_t elementCount_{};
	std::vector<SimdBlock> blocks_;
	std::uint32_t arrayId_{};
	std::uint64_t storageGeneration_{1};
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

enum class AssignmentKind { Assign, AddAssign, SubAssign, MulAssign, DivAssign };

struct Assignment {
	ArrayData *out_{};
	Expression expr_{};
	AssignmentKind kind{AssignmentKind::Assign};
};

enum class NodeKind : std::uint8_t { Variable, Scalar, Add, Sub, Mul, Div };

struct ExprNode {
	std::uint64_t key{};
	union {
		const FloatArray *array;
		float scalar;
	};
	NodeId lhs{INVALID_NODE};
	NodeId rhs{INVALID_NODE};
	NodeKind kind{NodeKind::Scalar};

	ExprNode() noexcept : array{} {}
};

static_assert(sizeof(ExprNode) <= 32);

enum class ValueKind : std::uint8_t { Reg, Array, Scalar };

struct ValueRef {
	const SimdBlock *array{};
	float scalar{};
	NodeId group{INVALID_NODE};
	int reg{-1};
	ValueKind kind{ValueKind::Scalar};
	bool hasGroup{};
};

static_assert(sizeof(ValueRef) <= 24);

struct AddRR {
	int dst;
	int a;
	int b;
};
struct AddRA {
	int dst;
	int a;
	const SimdBlock *b;
};
struct AddAR {
	int dst;
	const SimdBlock *a;
	int b;
};
struct AddAA {
	int dst;
	const SimdBlock *a;
	const SimdBlock *b;
};
struct AddRS {
	int dst;
	int a;
	SimdBlock b;
};
struct AddAS {
	int dst;
	const SimdBlock *a;
	SimdBlock b;
};

struct SubRR {
	int dst;
	int a;
	int b;
};
struct SubRA {
	int dst;
	int a;
	const SimdBlock *b;
};
struct SubAR {
	int dst;
	const SimdBlock *a;
	int b;
};
struct SubAA {
	int dst;
	const SimdBlock *a;
	const SimdBlock *b;
};
struct SubRS {
	int dst;
	int a;
	SimdBlock b;
};
struct SubAS {
	int dst;
	const SimdBlock *a;
	SimdBlock b;
};
struct SubSR {
	int dst;
	SimdBlock a;
	int b;
};
struct SubSA {
	int dst;
	SimdBlock a;
	const SimdBlock *b;
};

struct MulRR {
	int dst;
	int a;
	int b;
};
struct MulRA {
	int dst;
	int a;
	const SimdBlock *b;
};
struct MulAR {
	int dst;
	const SimdBlock *a;
	int b;
};
struct MulAA {
	int dst;
	const SimdBlock *a;
	const SimdBlock *b;
};
struct MulRS {
	int dst;
	int a;
	SimdBlock b;
};
struct MulAS {
	int dst;
	const SimdBlock *a;
	SimdBlock b;
};

struct DivRR {
	int dst;
	int a;
	int b;
};
struct DivRA {
	int dst;
	int a;
	const SimdBlock *b;
};
struct DivAR {
	int dst;
	const SimdBlock *a;
	int b;
};
struct DivAA {
	int dst;
	const SimdBlock *a;
	const SimdBlock *b;
};
struct DivRS {
	int dst;
	int a;
	SimdBlock b;
};
struct DivAS {
	int dst;
	const SimdBlock *a;
	SimdBlock b;
};
struct DivSR {
	int dst;
	SimdBlock a;
	int b;
};
struct DivSA {
	int dst;
	SimdBlock a;
	const SimdBlock *b;
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
	const SimdBlock *c;
};
struct FmaRAA {
	int dst;
	int a;
	const SimdBlock *b;
	const SimdBlock *c;
};
struct FmaAAA {
	int dst;
	const SimdBlock *a;
	const SimdBlock *b;
	const SimdBlock *c;
};
struct FmaASA {
	int dst;
	const SimdBlock *a;
	SimdBlock b;
	const SimdBlock *c;
};
struct FmaRAS {
	int dst;
	int a;
	const SimdBlock *b;
	SimdBlock c;
};
struct FmaRSA {
	int dst;
	int a;
	SimdBlock b;
	const SimdBlock *c;
};

struct StoreFmaAAA {
	SimdBlock *out;
	const SimdBlock *a;
	const SimdBlock *b;
	const SimdBlock *c;
};
struct StoreFmaASA {
	SimdBlock *out;
	const SimdBlock *a;
	SimdBlock b;
	const SimdBlock *c;
};
struct StoreFmaRRR {
	SimdBlock *out;
	int a;
	int b;
	int c;
};
struct StoreFmaRRA {
	SimdBlock *out;
	int a;
	int b;
	const SimdBlock *c;
};
struct StoreFmaRAA {
	SimdBlock *out;
	int a;
	const SimdBlock *b;
	const SimdBlock *c;
};
struct StoreFmaRAS {
	SimdBlock *out;
	int a;
	const SimdBlock *b;
	SimdBlock c;
};
struct StoreFmaRSA {
	SimdBlock *out;
	int a;
	SimdBlock b;
	const SimdBlock *c;
};
struct StoreNegFmaAAA {
	SimdBlock *out;
	const SimdBlock *a;
	const SimdBlock *b;
	const SimdBlock *c;
};
struct StoreNegFmaASA {
	SimdBlock *out;
	const SimdBlock *a;
	SimdBlock b;
	const SimdBlock *c;
};
struct StoreBinaryAA {
	SimdBlock *out;
	const SimdBlock *a;
	const SimdBlock *b;
};
struct StoreBinaryAS {
	SimdBlock *out;
	const SimdBlock *a;
	SimdBlock b;
};
struct StoreBinarySA {
	SimdBlock *out;
	SimdBlock a;
	const SimdBlock *b;
};

struct SetS {
	int dst;
	SimdBlock a;
};
struct StoreR {
	SimdBlock *out;
	int a;
};
struct StoreA {
	SimdBlock *out;
	const SimdBlock *a;
};
struct StoreS {
	SimdBlock *out;
	SimdBlock a;
};

struct StoreFmaPlusFma {
	SimdBlock *out{};
	const SimdBlock *leftA{};
	const SimdBlock *leftB{};
	const SimdBlock *leftC{};
	const SimdBlock *rightA{};
	const SimdBlock *rightB{};
	const SimdBlock *rightC{};
};

struct StoreNestedFma {
	SimdBlock *out{};
	const SimdBlock *innerA{};
	const SimdBlock *innerB{};
	const SimdBlock *innerC{};
	const SimdBlock *mul{};
	const SimdBlock *add{};
};

enum class StoreCompoundKind : std::uint8_t { Add, Sub, Mul, Div };

struct StoreCompoundFma {
	SimdBlock *out{};
	const SimdBlock *a{};
	const SimdBlock *b{};
	const SimdBlock *c{};
	StoreCompoundKind kind{StoreCompoundKind::Add};
};

struct FusionPlanData;

template <typename T>
class InstructionBuffer
{
public:
	using Storage = std::vector<T>;
	using const_iterator = typename Storage::const_iterator;

	void reserve(std::size_t count)
	{
		if (count != 0) {
			ensureStorage().reserve(count);
		}
	}

	void clear() noexcept
	{
		if (storage_ != nullptr) {
			storage_->clear();
		}
	}

	void push_back(const T &value) { ensureStorage().push_back(value); }

	std::size_t size() const noexcept
	{
		if (storage_ == nullptr) {
			return 0;
		}

		return storage_->size();
	}

	std::size_t capacity() const noexcept
	{
		if (storage_ == nullptr) {
			return 0;
		}

		return storage_->capacity();
	}

	bool empty() const noexcept { return size() == 0; }

	T &operator[](std::size_t index) noexcept { return (*storage_)[index]; }
	const T &operator[](std::size_t index) const noexcept { return (*storage_)[index]; }

	const_iterator begin() const noexcept { return storage().begin(); }
	const_iterator end() const noexcept { return storage().end(); }

	void releaseIfUnused(std::size_t maxUnusedCapacity) noexcept
	{
		if (storage_ != nullptr && storage_->empty() &&
		    storage_->capacity() > maxUnusedCapacity) {
			storage_.reset();
		}
	}

private:
	// 使用する命令種類だけvectorを生成し、空の命令種類が持つ固定費を抑えます。
	Storage &ensureStorage()
	{
		if (storage_ == nullptr) {
			storage_ = std::make_unique<Storage>();
		}

		return *storage_;
	}

	const Storage &storage() const noexcept
	{
		if (storage_ == nullptr) {
			static const Storage emptyStorage{};
			return emptyStorage;
		}

		return *storage_;
	}

	std::unique_ptr<Storage> storage_;
};

using SingleOpExecutor = void (*)(const FusionPlanData &plan, std::size_t blockIndex,
                                  SimdBlock *regs, std::size_t index) noexcept;
using OpExecutor = void (*)(const FusionPlanData &plan, std::size_t firstBlock,
                            std::size_t blockCount, SimdBlock *regs,
                            std::size_t index) noexcept;
using DirectBlockExecutor = void (*)(const FusionPlanData &plan,
                                     std::size_t blockIndex) noexcept;
using DirectPlanExecutor = void (*)(const FusionPlanData &plan) noexcept;

struct OpRef {
	OpExecutor execute{};
	std::size_t index{};
};

struct DirectGroupRef {
	DirectBlockExecutor executeBlock{};
	DirectPlanExecutor executePlan{};
};

struct FusionPlanData {
	std::size_t blockCount{};
	std::size_t directInstructionCount{};
	int nextRegister{};
	int maxRegisterCount{};
	bool directOnly{true};
	std::vector<int> freeRegisters;

	InstructionBuffer<AddRR> addRR;
	InstructionBuffer<AddRA> addRA;
	InstructionBuffer<AddAR> addAR;
	InstructionBuffer<AddAA> addAA;
	InstructionBuffer<AddRS> addRS;
	InstructionBuffer<AddAS> addAS;

	InstructionBuffer<SubRR> subRR;
	InstructionBuffer<SubRA> subRA;
	InstructionBuffer<SubAR> subAR;
	InstructionBuffer<SubAA> subAA;
	InstructionBuffer<SubRS> subRS;
	InstructionBuffer<SubAS> subAS;
	InstructionBuffer<SubSR> subSR;
	InstructionBuffer<SubSA> subSA;

	InstructionBuffer<MulRR> mulRR;
	InstructionBuffer<MulRA> mulRA;
	InstructionBuffer<MulAR> mulAR;
	InstructionBuffer<MulAA> mulAA;
	InstructionBuffer<MulRS> mulRS;
	InstructionBuffer<MulAS> mulAS;

	InstructionBuffer<DivRR> divRR;
	InstructionBuffer<DivRA> divRA;
	InstructionBuffer<DivAR> divAR;
	InstructionBuffer<DivAA> divAA;
	InstructionBuffer<DivRS> divRS;
	InstructionBuffer<DivAS> divAS;
	InstructionBuffer<DivSR> divSR;
	InstructionBuffer<DivSA> divSA;

	InstructionBuffer<FmaRRR> fmaRRR;
	InstructionBuffer<FmaRRA> fmaRRA;
	InstructionBuffer<FmaRAA> fmaRAA;
	InstructionBuffer<FmaAAA> fmaAAA;
	InstructionBuffer<FmaASA> fmaASA;
	InstructionBuffer<FmaRAS> fmaRAS;
	InstructionBuffer<FmaRSA> fmaRSA;

	InstructionBuffer<StoreFmaAAA> storeFmaAAA;
	InstructionBuffer<StoreFmaASA> storeFmaASA;
	InstructionBuffer<StoreFmaRRR> storeFmaRRR;
	InstructionBuffer<StoreFmaRRA> storeFmaRRA;
	InstructionBuffer<StoreFmaRAA> storeFmaRAA;
	InstructionBuffer<StoreFmaRAS> storeFmaRAS;
	InstructionBuffer<StoreFmaRSA> storeFmaRSA;
	InstructionBuffer<StoreNegFmaAAA> storeNegFmaAAA;
	InstructionBuffer<StoreNegFmaASA> storeNegFmaASA;
	InstructionBuffer<StoreBinaryAA> storeAddAA;
	InstructionBuffer<StoreBinaryAS> storeAddAS;
	InstructionBuffer<StoreBinaryAA> storeSubAA;
	InstructionBuffer<StoreBinaryAS> storeSubAS;
	InstructionBuffer<StoreBinarySA> storeSubSA;
	InstructionBuffer<StoreBinaryAA> storeMulAA;
	InstructionBuffer<StoreBinaryAS> storeMulAS;
	InstructionBuffer<StoreBinaryAA> storeDivAA;
	InstructionBuffer<StoreBinaryAS> storeDivAS;
	InstructionBuffer<StoreBinarySA> storeDivSA;

	InstructionBuffer<SetS> setS;
	InstructionBuffer<StoreR> storeR;
	InstructionBuffer<StoreA> storeA;
	InstructionBuffer<StoreS> storeS;
	InstructionBuffer<StoreFmaFmaAddProduct> storeFmaFmaAddProducts;
	InstructionBuffer<StoreFmaPlusFma> storeFmaPlusFmas;
	InstructionBuffer<StoreNestedFma> storeNestedFmas;
	InstructionBuffer<StoreCompoundFma> storeCompoundFmas;
	std::vector<OpRef> ops;
	std::vector<DirectGroupRef> directGroups;
	DirectPlanExecutor directExecutor{};
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

	void executeBlocks(std::size_t firstBlock, std::size_t blockCount,
	                   SimdBlock *regs) const noexcept;
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

struct ArrayBinding {
	const FloatArray *array{};
	std::uint64_t storageGeneration{};
};

struct KeyIndex {
	std::uint64_t key{};
	NodeId node{};
};

struct CompileGroup {
	std::uint64_t key{};
	ValueRef cachedValue{};
	NodeId node{INVALID_NODE};
	int useCount{};
	bool hasCachedValue{};
};

struct AssignmentRange {
	std::size_t begin{};
	std::size_t end{};
};

enum class PendingStoreKind : std::uint8_t {
	Value,
	FmaAAA,
	FmaASA,
	NegFmaAAA,
	NegFmaASA,
	AddAA,
	AddAS,
	SubAA,
	SubAS,
	SubSA,
	MulAA,
	MulAS,
	DivAA,
	DivAS,
	DivSA
};

struct PendingStore {
	PendingStoreKind kind{PendingStoreKind::Value};
	SimdBlock *out{};
	ValueRef value{};
	const SimdBlock *a{};
	const SimdBlock *bArray{};
	SimdBlock bScalar{zeroBlock()};
	const SimdBlock *c{};
};

struct CompileContext {
	std::vector<NodeId> groupByNode;
	std::vector<CompileGroup> groups;
};

struct EngineData {
	std::vector<ExprNode> nodes;
	std::vector<Assignment> pendingAssignments;
	std::size_t expressionReserveHint{};
	std::size_t assignmentReserveHint{};

	ScheduledPlan cachedPlan;
	PlanCacheKey cachedPlanKey;
	std::vector<ArrayBinding> cachedPlanBindings;
	std::vector<ArrayBinding> bindingScratch;
	bool hasCachedPlan{};
	std::uint32_t nextArrayId{1};
	std::uint32_t stageMarkGeneration{1};

	std::vector<AssignmentRange> stageRanges;
	std::vector<const FloatArray *> readScratch;
	std::vector<std::uint32_t> stageWriteMarks;
	std::vector<KeyIndex> nodeKeyIndex;
	std::vector<PendingStore> storeScratch;
	CompileContext compileContext;
};
}
