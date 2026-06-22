#pragma once

#include "simd.h"
#include "simd_backend.h"

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <utility>
#include <vector>

namespace rice::simd::internal
{
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

	std::size_t elementCount_{};
	std::vector<SimdBlock> blocks_;
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
	const SimdBlock *array{};
	SimdBlock scalar{zeroBlock()};
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

struct StoreFmaFmaAddProduct {
	SimdBlock *out{};
	const SimdBlock *leftA{};
	const SimdBlock *leftB{};
	const SimdBlock *leftC{};
	const SimdBlock *rightA{};
	const SimdBlock *rightB{};
	const SimdBlock *rightC{};
	const SimdBlock *addLeftA{};
	const SimdBlock *addLeftB{};
	const SimdBlock *addRightA{};
	const SimdBlock *addRightB{};
};

struct FusionPlanData;

using OpExecutor = void (*)(const FusionPlanData &plan, std::size_t blockIndex, SimdBlock *regs,
                            std::size_t index) noexcept;

struct OpRef {
	OpExecutor execute{};
	std::size_t index{};
};

struct FusionPlanData {
	std::size_t blockCount{};
	std::size_t directInstructionCount{};
	int nextRegister{};
	int maxRegisterCount{};
	bool directOnly{true};
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
	std::vector<StoreFmaRRR> storeFmaRRR;
	std::vector<StoreFmaRRA> storeFmaRRA;
	std::vector<StoreFmaRAA> storeFmaRAA;
	std::vector<StoreFmaRAS> storeFmaRAS;
	std::vector<StoreFmaRSA> storeFmaRSA;
	std::vector<StoreNegFmaAAA> storeNegFmaAAA;
	std::vector<StoreNegFmaASA> storeNegFmaASA;
	std::vector<StoreBinaryAA> storeAddAA;
	std::vector<StoreBinaryAS> storeAddAS;
	std::vector<StoreBinaryAA> storeSubAA;
	std::vector<StoreBinaryAS> storeSubAS;
	std::vector<StoreBinarySA> storeSubSA;
	std::vector<StoreBinaryAA> storeMulAA;
	std::vector<StoreBinaryAS> storeMulAS;
	std::vector<StoreBinaryAA> storeDivAA;
	std::vector<StoreBinaryAS> storeDivAS;
	std::vector<StoreBinarySA> storeDivSA;

	std::vector<SetS> setS;
	std::vector<StoreR> storeR;
	std::vector<StoreA> storeA;
	std::vector<StoreS> storeS;
	std::vector<StoreFmaFmaAddProduct> storeFmaFmaAddProducts;
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

	void executeBlock(std::size_t blockIndex, SimdBlock *regs) const noexcept;
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

enum class PendingStoreKind {
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
	std::vector<KeyIndex> nodeKeyIndex;
	std::vector<PendingStore> storeScratch;
	CompileContext compileContext;
};
}
