#include "benchmark.h"

#include "simd.h"
#include "simd_low_level.h"

#include <DirectXMath.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <vector>

namespace simdbench
{
namespace
{
using rice::simd::Array;
using rice::simd::Engine;
using namespace DirectX;
namespace low_level = rice::simd::low_level;

// 通常ループの基準値を「SIMDなし」として測るため、自動ベクトル化を抑えます。
#if defined(_MSC_VER)
#define SIMD_BENCH_DISABLE_LOOP_VECTORIZATION __pragma(loop(no_vector))
#elif defined(__clang__)
#define SIMD_BENCH_DISABLE_LOOP_VECTORIZATION _Pragma("clang loop vectorize(disable)")
#else
#define SIMD_BENCH_DISABLE_LOOP_VECTORIZATION
#endif

#if defined(__GNUC__) && !defined(__clang__)
#define SIMD_BENCH_NO_VECTORIZE_FUNCTION __attribute__((optimize("no-tree-vectorize")))
#else
#define SIMD_BENCH_NO_VECTORIZE_FUNCTION
#endif

class Timer
{
public:
	Timer() { reset(); }

	void reset() { start_ = std::chrono::high_resolution_clock::now(); }

	double elapsedMilliseconds() const
	{
		return std::chrono::duration<double, std::milli>(
		           std::chrono::high_resolution_clock::now() - start_)
		    .count();
	}

private:
	std::chrono::high_resolution_clock::time_point start_;
};

struct ThreeComponentArrays {
	ThreeComponentArrays(Engine &engine, std::size_t elementCount)
	    : x{engine, elementCount}, y{engine, elementCount}, z{engine, elementCount}
	{
	}

	Array x;
	Array y;
	Array z;
};

void fillRandom(Array &array, float minValue, float maxValue, std::uint32_t seed)
{
	std::mt19937 engine{seed};
	std::uniform_real_distribution<float> dist{minValue, maxValue};
	alignas(low_level::SIMD_ALIGNMENT) float temp[low_level::SIMD_WIDTH]{};

	for (std::size_t blockIndex{}; blockIndex < low_level::blockCount(array); ++blockIndex) {
		for (std::size_t lane{}; lane < low_level::SIMD_WIDTH; ++lane) {
			const std::size_t elementIndex{blockIndex * low_level::SIMD_WIDTH + lane};
			if (elementIndex < array.elementCount()) {
				temp[lane] = dist(engine);
			} else {
				temp[lane] = 0.0f;
			}
		}

		low_level::block(array, blockIndex) = low_level::loadAligned(temp);
	}
}

std::vector<float> makeRandomVector(std::size_t elementCount, float minValue, float maxValue,
                                    std::uint32_t seed)
{
	std::mt19937 engine{seed};
	std::uniform_real_distribution<float> dist{minValue, maxValue};
	std::vector<float> values(elementCount);

	for (float &value : values) {
		value = dist(engine);
	}

	return values;
}

std::vector<float> makeBatchedAddend(std::size_t elementCount, std::size_t outputIndex)
{
	return makeRandomVector(elementCount, -5.0f, 5.0f,
	                        static_cast<std::uint32_t>(200 + outputIndex));
}

std::vector<std::vector<float>> makeBatchedAddends(std::size_t elementCount,
                                                   std::size_t outputCount)
{
	std::vector<std::vector<float>> addends{};
	addends.reserve(outputCount);

	for (std::size_t i{}; i < outputCount; ++i) {
		addends.push_back(makeBatchedAddend(elementCount, i));
	}

	return addends;
}

float maxAbsError(const std::vector<float> &lhs, const std::vector<float> &rhs)
{
	if (lhs.size() != rhs.size()) {
		throw std::invalid_argument{"比較する配列の要素数が一致しません。"};
	}

	float maxError{};

	for (std::size_t i{}; i < lhs.size(); ++i) {
		maxError = std::max(maxError, std::fabs(lhs[i] - rhs[i]));
	}

	return maxError;
}

void executeExpressionThreeComponentUpdate(Engine &engine, ThreeComponentArrays &position,
                                           ThreeComponentArrays &velocity,
                                           const ThreeComponentArrays &acceleration, float dt)
{
	velocity.x += acceleration.x * dt;
	velocity.y += acceleration.y * dt;
	velocity.z += acceleration.z * dt;

	position.x += velocity.x * dt;
	position.y += velocity.y * dt;
	position.z += velocity.z * dt;

	engine.execute();
}

void executeManualMultiplyAdd(const Array &a, const Array &b, const Array &c, const Array &d,
                              const Array &e, Array &x, Array &y, Array &z)
{
	for (std::size_t i{}; i < low_level::blockCount(x); ++i) {
		low_level::block(x, i) = low_level::multiplyAdd(
		    low_level::block(a, i), low_level::block(b, i), low_level::block(c, i));
		low_level::block(y, i) = low_level::multiplyAdd(
		    low_level::block(a, i), low_level::block(b, i), low_level::block(d, i));
		low_level::block(z, i) = low_level::multiplyAdd(
		    low_level::block(a, i), low_level::block(b, i), low_level::block(e, i));
	}
}

void executeDirectXMathMultiplyAdd(const std::vector<float> &a, const std::vector<float> &b,
                                   const std::vector<float> &c, const std::vector<float> &d,
                                   const std::vector<float> &e, std::vector<float> &x,
                                   std::vector<float> &y, std::vector<float> &z)
{
	constexpr std::size_t directXMathWidth{4};
	std::size_t i{};

	for (; i + directXMathWidth <= x.size(); i += directXMathWidth) {
		const XMVECTOR av{XMLoadFloat4(reinterpret_cast<const XMFLOAT4 *>(a.data() + i))};
		const XMVECTOR bv{XMLoadFloat4(reinterpret_cast<const XMFLOAT4 *>(b.data() + i))};
		const XMVECTOR cv{XMLoadFloat4(reinterpret_cast<const XMFLOAT4 *>(c.data() + i))};
		const XMVECTOR dv{XMLoadFloat4(reinterpret_cast<const XMFLOAT4 *>(d.data() + i))};
		const XMVECTOR ev{XMLoadFloat4(reinterpret_cast<const XMFLOAT4 *>(e.data() + i))};

		XMStoreFloat4(reinterpret_cast<XMFLOAT4 *>(x.data() + i),
		              XMVectorMultiplyAdd(av, bv, cv));
		XMStoreFloat4(reinterpret_cast<XMFLOAT4 *>(y.data() + i),
		              XMVectorMultiplyAdd(av, bv, dv));
		XMStoreFloat4(reinterpret_cast<XMFLOAT4 *>(z.data() + i),
		              XMVectorMultiplyAdd(av, bv, ev));
	}

	for (; i < x.size(); ++i) {
		x[i] = a[i] * b[i] + c[i];
		y[i] = a[i] * b[i] + d[i];
		z[i] = a[i] * b[i] + e[i];
	}
}

SIMD_BENCH_NO_VECTORIZE_FUNCTION
void executeScalarMultiplyAdd(const std::vector<float> &a, const std::vector<float> &b,
                              const std::vector<float> &c, const std::vector<float> &d,
                              const std::vector<float> &e, std::vector<float> &x,
                              std::vector<float> &y, std::vector<float> &z)
{
	SIMD_BENCH_DISABLE_LOOP_VECTORIZATION
	for (std::size_t i{}; i < x.size(); ++i) {

		x[i] = a[i] * b[i] + c[i];
		y[i] = a[i] * b[i] + d[i];
		z[i] = a[i] * b[i] + e[i];
	}
}

void executeExpressionApiMultiplyAdd(Engine &engine, const Array &a, const Array &b, const Array &c,
                                     const Array &d, const Array &e, Array &x, Array &y, Array &z)
{
	x = a * b + c;
	y = a * b + d;
	z = a * b + e;

	engine.execute();
}

SIMD_BENCH_NO_VECTORIZE_FUNCTION
void executeScalarBatchedMultiplyAdd(const std::vector<float> &a, const std::vector<float> &b,
                                     const std::vector<std::vector<float>> &addends,
                                     std::vector<std::vector<float>> &outputs)
{
	for (std::size_t outputIndex{}; outputIndex < outputs.size(); ++outputIndex) {
		const std::vector<float> &addend{addends[outputIndex]};
		std::vector<float> &out{outputs[outputIndex]};

		SIMD_BENCH_DISABLE_LOOP_VECTORIZATION
		for (std::size_t i{}; i < out.size(); ++i) {
			out[i] = a[i] * b[i] + addend[i];
		}
	}
}

void executeManualBatchedMultiplyAdd(const Array &a, const Array &b,
                                     const std::vector<Array> &addends,
                                     std::vector<Array> &outputs)
{
	for (std::size_t outputIndex{}; outputIndex < outputs.size(); ++outputIndex) {
		Array &out{outputs[outputIndex]};
		const Array &addend{addends[outputIndex]};

		for (std::size_t blockIndex{}; blockIndex < low_level::blockCount(out);
		     ++blockIndex) {
			low_level::block(out, blockIndex) =
			    low_level::multiplyAdd(low_level::block(a, blockIndex),
			                           low_level::block(b, blockIndex),
			                           low_level::block(addend, blockIndex));
		}
	}
}

void executeDirectXMathBatchedMultiplyAdd(const std::vector<float> &a,
                                          const std::vector<float> &b,
                                          const std::vector<std::vector<float>> &addends,
                                          std::vector<std::vector<float>> &outputs)
{
	constexpr std::size_t directXMathWidth{4};

	for (std::size_t outputIndex{}; outputIndex < outputs.size(); ++outputIndex) {
		const std::vector<float> &addend{addends[outputIndex]};
		std::vector<float> &out{outputs[outputIndex]};
		std::size_t i{};

		for (; i + directXMathWidth <= out.size(); i += directXMathWidth) {
			const XMVECTOR av{
			    XMLoadFloat4(reinterpret_cast<const XMFLOAT4 *>(a.data() + i))};
			const XMVECTOR bv{
			    XMLoadFloat4(reinterpret_cast<const XMFLOAT4 *>(b.data() + i))};
			const XMVECTOR addendVector{
			    XMLoadFloat4(reinterpret_cast<const XMFLOAT4 *>(addend.data() + i))};

			XMStoreFloat4(reinterpret_cast<XMFLOAT4 *>(out.data() + i),
			              XMVectorMultiplyAdd(av, bv, addendVector));
		}

		for (; i < out.size(); ++i) {
			out[i] = a[i] * b[i] + addend[i];
		}
	}
}

SIMD_BENCH_NO_VECTORIZE_FUNCTION
void executeScalarHeavyExpression(const std::vector<float> &a, const std::vector<float> &b,
                                  const std::vector<float> &c, const std::vector<float> &d,
                                  const std::vector<float> &e, const std::vector<float> &f,
                                  std::vector<float> &out)
{
	SIMD_BENCH_DISABLE_LOOP_VECTORIZATION
	for (std::size_t i{}; i < out.size(); ++i) {
		out[i] = (a[i] * b[i] + c[i]) * (d[i] * e[i] + f[i]) +
		         (a[i] + d[i]) * (b[i] + e[i]);
	}
}

void executeManualHeavyExpression(const Array &a, const Array &b, const Array &c, const Array &d,
                                  const Array &e, const Array &f, Array &out)
{
	for (std::size_t i{}; i < low_level::blockCount(out); ++i) {
		low_level::block(out, i) = low_level::multiplyAdd(
		    low_level::multiplyAdd(low_level::block(a, i), low_level::block(b, i),
		                           low_level::block(c, i)),
		    low_level::multiplyAdd(low_level::block(d, i), low_level::block(e, i),
		                           low_level::block(f, i)),
		    low_level::mul(low_level::add(low_level::block(a, i), low_level::block(d, i)),
		                   low_level::add(low_level::block(b, i), low_level::block(e, i))));
	}
}

void executeDirectXMathHeavyExpression(const std::vector<float> &a, const std::vector<float> &b,
                                       const std::vector<float> &c, const std::vector<float> &d,
                                       const std::vector<float> &e, const std::vector<float> &f,
                                       std::vector<float> &out)
{
	constexpr std::size_t directXMathWidth{4};
	std::size_t i{};

	for (; i + directXMathWidth <= out.size(); i += directXMathWidth) {
		const XMVECTOR av{XMLoadFloat4(reinterpret_cast<const XMFLOAT4 *>(a.data() + i))};
		const XMVECTOR bv{XMLoadFloat4(reinterpret_cast<const XMFLOAT4 *>(b.data() + i))};
		const XMVECTOR cv{XMLoadFloat4(reinterpret_cast<const XMFLOAT4 *>(c.data() + i))};
		const XMVECTOR dv{XMLoadFloat4(reinterpret_cast<const XMFLOAT4 *>(d.data() + i))};
		const XMVECTOR ev{XMLoadFloat4(reinterpret_cast<const XMFLOAT4 *>(e.data() + i))};
		const XMVECTOR fv{XMLoadFloat4(reinterpret_cast<const XMFLOAT4 *>(f.data() + i))};

		XMStoreFloat4(reinterpret_cast<XMFLOAT4 *>(out.data() + i),
		              XMVectorMultiplyAdd(
		                  XMVectorMultiplyAdd(av, bv, cv),
		                  XMVectorMultiplyAdd(dv, ev, fv),
		                  XMVectorMultiply(XMVectorAdd(av, dv), XMVectorAdd(bv, ev))));
	}

	for (; i < out.size(); ++i) {
		out[i] = (a[i] * b[i] + c[i]) * (d[i] * e[i] + f[i]) +
		         (a[i] + d[i]) * (b[i] + e[i]);
	}
}

void executeExpressionHeavyExpression(Engine &engine, const Array &a, const Array &b,
                                      const Array &c, const Array &d, const Array &e,
                                      const Array &f, Array &out)
{
	out = (a * b + c) * (d * e + f) + (a + d) * (b + e);
	engine.execute();
}

void directThreeComponentUpdate(ThreeComponentArrays &position, ThreeComponentArrays &velocity,
                                const ThreeComponentArrays &acceleration, float dt)
{
	const low_level::Block dtBlock{low_level::set1(dt)};

	for (std::size_t i{}; i < low_level::blockCount(position.x); ++i) {
		low_level::block(velocity.x, i) = low_level::multiplyAdd(
		    low_level::block(acceleration.x, i), dtBlock, low_level::block(velocity.x, i));

		low_level::block(velocity.y, i) = low_level::multiplyAdd(
		    low_level::block(acceleration.y, i), dtBlock, low_level::block(velocity.y, i));

		low_level::block(velocity.z, i) = low_level::multiplyAdd(
		    low_level::block(acceleration.z, i), dtBlock, low_level::block(velocity.z, i));

		low_level::block(position.x, i) = low_level::multiplyAdd(
		    low_level::block(velocity.x, i), dtBlock, low_level::block(position.x, i));

		low_level::block(position.y, i) = low_level::multiplyAdd(
		    low_level::block(velocity.y, i), dtBlock, low_level::block(position.y, i));

		low_level::block(position.z, i) = low_level::multiplyAdd(
		    low_level::block(velocity.z, i), dtBlock, low_level::block(position.z, i));
	}
}

void directXMathThreeComponentUpdate(std::vector<float> &positionX, std::vector<float> &positionY,
                                     std::vector<float> &positionZ, std::vector<float> &velocityX,
                                     std::vector<float> &velocityY, std::vector<float> &velocityZ,
                                     const std::vector<float> &accelerationX,
                                     const std::vector<float> &accelerationY,
                                     const std::vector<float> &accelerationZ, float dt)
{
	constexpr std::size_t directXMathWidth{4};
	const XMVECTOR dtVector{XMVectorReplicate(dt)};
	std::size_t i{};

	for (; i + directXMathWidth <= positionX.size(); i += directXMathWidth) {
		const XMVECTOR accelerationXV{
		    XMLoadFloat4(reinterpret_cast<const XMFLOAT4 *>(accelerationX.data() + i))};
		const XMVECTOR accelerationYV{
		    XMLoadFloat4(reinterpret_cast<const XMFLOAT4 *>(accelerationY.data() + i))};
		const XMVECTOR accelerationZV{
		    XMLoadFloat4(reinterpret_cast<const XMFLOAT4 *>(accelerationZ.data() + i))};

		XMVECTOR velocityXV{
		    XMLoadFloat4(reinterpret_cast<const XMFLOAT4 *>(velocityX.data() + i))};
		XMVECTOR velocityYV{
		    XMLoadFloat4(reinterpret_cast<const XMFLOAT4 *>(velocityY.data() + i))};
		XMVECTOR velocityZV{
		    XMLoadFloat4(reinterpret_cast<const XMFLOAT4 *>(velocityZ.data() + i))};

		velocityXV = XMVectorMultiplyAdd(accelerationXV, dtVector, velocityXV);
		velocityYV = XMVectorMultiplyAdd(accelerationYV, dtVector, velocityYV);
		velocityZV = XMVectorMultiplyAdd(accelerationZV, dtVector, velocityZV);

		XMStoreFloat4(reinterpret_cast<XMFLOAT4 *>(velocityX.data() + i), velocityXV);
		XMStoreFloat4(reinterpret_cast<XMFLOAT4 *>(velocityY.data() + i), velocityYV);
		XMStoreFloat4(reinterpret_cast<XMFLOAT4 *>(velocityZ.data() + i), velocityZV);

		const XMVECTOR positionXV{
		    XMLoadFloat4(reinterpret_cast<const XMFLOAT4 *>(positionX.data() + i))};
		const XMVECTOR positionYV{
		    XMLoadFloat4(reinterpret_cast<const XMFLOAT4 *>(positionY.data() + i))};
		const XMVECTOR positionZV{
		    XMLoadFloat4(reinterpret_cast<const XMFLOAT4 *>(positionZ.data() + i))};

		XMStoreFloat4(reinterpret_cast<XMFLOAT4 *>(positionX.data() + i),
		              XMVectorMultiplyAdd(velocityXV, dtVector, positionXV));
		XMStoreFloat4(reinterpret_cast<XMFLOAT4 *>(positionY.data() + i),
		              XMVectorMultiplyAdd(velocityYV, dtVector, positionYV));
		XMStoreFloat4(reinterpret_cast<XMFLOAT4 *>(positionZ.data() + i),
		              XMVectorMultiplyAdd(velocityZV, dtVector, positionZV));
	}

	for (; i < positionX.size(); ++i) {
		velocityX[i] += accelerationX[i] * dt;
		velocityY[i] += accelerationY[i] * dt;
		velocityZ[i] += accelerationZ[i] * dt;

		positionX[i] += velocityX[i] * dt;
		positionY[i] += velocityY[i] * dt;
		positionZ[i] += velocityZ[i] * dt;
	}
}

SIMD_BENCH_NO_VECTORIZE_FUNCTION void directScalarThreeComponentUpdate(
    std::vector<float> &positionX, std::vector<float> &positionY, std::vector<float> &positionZ,
    std::vector<float> &velocityX, std::vector<float> &velocityY, std::vector<float> &velocityZ,
    const std::vector<float> &accelerationX, const std::vector<float> &accelerationY,
    const std::vector<float> &accelerationZ, float dt)
{
	SIMD_BENCH_DISABLE_LOOP_VECTORIZATION
	for (std::size_t i{}; i < positionX.size(); ++i) {
		velocityX[i] += accelerationX[i] * dt;
		velocityY[i] += accelerationY[i] * dt;
		velocityZ[i] += accelerationZ[i] * dt;

		positionX[i] += velocityX[i] * dt;
		positionY[i] += velocityY[i] * dt;
		positionZ[i] += velocityZ[i] * dt;
	}
}

void copyThreeComponent(const ThreeComponentArrays &source, ThreeComponentArrays &destination)
{
	destination.x.copyFrom(source.x);
	destination.y.copyFrom(source.y);
	destination.z.copyFrom(source.z);
}
}

FusedExpressionBenchmarkResult runFusedExpressionBenchmark(std::size_t elementCount,
                                                           int repeatCount)
{
	Engine engine{};

	Array a{engine, elementCount};
	Array b{engine, elementCount};
	Array c{engine, elementCount};
	Array d{engine, elementCount};
	Array e{engine, elementCount};

	fillRandom(a, -10.0f, 10.0f, 1);
	fillRandom(b, -10.0f, 10.0f, 2);
	fillRandom(c, -10.0f, 10.0f, 3);
	fillRandom(d, -10.0f, 10.0f, 4);
	fillRandom(e, -10.0f, 10.0f, 5);

	std::vector<float> scalarA{};
	std::vector<float> scalarB{};
	std::vector<float> scalarC{};
	std::vector<float> scalarD{};
	std::vector<float> scalarE{};

	a.copyTo(scalarA);
	b.copyTo(scalarB);
	c.copyTo(scalarC);
	d.copyTo(scalarD);
	e.copyTo(scalarE);

	std::vector<float> scalarX{};
	std::vector<float> scalarY{};
	std::vector<float> scalarZ{};
	scalarX.resize(elementCount);
	scalarY.resize(elementCount);
	scalarZ.resize(elementCount);

	Timer scalarTimer{};

	for (int i{}; i < repeatCount; ++i) {
		executeScalarMultiplyAdd(scalarA, scalarB, scalarC, scalarD, scalarE, scalarX,
		                         scalarY, scalarZ);
	}

	const double scalarMs{scalarTimer.elapsedMilliseconds()};

	Array manualX{engine, elementCount};
	Array manualY{engine, elementCount};
	Array manualZ{engine, elementCount};

	Timer manualTimer{};

	for (int i{}; i < repeatCount; ++i) {
		executeManualMultiplyAdd(a, b, c, d, e, manualX, manualY, manualZ);
	}

	const double manualMs{manualTimer.elapsedMilliseconds()};

	std::vector<float> directXMathX{};
	std::vector<float> directXMathY{};
	std::vector<float> directXMathZ{};
	directXMathX.resize(elementCount);
	directXMathY.resize(elementCount);
	directXMathZ.resize(elementCount);

	Timer directXMathTimer{};

	for (int i{}; i < repeatCount; ++i) {
		executeDirectXMathMultiplyAdd(scalarA, scalarB, scalarC, scalarD, scalarE,
		                              directXMathX, directXMathY, directXMathZ);
	}

	const double directXMathMs{directXMathTimer.elapsedMilliseconds()};

	Array expressionX{engine};
	Array expressionY{engine};
	Array expressionZ{engine};

	Timer expressionTimer{};

	for (int i{}; i < repeatCount; ++i) {
		executeExpressionApiMultiplyAdd(engine, a, b, c, d, e, expressionX, expressionY,
		                                expressionZ);
	}

	const double expressionMs{expressionTimer.elapsedMilliseconds()};

	std::vector<float> manualXS{};
	std::vector<float> expressionXS{};

	manualX.copyTo(manualXS);
	expressionX.copyTo(expressionXS);

	FusedExpressionBenchmarkResult result{};
	result.elementCount = elementCount;
	result.repeatCount = repeatCount;
	result.scalarMs = scalarMs;
	result.manualSimdMs = manualMs;
	result.directXMathMs = directXMathMs;
	result.normalExpressionMs = expressionMs;
	result.scalarXError = maxAbsError(scalarX, manualXS);
	result.directXMathXError = maxAbsError(scalarX, directXMathX);
	result.expressionXError = maxAbsError(scalarX, expressionXS);
	return result;
}

ThreeComponentUpdateBenchmarkResult
runThreeComponentUpdateBenchmark(std::size_t elementCount, int repeatCount, float deltaTime)
{
	Engine engine{};

	ThreeComponentArrays initialPosition{engine, elementCount};
	ThreeComponentArrays initialVelocity{engine, elementCount};
	ThreeComponentArrays acceleration{engine, elementCount};

	fillRandom(initialPosition.x, -100.0f, 100.0f, 10);
	fillRandom(initialPosition.y, -100.0f, 100.0f, 11);
	fillRandom(initialPosition.z, -100.0f, 100.0f, 12);

	fillRandom(initialVelocity.x, -10.0f, 10.0f, 20);
	fillRandom(initialVelocity.y, -10.0f, 10.0f, 21);
	fillRandom(initialVelocity.z, -10.0f, 10.0f, 22);

	fillRandom(acceleration.x, -1.0f, 1.0f, 30);
	fillRandom(acceleration.y, -1.0f, 1.0f, 31);
	fillRandom(acceleration.z, -1.0f, 1.0f, 32);

	std::vector<float> scalarPositionX{};
	std::vector<float> scalarPositionY{};
	std::vector<float> scalarPositionZ{};
	std::vector<float> scalarVelocityX{};
	std::vector<float> scalarVelocityY{};
	std::vector<float> scalarVelocityZ{};
	std::vector<float> scalarAccelerationX{};
	std::vector<float> scalarAccelerationY{};
	std::vector<float> scalarAccelerationZ{};

	initialPosition.x.copyTo(scalarPositionX);
	initialPosition.y.copyTo(scalarPositionY);
	initialPosition.z.copyTo(scalarPositionZ);
	initialVelocity.x.copyTo(scalarVelocityX);
	initialVelocity.y.copyTo(scalarVelocityY);
	initialVelocity.z.copyTo(scalarVelocityZ);
	acceleration.x.copyTo(scalarAccelerationX);
	acceleration.y.copyTo(scalarAccelerationY);
	acceleration.z.copyTo(scalarAccelerationZ);

	const std::vector<float> initialScalarPositionX{scalarPositionX};
	const std::vector<float> initialScalarPositionY{scalarPositionY};
	const std::vector<float> initialScalarPositionZ{scalarPositionZ};
	const std::vector<float> initialScalarVelocityX{scalarVelocityX};
	const std::vector<float> initialScalarVelocityY{scalarVelocityY};
	const std::vector<float> initialScalarVelocityZ{scalarVelocityZ};

	Timer scalarTimer{};

	for (int i{}; i < repeatCount; ++i) {
		directScalarThreeComponentUpdate(scalarPositionX, scalarPositionY, scalarPositionZ,
		                                 scalarVelocityX, scalarVelocityY, scalarVelocityZ,
		                                 scalarAccelerationX, scalarAccelerationY,
		                                 scalarAccelerationZ, deltaTime);
	}

	const double scalarMs{scalarTimer.elapsedMilliseconds()};

	ThreeComponentArrays directPosition{engine, elementCount};
	ThreeComponentArrays directVelocity{engine, elementCount};
	copyThreeComponent(initialPosition, directPosition);
	copyThreeComponent(initialVelocity, directVelocity);

	Timer directThreeComponentTimer{};

	for (int i{}; i < repeatCount; ++i) {
		directThreeComponentUpdate(directPosition, directVelocity, acceleration, deltaTime);
	}

	const double directThreeComponentMs{directThreeComponentTimer.elapsedMilliseconds()};

	std::vector<float> directXMathPositionX{initialScalarPositionX};
	std::vector<float> directXMathPositionY{initialScalarPositionY};
	std::vector<float> directXMathPositionZ{initialScalarPositionZ};
	std::vector<float> directXMathVelocityX{initialScalarVelocityX};
	std::vector<float> directXMathVelocityY{initialScalarVelocityY};
	std::vector<float> directXMathVelocityZ{initialScalarVelocityZ};

	Timer directXMathTimer{};

	for (int i{}; i < repeatCount; ++i) {
		directXMathThreeComponentUpdate(
		    directXMathPositionX, directXMathPositionY, directXMathPositionZ,
		    directXMathVelocityX, directXMathVelocityY, directXMathVelocityZ,
		    scalarAccelerationX, scalarAccelerationY, scalarAccelerationZ, deltaTime);
	}

	const double directXMathMs{directXMathTimer.elapsedMilliseconds()};

	ThreeComponentArrays specializedPosition{engine, elementCount};
	ThreeComponentArrays specializedVelocity{engine, elementCount};
	copyThreeComponent(initialPosition, specializedPosition);
	copyThreeComponent(initialVelocity, specializedVelocity);

	Timer specializedTimer{};

	for (int i{}; i < repeatCount; ++i) {
		executeExpressionThreeComponentUpdate(engine, specializedPosition,
		                                      specializedVelocity, acceleration, deltaTime);
	}

	const double specializedMs{specializedTimer.elapsedMilliseconds()};

	std::vector<float> directPosXS{};
	std::vector<float> specializedPosXS{};

	directPosition.x.copyTo(directPosXS);
	specializedPosition.x.copyTo(specializedPosXS);

	ThreeComponentUpdateBenchmarkResult result{};
	result.elementCount = elementCount;
	result.repeatCount = repeatCount;
	result.deltaTime = deltaTime;
	result.scalarMs = scalarMs;
	result.manualSimdMs = directThreeComponentMs;
	result.directXMathMs = directXMathMs;
	result.specializedUpdateMs = specializedMs;
	result.scalarPositionXError = maxAbsError(scalarPositionX, directPosXS);
	result.directXMathPositionXError = maxAbsError(scalarPositionX, directXMathPositionX);
	result.specializedPositionXError = maxAbsError(scalarPositionX, specializedPosXS);
	return result;
}

BatchedExecutionBenchmarkResult runBatchedExecutionBenchmark(std::size_t elementCount,
                                                             std::size_t outputCount,
                                                             int repeatCount)
{
	if (outputCount == 0) {
		throw std::invalid_argument{"出力配列数は1以上にしてください。"};
	}

	const std::vector<float> scalarA{makeRandomVector(elementCount, -10.0f, 10.0f, 101)};
	const std::vector<float> scalarB{makeRandomVector(elementCount, -10.0f, 10.0f, 102)};

	double scalarMs{};
	double manualMs{};
	double directXMathMs{};
	double sequentialMs{};
	double batchedMs{};

	std::vector<float> scalarFirstOutput{};
	std::vector<float> manualFirstOutput{};
	std::vector<float> directXMathFirstOutput{};
	std::vector<float> sequentialFirstOutput{};
	std::vector<float> batchedFirstOutput{};

	{
		std::vector<std::vector<float>> scalarAddends{
		    makeBatchedAddends(elementCount, outputCount)};
		std::vector<std::vector<float>> scalarOutputs(outputCount);
		for (std::vector<float> &output : scalarOutputs) {
			output.resize(elementCount);
		}

		Timer scalarTimer{};

		for (int i{}; i < repeatCount; ++i) {
			executeScalarBatchedMultiplyAdd(scalarA, scalarB, scalarAddends, scalarOutputs);
		}

		scalarMs = scalarTimer.elapsedMilliseconds();
		scalarFirstOutput = scalarOutputs.front();
	}

	{
		Engine manualEngine{};
		Array manualA{manualEngine, scalarA};
		Array manualB{manualEngine, scalarB};
		std::vector<Array> manualAddends{};
		std::vector<Array> manualOutputs{};
		manualAddends.reserve(outputCount);
		manualOutputs.reserve(outputCount);

		for (std::size_t i{}; i < outputCount; ++i) {
			manualAddends.emplace_back(manualEngine, makeBatchedAddend(elementCount, i));
			manualOutputs.emplace_back(manualEngine, elementCount);
		}

		Timer manualTimer{};

		for (int i{}; i < repeatCount; ++i) {
			executeManualBatchedMultiplyAdd(manualA, manualB, manualAddends,
			                                manualOutputs);
		}

		manualMs = manualTimer.elapsedMilliseconds();
		manualOutputs.front().copyTo(manualFirstOutput);
	}

	{
		std::vector<std::vector<float>> directXMathAddends{
		    makeBatchedAddends(elementCount, outputCount)};
		std::vector<std::vector<float>> directXMathOutputs(outputCount);
		for (std::vector<float> &output : directXMathOutputs) {
			output.resize(elementCount);
		}

		Timer directXMathTimer{};

		for (int i{}; i < repeatCount; ++i) {
			executeDirectXMathBatchedMultiplyAdd(scalarA, scalarB, directXMathAddends,
			                                     directXMathOutputs);
		}

		directXMathMs = directXMathTimer.elapsedMilliseconds();
		directXMathFirstOutput = directXMathOutputs.front();
	}

	{
		Engine sequentialEngine{};
		Array sequentialA{sequentialEngine, scalarA};
		Array sequentialB{sequentialEngine, scalarB};
		std::vector<Array> sequentialAddends{};
		std::vector<Array> sequentialOutputs{};
		sequentialAddends.reserve(outputCount);
		sequentialOutputs.reserve(outputCount);

		for (std::size_t i{}; i < outputCount; ++i) {
			sequentialAddends.emplace_back(sequentialEngine,
			                               makeBatchedAddend(elementCount, i));
			sequentialOutputs.emplace_back(sequentialEngine, elementCount);
		}

		Timer sequentialTimer{};

		for (int i{}; i < repeatCount; ++i) {
			for (std::size_t outputIndex{}; outputIndex < outputCount; ++outputIndex) {
				sequentialOutputs[outputIndex] =
				    sequentialA * sequentialB + sequentialAddends[outputIndex];
				sequentialEngine.execute();
			}
		}

		sequentialMs = sequentialTimer.elapsedMilliseconds();
		sequentialOutputs.front().copyTo(sequentialFirstOutput);
	}

	{
		Engine batchedEngine{};
		Array batchedA{batchedEngine, scalarA};
		Array batchedB{batchedEngine, scalarB};
		std::vector<Array> batchedAddends{};
		std::vector<Array> batchedOutputs{};
		batchedAddends.reserve(outputCount);
		batchedOutputs.reserve(outputCount);

		for (std::size_t i{}; i < outputCount; ++i) {
			batchedAddends.emplace_back(batchedEngine, makeBatchedAddend(elementCount, i));
			batchedOutputs.emplace_back(batchedEngine, elementCount);
		}

		Timer batchedTimer{};

		for (int i{}; i < repeatCount; ++i) {
			for (std::size_t outputIndex{}; outputIndex < outputCount; ++outputIndex) {
				batchedOutputs[outputIndex] =
				    batchedA * batchedB + batchedAddends[outputIndex];
			}

			batchedEngine.execute();
		}

		batchedMs = batchedTimer.elapsedMilliseconds();
		batchedOutputs.front().copyTo(batchedFirstOutput);
	}

	BatchedExecutionBenchmarkResult result{};
	result.elementCount = elementCount;
	result.outputCount = outputCount;
	result.repeatCount = repeatCount;
	result.scalarMs = scalarMs;
	result.manualSimdMs = manualMs;
	result.directXMathMs = directXMathMs;
	result.sequentialExpressionMs = sequentialMs;
	result.batchedExpressionMs = batchedMs;
	result.manualSimdFirstOutputError = maxAbsError(scalarFirstOutput, manualFirstOutput);
	result.directXMathFirstOutputError =
	    maxAbsError(scalarFirstOutput, directXMathFirstOutput);
	result.sequentialFirstOutputError = maxAbsError(scalarFirstOutput, sequentialFirstOutput);
	result.batchedFirstOutputError = maxAbsError(scalarFirstOutput, batchedFirstOutput);
	return result;
}

HeavyExpressionBenchmarkResult runHeavyExpressionBenchmark(std::size_t elementCount,
                                                           int repeatCount)
{
	const std::vector<float> scalarA{makeRandomVector(elementCount, -2.0f, 2.0f, 301)};
	const std::vector<float> scalarB{makeRandomVector(elementCount, -2.0f, 2.0f, 302)};
	const std::vector<float> scalarC{makeRandomVector(elementCount, -2.0f, 2.0f, 303)};
	const std::vector<float> scalarD{makeRandomVector(elementCount, -2.0f, 2.0f, 304)};
	const std::vector<float> scalarE{makeRandomVector(elementCount, -2.0f, 2.0f, 305)};
	const std::vector<float> scalarF{makeRandomVector(elementCount, -2.0f, 2.0f, 306)};

	std::vector<float> scalarOut(elementCount);

	Timer scalarTimer{};

	for (int i{}; i < repeatCount; ++i) {
		executeScalarHeavyExpression(scalarA, scalarB, scalarC, scalarD, scalarE, scalarF,
		                             scalarOut);
	}

	const double scalarMs{scalarTimer.elapsedMilliseconds()};

	Engine manualEngine{};
	Array manualA{manualEngine, scalarA};
	Array manualB{manualEngine, scalarB};
	Array manualC{manualEngine, scalarC};
	Array manualD{manualEngine, scalarD};
	Array manualE{manualEngine, scalarE};
	Array manualF{manualEngine, scalarF};
	Array manualOut{manualEngine, elementCount};

	Timer manualTimer{};

	for (int i{}; i < repeatCount; ++i) {
		executeManualHeavyExpression(manualA, manualB, manualC, manualD, manualE, manualF,
		                             manualOut);
	}

	const double manualMs{manualTimer.elapsedMilliseconds()};

	std::vector<float> directXMathOut(elementCount);

	Timer directXMathTimer{};

	for (int i{}; i < repeatCount; ++i) {
		executeDirectXMathHeavyExpression(scalarA, scalarB, scalarC, scalarD, scalarE,
		                                  scalarF, directXMathOut);
	}

	const double directXMathMs{directXMathTimer.elapsedMilliseconds()};

	Engine expressionEngine{};
	Array expressionA{expressionEngine, scalarA};
	Array expressionB{expressionEngine, scalarB};
	Array expressionC{expressionEngine, scalarC};
	Array expressionD{expressionEngine, scalarD};
	Array expressionE{expressionEngine, scalarE};
	Array expressionF{expressionEngine, scalarF};
	Array expressionOut{expressionEngine, elementCount};

	Timer expressionTimer{};

	for (int i{}; i < repeatCount; ++i) {
		executeExpressionHeavyExpression(expressionEngine, expressionA, expressionB,
		                                 expressionC, expressionD, expressionE,
		                                 expressionF, expressionOut);
	}

	const double expressionMs{expressionTimer.elapsedMilliseconds()};

	std::vector<float> manualResult{};
	std::vector<float> expressionResult{};
	manualOut.copyTo(manualResult);
	expressionOut.copyTo(expressionResult);

	HeavyExpressionBenchmarkResult result{};
	result.elementCount = elementCount;
	result.repeatCount = repeatCount;
	result.scalarMs = scalarMs;
	result.manualSimdMs = manualMs;
	result.directXMathMs = directXMathMs;
	result.expressionMs = expressionMs;
	result.manualSimdError = maxAbsError(scalarOut, manualResult);
	result.directXMathError = maxAbsError(scalarOut, directXMathOut);
	result.expressionError = maxAbsError(scalarOut, expressionResult);
	return result;
}
}
