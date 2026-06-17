#include "benchmark.h"

#include "simd.h"
#include "simd_low_level.h"

#include <DirectXMath.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <immintrin.h>
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
	alignas(32) float temp[low_level::SIMD_WIDTH]{};

	for (std::size_t blockIndex{}; blockIndex < low_level::blockCount(array); ++blockIndex) {
		for (std::size_t lane{}; lane < low_level::SIMD_WIDTH; ++lane) {
			const std::size_t elementIndex{blockIndex * low_level::SIMD_WIDTH + lane};
			if (elementIndex < array.elementCount()) {
				temp[lane] = dist(engine);
			} else {
				temp[lane] = 0.0f;
			}
		}

		low_level::block(array, blockIndex) = _mm256_load_ps(temp);
	}
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

void makeThreeComponentUpdatePlan(Engine &engine, ThreeComponentArrays &position,
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
		low_level::block(x, i) =
		    _mm256_add_ps((_mm256_mul_ps(low_level::block(a, i), low_level::block(b, i))),
		                  low_level::block(c, i));
		low_level::block(y, i) =
		    _mm256_add_ps((_mm256_mul_ps(low_level::block(a, i), low_level::block(b, i))),
		                  low_level::block(d, i));
		low_level::block(z, i) =
		    _mm256_add_ps((_mm256_mul_ps(low_level::block(a, i), low_level::block(b, i))),
		                  low_level::block(e, i));
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
                         const std::vector<float> &e, std::vector<float> &x, std::vector<float> &y,
                         std::vector<float> &z)
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

void directThreeComponentUpdate(ThreeComponentArrays &position, ThreeComponentArrays &velocity,
                                const ThreeComponentArrays &acceleration, float dt)
{
	const __m256 dtBlock{_mm256_set1_ps(dt)};

	for (std::size_t i{}; i < low_level::blockCount(position.x); ++i) {
		low_level::block(velocity.x, i) = _mm256_fmadd_ps(
		    low_level::block(acceleration.x, i), dtBlock, low_level::block(velocity.x, i));

		low_level::block(velocity.y, i) = _mm256_fmadd_ps(
		    low_level::block(acceleration.y, i), dtBlock, low_level::block(velocity.y, i));

		low_level::block(velocity.z, i) = _mm256_fmadd_ps(
		    low_level::block(acceleration.z, i), dtBlock, low_level::block(velocity.z, i));

		low_level::block(position.x, i) = _mm256_fmadd_ps(
		    low_level::block(velocity.x, i), dtBlock, low_level::block(position.x, i));

		low_level::block(position.y, i) = _mm256_fmadd_ps(
		    low_level::block(velocity.y, i), dtBlock, low_level::block(position.y, i));

		low_level::block(position.z, i) = _mm256_fmadd_ps(
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
		makeThreeComponentUpdatePlan(engine, specializedPosition, specializedVelocity,
		                             acceleration, deltaTime);
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
}
