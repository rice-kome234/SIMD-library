#include "benchmark.h"

#include "simd.h"
#include "simd_low_level.h"

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
namespace low_level = rice::simd::low_level;

#if defined(__GNUC__) && !defined(__clang__)
#define SIMDTEST_NO_VECTORIZE __attribute__((optimize("no-tree-vectorize")))
#else
#define SIMDTEST_NO_VECTORIZE
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

struct ThreeComponentUpdatePlan {
	std::size_t blockCount{};
	__m256 dt{_mm256_setzero_ps()};

	__m256 *positionX{};
	__m256 *positionY{};
	__m256 *positionZ{};

	__m256 *velocityX{};
	__m256 *velocityY{};
	__m256 *velocityZ{};

	const __m256 *accelerationX{};
	const __m256 *accelerationY{};
	const __m256 *accelerationZ{};

	void execute() const
	{
		for (std::size_t i{}; i < blockCount; ++i) {
			velocityX[i] = _mm256_fmadd_ps(accelerationX[i], dt, velocityX[i]);
			velocityY[i] = _mm256_fmadd_ps(accelerationY[i], dt, velocityY[i]);
			velocityZ[i] = _mm256_fmadd_ps(accelerationZ[i], dt, velocityZ[i]);

			positionX[i] = _mm256_fmadd_ps(velocityX[i], dt, positionX[i]);
			positionY[i] = _mm256_fmadd_ps(velocityY[i], dt, positionY[i]);
			positionZ[i] = _mm256_fmadd_ps(velocityZ[i], dt, positionZ[i]);
		}
	}
};

ThreeComponentUpdatePlan makeThreeComponentUpdatePlan(ThreeComponentArrays &position,
                                                      ThreeComponentArrays &velocity,
                                                      const ThreeComponentArrays &acceleration,
                                                      float dt)
{
	ThreeComponentUpdatePlan plan{};
	plan.blockCount = low_level::blockCount(position.x);
	plan.dt = _mm256_set1_ps(dt);

	plan.positionX = low_level::data(position.x);
	plan.positionY = low_level::data(position.y);
	plan.positionZ = low_level::data(position.z);

	plan.velocityX = low_level::data(velocity.x);
	plan.velocityY = low_level::data(velocity.y);
	plan.velocityZ = low_level::data(velocity.z);

	plan.accelerationX = low_level::data(acceleration.x);
	plan.accelerationY = low_level::data(acceleration.y);
	plan.accelerationZ = low_level::data(acceleration.z);

	return plan;
}

void executeManualSharedMultiplyAdd(const Array &a, const Array &b, const Array &c, const Array &d,
                                    const Array &e, Array &x, Array &y, Array &z)
{
	// ab = a * b を共有し、x/y/zには別々の加算項を足します。
	for (std::size_t i{}; i < low_level::blockCount(x); ++i) {
		const __m256 ab{_mm256_mul_ps(low_level::block(a, i), low_level::block(b, i))};

		low_level::block(x, i) = _mm256_add_ps(ab, low_level::block(c, i));
		low_level::block(y, i) = _mm256_add_ps(ab, low_level::block(d, i));
		low_level::block(z, i) = _mm256_add_ps(ab, low_level::block(e, i));
	}
}

SIMDTEST_NO_VECTORIZE void
executeScalarSharedMultiplyAdd(const std::vector<float> &a, const std::vector<float> &b,
                               const std::vector<float> &c, const std::vector<float> &d,
                               const std::vector<float> &e, std::vector<float> &x,
                               std::vector<float> &y, std::vector<float> &z)
{
#if defined(_MSC_VER)
#pragma loop(no_vector)
#elif defined(__clang__)
#pragma clang loop vectorize(disable)
#endif
	for (std::size_t i{}; i < x.size(); ++i) {
		const float ab{a[i] * b[i]};

		x[i] = ab + c[i];
		y[i] = ab + d[i];
		z[i] = ab + e[i];
	}
}

void executeExpressionApiSharedMultiplyAdd(Engine &engine, const Array &a, const Array &b,
                                           const Array &c, const Array &d, const Array &e, Array &x,
                                           Array &y, Array &z)
{
	const auto ab{a * b};

	x = ab + c;
	y = ab + d;
	z = ab + e;

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

SIMDTEST_NO_VECTORIZE void directScalarThreeComponentUpdate(
    std::vector<float> &positionX, std::vector<float> &positionY, std::vector<float> &positionZ,
    std::vector<float> &velocityX, std::vector<float> &velocityY, std::vector<float> &velocityZ,
    const std::vector<float> &accelerationX, const std::vector<float> &accelerationY,
    const std::vector<float> &accelerationZ, float dt)
{
#if defined(_MSC_VER)
#pragma loop(no_vector)
#elif defined(__clang__)
#pragma clang loop vectorize(disable)
#endif
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
		executeScalarSharedMultiplyAdd(scalarA, scalarB, scalarC, scalarD, scalarE, scalarX,
		                               scalarY, scalarZ);
	}

	const double scalarMs{scalarTimer.elapsedMilliseconds()};

	Array manualX{engine, elementCount};
	Array manualY{engine, elementCount};
	Array manualZ{engine, elementCount};

	Timer manualTimer{};

	for (int i{}; i < repeatCount; ++i) {
		executeManualSharedMultiplyAdd(a, b, c, d, e, manualX, manualY, manualZ);
	}

	const double manualMs{manualTimer.elapsedMilliseconds()};

	Array expressionX{engine};
	Array expressionY{engine};
	Array expressionZ{engine};

	Timer expressionTimer{};

	for (int i{}; i < repeatCount; ++i) {
		executeExpressionApiSharedMultiplyAdd(engine, a, b, c, d, e, expressionX,
		                                      expressionY, expressionZ);
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
	result.normalExpressionMs = expressionMs;
	result.scalarXError = maxAbsError(manualXS, scalarX);
	result.expressionXError = maxAbsError(manualXS, expressionXS);
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

	ThreeComponentArrays specializedPosition{engine, elementCount};
	ThreeComponentArrays specializedVelocity{engine, elementCount};
	copyThreeComponent(initialPosition, specializedPosition);
	copyThreeComponent(initialVelocity, specializedVelocity);

	ThreeComponentUpdatePlan specializedUpdate{makeThreeComponentUpdatePlan(
	    specializedPosition, specializedVelocity, acceleration, deltaTime)};

	Timer specializedTimer{};

	for (int i{}; i < repeatCount; ++i) {
		specializedUpdate.execute();
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
	result.specializedUpdateMs = specializedMs;
	result.scalarPositionXError = maxAbsError(directPosXS, scalarPositionX);
	result.specializedPositionXError = maxAbsError(directPosXS, specializedPosXS);
	return result;
}
}
