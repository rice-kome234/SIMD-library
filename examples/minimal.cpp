#include "simd.h"

#include <cstddef>
#include <iostream>
#include <vector>

int main()
{
	rice::simd::Engine engine{};

	rice::simd::Array a{engine, {1.0f, 2.0f, 3.0f, 4.0f}};
	rice::simd::Array b{engine};
	rice::simd::Array c{engine, {10.0f, 20.0f, 30.0f, 40.0f}};
	rice::simd::Array out{engine};

	// assign()で要素数と値を一括設定できます。
	b.assign(a.size(), 2.0f);

	// []だけで1要素の読み取りと書き換えができます。
	float lastValue = a[3];
	a[3] = lastValue + 1.0f;

	// outが空の場合は、入力配列と同じ要素数へ自動で確保されます。
	out = a * b + c;
	engine.execute();

	std::vector<float> result{out.toVector()};

	std::cout << "result: ";

	for (const auto &value : result) {
		std::cout << value << " ";
	}

	return 0;
}
