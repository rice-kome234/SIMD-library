# SIMD-library

`SIMD-library` は、`float` 配列に対する要素ごとの計算を、通常の式に近い書き方でSIMD実行するための小さなC++ライブラリです。利用側は基本的に `simd.h` だけを読み、ビルド済みの `simd.lib` をリンクします。

## 特徴

- `rice::simd::Array` が配列の寿命を所有します。
- `rice::simd::Engine` が配列の作成、式ノード生成、コンパイル、実行を担当します。
- `x = a * b + c;` のように、計算式をそのまま書けます。
- 同じ式を何度も実行する場合は、`ScheduledPlan` として事前コンパイルできます。
- 配列サイズの不一致は `compile()` または `execute()` のタイミングで検出します。

## ビルド

```powershell
cmake -S . -B build-cmake
cmake --build build-cmake --config Release
```

Visual Studio のジェネレーターでは、`build-cmake\Release\simd.lib` が生成されます。

## バイナリ配布パッケージ

公開ヘッダーと静的ライブラリだけを含む、組み込み用の配布パッケージを生成できます。
アプリケーション側は公開インターフェースを `simd.h` から参照し、実装本体は `simd.lib` としてリンクします。

```powershell
cmake --build build-cmake --config Release --target simd_dist
```

生成される `dist/` には、利用側に必要なファイルだけが配置されます。

```text
dist/
  simd.h
  simd.lib
```

`src/` や `tests/` は含めないため、利用者は実装の詳細を意識せずにライブラリとして組み込めます。

## 構成

```text
include/
  simd.h              公開インターフェース
src/
  simd.cpp            実装
  simd_internal.h     内部型
  simd_low_level.h    ベンチマーク用の低レベル補助API
  simd_debug.h        開発用の診断補助API
tests/
  ベンチマークと検証コード
```

利用側は `include/simd.h` とビルド済みライブラリだけで使えます。`src/` 配下のファイルは、実装を確認したい場合やライブラリを再ビルドする場合に見るためのものです。

## インストール

```powershell
cmake --install build-cmake --config Release --prefix install
```

インストール対象は、利用者向けの `include/simd.h` とライブラリ本体です。`src/simd_internal.h`、`src/simd_low_level.h`、`src/simd_debug.h` は開発・ベンチマーク用の補助ヘッダーなので、通常利用では配布しません。

```cmake
find_package(SIMDLibrary CONFIG REQUIRED)
target_link_libraries(app PRIVATE SIMDLibrary::simd)
```

## 最小サンプル

```cpp
#include "simd.h"

#include <vector>

int main()
{
	rice::simd::Engine engine{};

	rice::simd::Array a{engine.createArray({1.0f, 2.0f, 3.0f, 4.0f})};
	rice::simd::Array b{engine.createArray(a.size(), 2.0f)};
	rice::simd::Array c{engine.createArray({10.0f, 20.0f, 30.0f, 40.0f})};
	rice::simd::Array out{engine.createArray()};

	out.resizeLike(a);
	out = a * b + c;
	engine.execute();

	std::vector<float> result{out.toVector()};
	return 0;
}
```

## 値の入れ方

```cpp
rice::simd::Array a{engine.createArray({1.0f, 2.0f, 3.0f, 4.0f})};

a.fill(1.0f);
a.assign(4, 2.0f);
a.copyFrom(std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f});
a = {5.0f, 6.0f, 7.0f, 8.0f};
a.push_back(9.0f);

std::vector<float> result{};
a.copyTo(result);
```

## コンパイル済みプラン

同じ式を何度も実行する場合は、代入式を一度積んでから `compile()` します。

```cpp
// a, b, c, d, e, x, y, z は同じEngineから作ったArrayです。
// 入力値はfill()、copyFrom()、operator=などで設定済みとします。
x = a * b + c;
y = a * b + d;
z = a * b + e;

rice::simd::ScheduledPlan plan{engine.compile()};

plan.execute();
plan.execute();
```

`compile()` は、それまでに `Array::operator=` で積まれた代入をプラン化し、成功後に内部キューと一時式ノードを空にします。`ScheduledPlan` は配列の内部データを参照するため、プラン実行中は参照先の `Array` を破棄したり `resize()` したりしないでください。

## ベンチマーク

ベンチマークは `tests/` に分けています。

```powershell
ctest --test-dir build-cmake -C Release --output-on-failure
```

ベンチマーク内では手書きSIMD比較のために `simd_low_level.h` を使っていますが、これは利用者向けAPIではありません。通常の利用側は `simd.h` の `Array`、`Engine`、`ScheduledPlan` だけを知っていれば使えます。

