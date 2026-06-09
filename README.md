# SIMD-library

`SIMD-library` は、`float` 配列に対する要素ごとの計算を、通常の式に近い書き方でSIMD実行するための小さなC++ライブラリです。利用側は基本的に `simd.h` だけを読み、ビルド済みの `simd.lib` をリンクします。

## 特徴

- `rice::simd::Array` が配列の寿命を所有します。
- `rice::simd::Engine` が式ノード生成と実行を担当します。
- `x = a * b + c;` のように、計算式をそのまま書けます。
- 出力配列が空の場合は、式に含まれる入力配列のサイズへ自動で合わせます。
- 配列サイズの不一致は `execute()` のタイミングで検出します。

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

## 最小サンプル

```cpp
#include "simd.h"

#include <vector>

int main()
{
	rice::simd::Engine engine{};

	rice::simd::Array a{engine, {1.0f, 2.0f, 3.0f, 4.0f}};
	rice::simd::Array b{engine, a.size(), 2.0f};
	rice::simd::Array c{engine, {10.0f, 20.0f, 30.0f, 40.0f}};
	rice::simd::Array out{engine};

	out = a * b + c;
	engine.execute();

	std::vector<float> result{out.toVector()};
	return 0;
}
```

## 値の入れ方

```cpp
rice::simd::Array a{engine, {1.0f, 2.0f, 3.0f, 4.0f}};

a.fill(1.0f);
a.assign(4, 2.0f);
a.copyFrom(std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f});
a = {5.0f, 6.0f, 7.0f, 8.0f};
a.push_back(9.0f);

std::vector<float> result{};
a.copyTo(result);
```

## 計算の実行

代入式を書いた時点では、計算はまだ実行されません。
複数の式を積んでから `engine.execute()` を呼ぶと、まとめて処理されます。

```cpp
// a, b, c, d, e, x, y, z は同じEngineから作ったArrayです。
// 入力値はfill()、copyFrom()、operator=などで設定済みとします。
x = a * b + c;
y = a * b + d;
z = a * b + e;

engine.execute();
```

`x`、`y`、`z` が空の配列だった場合は、入力配列の要素数に合わせて自動で確保されます。
すでにサイズを持っている出力配列は、そのサイズが入力配列と一致しているかを実行時に確認します。

## ベンチマーク

ベンチマークは `tests/` に分けています。

```powershell
# ベンチマーク結果の標準出力を表示
ctest --test-dir build-cmake -C Release --output-on-failure -V
```

ベンチマーク内では手書きSIMD比較のために `simd_low_level.h` を使っていますが、これは利用者向けAPIではありません。通常の利用側は `simd.h` の `Array`、`Engine`、式演算子だけを知っていれば使えます。

