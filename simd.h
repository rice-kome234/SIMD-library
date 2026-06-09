#pragma once

#include <cstddef>
#include <initializer_list>
#include <memory>
#include <vector>

namespace rice::simd
{
namespace internal
{
struct Compiler;
struct DebugAccess;
struct LowLevelAccess;
struct ArrayData;
struct ScheduledPlanData;
struct EngineData;
}

class Engine;
class Expression;

/*!
 *   @brief SIMD計算に使うfloat配列
 *   @details Arrayは配列の寿命を所有し、同時に式APIで使える変数として振る舞います。
 *            通常はEngine::createArray()から作成し、式にはそのまま使えます。
 *   @warning コンパイル済みプランはArray内部のデータを参照します。
 *            プランを実行する間は、参照先のArrayを破棄したりresize()しないでください。
 */
class Array
{
public:
	/*!
	 *   @brief 配列を破棄
	 */
	~Array() noexcept;

	Array(const Array &) = delete;
	Array(Array &&) noexcept;
	Array &operator=(Array &&) noexcept;

	/*!
	 *   @brief 別の配列を参照する遅延代入を追加
	 *   @param[in] value 代入元の配列
	 *   @return 自分自身への参照
	 *   @details 実際のデータコピーではなく、`this = value` という式をEngineへ追加します。
	 *            即時にデータを複製したい場合はcopyFrom()を使ってください。
	 */
	Array &operator=(const Array &value);

	/*!
	 *   @brief 式の遅延代入を追加
	 *   @param[in] expr 代入する式
	 *   @return 自分自身への参照
	 *   @details
	 * 実際の計算はEngine::execute()、またはコンパイル済みプランのexecute()で行います。
	 */
	Array &operator=(const Expression &expr);

	/*!
	 *   @brief vectorの値を即時コピー
	 *   @param[in] values 入力するfloat配列
	 *   @return 自分自身への参照
	 *   @details values.size()に合わせて配列サイズを変更し、各要素をコピーします。
	 *            式APIの遅延代入ではなく、その場で値を設定します。
	 */
	Array &operator=(const std::vector<float> &values);

	/*!
	 *   @brief 初期化リストの値を即時コピー
	 *   @param[in] values 入力するfloat値
	 *   @return 自分自身への参照
	 *   @details `array = {1.0f, 2.0f, 3.0f};` のように値を設定できます。
	 */
	Array &operator=(std::initializer_list<float> values);

	/*!
	 *   @brief 配列サイズを変更して0初期化
	 *   @param[in] elementCount 新しい論理要素数
	 *   @details 内部の処理単位に合わせた領域を確保し、値は0で初期化します。
	 *   @warning
	 * 既存のコンパイル済みプランがこの配列を参照している場合、そのプランは使わないでください。
	 */
	void resize(std::size_t elementCount);

	/*!
	 *   @brief 別の配列と同じ要素数に変更して0初期化
	 *   @param[in] source サイズの参照元
	 *   @details 出力配列を入力配列と同じ長さにしたい場合に使います。
	 *            既存のコンパイル済みプランがこの配列を参照している場合、そのプランは使わないでください。
	 */
	void resizeLike(const Array &source);

	/*!
	 *   @brief 全要素を同じ値で埋める
	 *   @param[in] value 設定する値
	 *   @details 現在の要素数を変えずに、既存の要素だけを更新します。
	 */
	void fill(float value) noexcept;

	/*!
	 *   @brief サイズを指定して全要素を同じ値にする
	 *   @param[in] elementCount 新しい論理要素数
	 *   @param[in] value 設定する値
	 *   @details std::vector::assign(count, value) に近い使い方ができます。
	 *            既存のコンパイル済みプランがこの配列を参照している場合、そのプランは使わないでください。
	 */
	void assign(std::size_t elementCount, float value);

	/*!
	 *   @brief 別の配列の値を即時コピー
	 *   @param[in] source コピー元の配列
	 *   @details sourceと同じ要素数、同じ値に更新します。式APIの遅延代入ではなく、
	 *            その場で内部データを複製します。
	 */
	void copyFrom(const Array &source);

	/*!
	 *   @brief vectorの値を即時コピー
	 *   @param[in] values 入力するfloat配列
	 *   @details values.size()に合わせて配列サイズを変更し、各要素をコピーします。
	 */
	void copyFrom(const std::vector<float> &values);

	/*!
	 *   @brief 初期化リストの値を即時コピー
	 *   @param[in] values 入力するfloat値
	 *   @details values.size()に合わせて配列サイズを変更し、各要素をコピーします。
	 */
	void copyFrom(std::initializer_list<float> values);

	/*!
	 *   @brief 末尾に1要素を追加
	 *   @param[in] value 追加するfloat値
	 *   @details vectorのpush_back()に近い使い方ができます。
	 *            既存のコンパイル済みプランがこの配列を参照している場合、そのプランは使わないでください。
	 */
	void push_back(float value);

	/*!
	 *   @brief 論理要素数を取得
	 *   @return floatとして扱う要素数
	 */
	std::size_t elementCount() const noexcept;

	/*!
	 *   @brief 論理要素数を取得
	 *   @return floatとして扱う要素数
	 *   @details elementCount()と同じ値を返します。
	 */
	std::size_t size() const noexcept;

	/*!
	 *   @brief 配列の値を通常のfloat配列へコピー
	 *   @param[out] out コピー先のvector
	 *   @details 出力先のvectorはこの配列の要素数に合わせてresizeされます。
	 */
	void copyTo(std::vector<float> &out) const;

	/*!
	 *   @brief 配列の値をvectorとして取得
	 *   @return 通常のfloat配列へコピーした結果
	 */
	std::vector<float> toVector() const;

private:
	friend class Engine;
	friend struct internal::LowLevelAccess;
	friend Expression operator+(const Array &lhs, const Array &rhs);
	friend Expression operator+(const Array &lhs, const Expression &rhs);
	friend Expression operator+(const Expression &lhs, const Array &rhs);
	friend Expression operator+(const Array &lhs, float rhs);
	friend Expression operator+(float lhs, const Array &rhs);
	friend Expression operator-(const Array &value);
	friend Expression operator-(const Array &lhs, const Array &rhs);
	friend Expression operator-(const Array &lhs, const Expression &rhs);
	friend Expression operator-(const Expression &lhs, const Array &rhs);
	friend Expression operator-(const Array &lhs, float rhs);
	friend Expression operator-(float lhs, const Array &rhs);
	friend Expression operator*(const Array &lhs, const Array &rhs);
	friend Expression operator*(const Array &lhs, const Expression &rhs);
	friend Expression operator*(const Expression &lhs, const Array &rhs);
	friend Expression operator*(const Array &lhs, float rhs);
	friend Expression operator*(float lhs, const Array &rhs);
	friend Expression operator/(const Array &lhs, const Array &rhs);
	friend Expression operator/(const Array &lhs, const Expression &rhs);
	friend Expression operator/(const Expression &lhs, const Array &rhs);
	friend Expression operator/(const Array &lhs, float rhs);
	friend Expression operator/(float lhs, const Array &rhs);

	Array(Engine &engine, std::size_t elementCount);
	Engine *engine() const noexcept;

	std::unique_ptr<internal::ArrayData> impl_;
};

/*!
 *   @brief 遅延生成された式ノードへのハンドル
 *   @details Expressionは式木のノードIDを持つ軽量オブジェクトです。
 *            加算や乗算の演算子は、このExpressionを組み合わせて新しい式ノードを作ります。
 *   @warning Engine::compile()またはEngine::execute()の後、その前に作った
 *            Expressionは使わないでください。
 */
class Expression
{
public:
	/*!
	 *   @brief 空の式ハンドルを生成
	 *   @details 主にコンテナや内部キューの初期値として使います。
	 *            通常の式は `a * b + c` のような演算子から生成してください。
	 */
	Expression() noexcept = default;

private:
	friend class Engine;
	friend struct internal::Compiler;
	friend Expression operator+(const Expression &lhs, const Expression &rhs);
	friend Expression operator+(const Array &lhs, const Array &rhs);
	friend Expression operator+(const Array &lhs, const Expression &rhs);
	friend Expression operator+(const Expression &lhs, const Array &rhs);
	friend Expression operator+(const Expression &lhs, float rhs);
	friend Expression operator+(float lhs, const Expression &rhs);
	friend Expression operator+(const Array &lhs, float rhs);
	friend Expression operator+(float lhs, const Array &rhs);
	friend Expression operator-(const Expression &value);
	friend Expression operator-(const Array &value);
	friend Expression operator-(const Expression &lhs, const Expression &rhs);
	friend Expression operator-(const Array &lhs, const Array &rhs);
	friend Expression operator-(const Array &lhs, const Expression &rhs);
	friend Expression operator-(const Expression &lhs, const Array &rhs);
	friend Expression operator-(const Expression &lhs, float rhs);
	friend Expression operator-(float lhs, const Expression &rhs);
	friend Expression operator-(const Array &lhs, float rhs);
	friend Expression operator-(float lhs, const Array &rhs);
	friend Expression operator*(const Expression &lhs, const Expression &rhs);
	friend Expression operator*(const Array &lhs, const Array &rhs);
	friend Expression operator*(const Array &lhs, const Expression &rhs);
	friend Expression operator*(const Expression &lhs, const Array &rhs);
	friend Expression operator*(const Expression &lhs, float rhs);
	friend Expression operator*(float lhs, const Expression &rhs);
	friend Expression operator*(const Array &lhs, float rhs);
	friend Expression operator*(float lhs, const Array &rhs);
	friend Expression operator/(const Expression &lhs, const Expression &rhs);
	friend Expression operator/(const Array &lhs, const Array &rhs);
	friend Expression operator/(const Array &lhs, const Expression &rhs);
	friend Expression operator/(const Expression &lhs, const Array &rhs);
	friend Expression operator/(const Expression &lhs, float rhs);
	friend Expression operator/(float lhs, const Expression &rhs);
	friend Expression operator/(const Array &lhs, float rhs);
	friend Expression operator/(float lhs, const Array &rhs);

	/*!
	 *   @brief エンジンとノードIDを指定して内部用の式ハンドルを生成
	 *   @param[in] engine 式ノードを所有するエンジン
	 *   @param[in] nodeId 内部ノードID
	 */
	Expression(Engine *engine, std::size_t nodeId) noexcept;

	/*!
	 *   @brief 式ノードの所有元エンジンを取得
	 *   @return この式を所有するEngine
	 */
	Engine *engine() const noexcept;

	/*!
	 *   @brief 内部ノードIDを取得
	 *   @return Engine内の式ノードID
	 */
	std::size_t nodeId() const noexcept;

	Engine *engine_{};
	std::size_t nodeId_{static_cast<std::size_t>(-1)};
};

/*!
 *   @brief 依存関係を考慮した複数ステージのプラン
 *   @details ある代入結果を後続の式が読む場合、同じステージにまとめると結果が壊れるため、
 *            ScheduledPlanは安全なステージ順に分割して保持します。
 */
class ScheduledPlan
{
public:
	/*!
	 *   @brief 空のスケジュール済みプランを生成
	 */
	ScheduledPlan();
	~ScheduledPlan() noexcept;

	ScheduledPlan(const ScheduledPlan &) noexcept = default;
	ScheduledPlan &operator=(const ScheduledPlan &) noexcept = default;
	ScheduledPlan(ScheduledPlan &&) noexcept = default;
	ScheduledPlan &operator=(ScheduledPlan &&) noexcept = default;

	/*!
	 *   @brief 全ステージを実行
	 */
	void execute() const noexcept;

	/*!
	 *   @brief ステージ数を取得
	 *   @return 依存関係によって分割されたステージ数
	 */
	std::size_t stageCount() const noexcept;

	/*!
	 *   @brief 命令数の合計を取得
	 *   @return 全ステージの命令数合計
	 */
	std::size_t instructionCount() const noexcept;

	/*!
	 *   @brief 最大一時値数を取得
	 *   @return 全ステージの中で最大の一時値数
	 */
	int maxRegisterCount() const noexcept;

private:
	friend class Engine;
	friend struct internal::Compiler;
	friend struct internal::DebugAccess;

	std::shared_ptr<internal::ScheduledPlanData> impl_;
};

/*!
 *   @brief SIMD配列の作成と式コンパイルを行うエンジン
 *   @details Engineは式ノードの作成、遅延代入、ScheduledPlanへの
 *            コンパイルを担当します。配列そのものの寿命はArrayが所有します。
 *   @warning コンパイル済みプランはArray内部のデータを参照します。
 *            プラン実行中は参照先配列の寿命を呼び出し側で管理してください。
 */
class Engine
{
public:
	/*!
	 *   @brief 空のエンジンを生成
	 */
	Engine();
	~Engine() noexcept;

	Engine(const Engine &) = delete;
	Engine &operator=(const Engine &) = delete;
	Engine(Engine &&) noexcept = delete;
	Engine &operator=(Engine &&) noexcept = delete;

	/*!
	 *   @brief SIMD配列を作成
	 *   @param[in] elementCount 論理的なfloat要素数
	 *   @return このエンジンに紐づいたArray
	 *   @details 配列の寿命は返されたArrayが所有します。
	 *            式APIでは、このArrayをそのまま演算子へ渡せます。
	 */
	Array createArray(std::size_t elementCount = {});

	/*!
	 *   @brief 値を指定してSIMD配列を作成
	 *   @param[in] values 初期値としてコピーするfloat配列
	 *   @return このエンジンに紐づいたArray
	 */
	Array createArray(const std::vector<float> &values);

	/*!
	 *   @brief 初期化リストからSIMD配列を作成
	 *   @param[in] values 初期値としてコピーするfloat値
	 *   @return このエンジンに紐づいたArray
	 */
	Array createArray(std::initializer_list<float> values);

	/*!
	 *   @brief サイズと初期値を指定してSIMD配列を作成
	 *   @param[in] elementCount 論理的なfloat要素数
	 *   @param[in] value 全要素へ設定する値
	 *   @return このエンジンに紐づいたArray
	 */
	Array createArray(std::size_t elementCount, float value);

	/*!
	 *   @brief 遅延代入をコンパイルして実行
	 *   @details Array::operator=で積まれた代入をcompile()でプラン化し、すぐに実行します。
	 */
	void execute();

	/*!
	 *   @brief 遅延代入をコンパイル済みプランへ変換
	 *   @return 依存関係ごとにステージ分割されたScheduledPlan
	 *   @details Array::operator=で積まれた代入をコンパイルし、
	 *            成功後に遅延実行キューと式ノードを空にします。
	 *   @throws std::invalid_argument 代入先や式に含まれる配列の要素数が一致しない場合
	 */
	ScheduledPlan compile();

private:
	friend class Array;
	friend struct internal::Compiler;
	friend Expression operator+(const Expression &lhs, const Expression &rhs);
	friend Expression operator+(const Array &lhs, const Array &rhs);
	friend Expression operator+(const Array &lhs, const Expression &rhs);
	friend Expression operator+(const Expression &lhs, const Array &rhs);
	friend Expression operator+(const Expression &lhs, float rhs);
	friend Expression operator+(float lhs, const Expression &rhs);
	friend Expression operator+(const Array &lhs, float rhs);
	friend Expression operator+(float lhs, const Array &rhs);
	friend Expression operator-(const Expression &value);
	friend Expression operator-(const Array &value);
	friend Expression operator-(const Expression &lhs, const Expression &rhs);
	friend Expression operator-(const Array &lhs, const Array &rhs);
	friend Expression operator-(const Array &lhs, const Expression &rhs);
	friend Expression operator-(const Expression &lhs, const Array &rhs);
	friend Expression operator-(const Expression &lhs, float rhs);
	friend Expression operator-(float lhs, const Expression &rhs);
	friend Expression operator-(const Array &lhs, float rhs);
	friend Expression operator-(float lhs, const Array &rhs);
	friend Expression operator*(const Expression &lhs, const Expression &rhs);
	friend Expression operator*(const Array &lhs, const Array &rhs);
	friend Expression operator*(const Array &lhs, const Expression &rhs);
	friend Expression operator*(const Expression &lhs, const Array &rhs);
	friend Expression operator*(const Expression &lhs, float rhs);
	friend Expression operator*(float lhs, const Expression &rhs);
	friend Expression operator*(const Array &lhs, float rhs);
	friend Expression operator*(float lhs, const Array &rhs);
	friend Expression operator/(const Expression &lhs, const Expression &rhs);
	friend Expression operator/(const Array &lhs, const Array &rhs);
	friend Expression operator/(const Array &lhs, const Expression &rhs);
	friend Expression operator/(const Expression &lhs, const Array &rhs);
	friend Expression operator/(const Expression &lhs, float rhs);
	friend Expression operator/(float lhs, const Expression &rhs);
	friend Expression operator/(const Array &lhs, float rhs);
	friend Expression operator/(float lhs, const Array &rhs);

	Expression makeVariable(const Array &value);
	Expression makeScalar(float value);
	Expression makeAdd(const Expression &lhs, const Expression &rhs);
	Expression makeSub(const Expression &lhs, const Expression &rhs);
	Expression makeMul(const Expression &lhs, const Expression &rhs);
	Expression makeDiv(const Expression &lhs, const Expression &rhs);
	void deferAssign(Array &out, const Expression &expr);

	std::unique_ptr<internal::EngineData> impl_;
};

/*!
 *   @name スカラー式演算子
 *   @brief 要素ごとのfloat演算を表す遅延式ノードを作成します。
 *   @details 実際の計算はこの時点では行わず、式木としてEngineへ登録します。
 *   @{
 */
Expression operator+(const Expression &lhs, const Expression &rhs);
Expression operator+(const Array &lhs, const Array &rhs);
Expression operator+(const Array &lhs, const Expression &rhs);
Expression operator+(const Expression &lhs, const Array &rhs);
Expression operator+(const Expression &lhs, float rhs);
Expression operator+(float lhs, const Expression &rhs);
Expression operator+(const Array &lhs, float rhs);
Expression operator+(float lhs, const Array &rhs);

Expression operator-(const Expression &value);
Expression operator-(const Array &value);
Expression operator-(const Expression &lhs, const Expression &rhs);
Expression operator-(const Array &lhs, const Array &rhs);
Expression operator-(const Array &lhs, const Expression &rhs);
Expression operator-(const Expression &lhs, const Array &rhs);
Expression operator-(const Expression &lhs, float rhs);
Expression operator-(float lhs, const Expression &rhs);
Expression operator-(const Array &lhs, float rhs);
Expression operator-(float lhs, const Array &rhs);

Expression operator*(const Expression &lhs, const Expression &rhs);
Expression operator*(const Array &lhs, const Array &rhs);
Expression operator*(const Array &lhs, const Expression &rhs);
Expression operator*(const Expression &lhs, const Array &rhs);
Expression operator*(const Expression &lhs, float rhs);
Expression operator*(float lhs, const Expression &rhs);
Expression operator*(const Array &lhs, float rhs);
Expression operator*(float lhs, const Array &rhs);

Expression operator/(const Expression &lhs, const Expression &rhs);
Expression operator/(const Array &lhs, const Array &rhs);
Expression operator/(const Array &lhs, const Expression &rhs);
Expression operator/(const Expression &lhs, const Array &rhs);
Expression operator/(const Expression &lhs, float rhs);
Expression operator/(float lhs, const Expression &rhs);
Expression operator/(const Array &lhs, float rhs);
Expression operator/(float lhs, const Array &rhs);
/*! @} */

}
