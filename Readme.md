# pt5

C++/OptiX7で書かれたパストレーサーであり、そのPython3のラッパ、またBlenderのアドオンです。
リポジトリの構成はは次の通りです。

* host	C++から利用するサンプル
* interface python3 への公開
* libpt5 関数等の本体
* pt5 Blenderアドオン
* pt5/blender Blnder用モジュール
* pt5/core python3から利用できるモジュール


## 要求 (tested on)

* C++17
* CUDA 11.4
* OptiX 7.3 （`OptiX_INSTALL_DIR` を設定してください。）
* Python3.9 （Blender とバージョンを揃えてください。）


## libpt5の内容

host/main.cpp はレンダリングの経過をウィンドウに表示するプログラムで、このライブラリの具体的な使われ方を見ることができます。


### PathTracerState

シーン（Scene）を受け取り出力先（View）へ画像をレンダリングします。


### Scene

シーンデータです。
トライアングルメッシュ、メッシュの各面に適用されるマテリアル、テクスチャ等で使われる画像から構成されます。
マテリアルはシェーダノードの木です。
表面用のノードとして Diffuse, Emission, Mix, 画像テクスチャを実装済みです。


### View

レンダリングされる画像の書き出し先を管理します。
ウィンドウ等に出力を表示したい場合のために、画像をOpenGLのテクスチャにコピーする機能も持ちます。
