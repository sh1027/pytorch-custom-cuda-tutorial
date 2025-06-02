# Custom CUDAカーネルによる線形フィッティングデモ

このチュートリアルでは，PyTorch上で**自作CUDAカーネル**を用いて，1次関数のフィッティングを行う．  


## ✅ CUDAカーネルの特徴と使い方

CUDAは，C++の拡張として記述され，以下に示す主な特徴を持つ．

- `__global__` 関数として定義

    CUDAでは，GPU上で並列に実行される関数（＝カーネル）を `__global__` 修飾子を付けて定義する．
    ```cpp
    __global__ void linear_fwd_kernel(const float* x, float a, float b, float* y, int N) {
    ...
    }
    ```

- `<<<>>>` 構文で呼び出す

    CUDAの `__global__` 関数は、` <<<grid, block>>> `を使って呼び出す．

    ```cpp
    linear_forward_kernel<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        x.data_ptr<float>(), a, b, y.data_ptr<float>(), N
    );
    ```
    - grid：ブロック数（並列に起動する計算グループ）
    - block：各ブロックのスレッド数
    この例では，N個のデータを`BLOCK_SIZE`のスレッド単位で並列処理している．

## Custom CUDAカーネルをPyTorchに統合するまでの流れ

1. CUDAカーネルの実装  (linear_fitting.cu)
    - forward と backward の CUDA 関数を `__global__` で定義

2. C++ラッパーの実装 (linear_fitting.cu)
    - PyTorch の load 経由で呼び出すために torch::Tensor を受けるラッパー関数を定義
    - PYBIND11_MODULE() でバインド

3. PyTorchの `torch.autograd.Function` を継承したクラスを実装 (linear_fitting.py)
    - `forward` と `backward` メソッドを定義
    - これにより，`loss.backward()` で自作CUDAカーネルが呼び出されるようになる
    ```python
    class LinearFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, a, b):
            ...
        @staticmethod
        def backward(ctx, grad_output):
            ...
    def linear_model(x, a, b):
        return LinearFunction.apply(x, a, b)
    ```

4. gradcheck() で勾配検証
    - `torch.autograd.gradcheck` を使って，自作backwardの勾配が正しいか検証する．

5. 最適化への統合
    - PyTorchの最適化ループに組み込む．
