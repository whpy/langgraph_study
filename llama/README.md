
The commands to install llama-cpp-python:
```shell
CMAKE_ARGS=-DGGML_CUDA=on -DLLAVA_BUILD=off
pip install -U llama-cpp-python --force-reinstall --no-cache-dir
```
The commands are not normal as we encounter the error like:
```/lib64/libcudart.so.11.0: undefined reference to `dlvsym@GLIBC_2.2.5'```