本文件夹为「可直接用于 Git / GitHub」的精简拷贝（约 1.5MB 量级）。

已排除（勿再提交进仓库）：
  - eigen-5.0.1/     （由 setup.py 下载或设置环境变量 EIGEN3_INCLUDE_DIR）
  - data/            （MNIST 等数据集，请本机自备或从脚本/ot.datasets 获取）
  - build、*.so、venv、benchmark 运行结果等

使用方式建议：
  1) 若已有 clone 的 Fork：把本目录内文件覆盖到该仓库根目录（保留其中的 .git），再
     git add -A && git commit && git push
  2) 若从零：在本目录执行 git init，添加 remote 后 push。

安装与测试：
  pip install -e .
  python -m unittest tests/test_pdip_api.py -v
