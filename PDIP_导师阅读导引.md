# PDIP 导师阅读导引

本导引用于快速理解 `regot` 中 PDIP-CG / PDIP-FP 的代码组织与阅读顺序。

## 1. 建议阅读顺序

1. `src/pdip_result.h`  
   先看统一输出字段语义：`niter`, `plan`, `obj_vals`, `mar_errs`, `run_times`。
2. `src/pdip_solvers.h`  
   看可调参数默认值和含义，理解 CG/FP 路径控制。
3. `src/pdip_cg.cpp`  
   看 CG 主循环分阶段流程（残差 -> 线性系统 -> 校正步 -> 收敛）。
4. `src/pdip_fp.cpp`  
   看 FP 主循环分阶段流程和分段耗时统计点。
5. `src/pdip_dense_chol.cpp`  
   看 Cholesky 后端（LAPACK / Eigen）实现细节。
6. `src/solvers.cpp`  
   看 Python kwargs 如何映射到 C++ 求解器参数。
7. `benchmarks/run_compare_pdip_qrot_4datasets.py`  
   看实验入口、输出 CSV 字段定义、诊断文件解释。

## 2. 主循环阶段映射

- CG (`pdip_cg_internal`)
  - 阶段0：参数与容器初始化
  - 阶段1：初始点构造
  - 阶段2：外层迭代
    - 2.1 残差与缩放量
    - 2.2 构建预条件矩阵
    - 2.3 predictor 方向
    - 2.4 校正步
    - 2.5 收敛判定与历史记录

- FP (`pdip_fp_internal`)
  - 阶段0：参数与容器初始化
  - 阶段1：初始可行点构造
  - 阶段2：外层迭代
    - 2.1 残差与缩放量
    - 2.2 构建牛顿系统矩阵（B1/B2 路径）
    - 2.3 Cholesky 分解（含 fallback）
    - 2.4 predictor 方向
    - 2.5 校正步
    - 2.6 收敛判定

## 3. 等价性说明（对比改动前）

- 未改变算法输出协议：字段名、字段含义、单位保持一致。
- 未改变收敛判据与目标值/边际误差口径。
- 可读性优化主要是：
  - 增加模块头注释与阶段注释；
  - 抽取重复的分解回退逻辑；
  - 统一命名与脚本结果解释注释。

## 4. 性能与正确性回归

- 单元测试：`python -m unittest tests/test_pdip_api.py -v`（PDIP-CG / PDIP-FP 接口与基本精度）。
- 四数据集对比：运行 `benchmarks/run_compare_pdip_qrot_4datasets.py`；`summary.csv` 与图默认写入 `--out-dir`（仓库内默认路径已列入 `.gitignore`，需自行生成）。

