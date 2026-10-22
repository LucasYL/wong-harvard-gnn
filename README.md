# Wong → Harvard: assay-aware antibacterial GNN

基于 Wong 公开实验数据训练分子图神经网络，用 Harvard / PubChem **AID 1259311** 的另一套 *Staphylococcus aureus* RN4220 实验作外部回顾性排序评价。

**English overview:** A research codebase for structure-based antibacterial ranking with Chemprop D-MPNNs, fixed molecular descriptors, Morgan-RF and nearest-neighbour controls. Models developed on the Wong RN4220 assay were evaluated on an audited external Harvard RN4220 screen. This repository preserves the completed experiment and provides a starting point for collaborative development; it is not an exact reproduction of the Wong paper.

**交接状态（2026-10-22）：** 原实验已完成；本仓库整理代码、冻结协议、报告及可下载的历史模型/数据。Harvard 的结果已经查看，后续不能再把它称为未见测试集。本次交接没有启动新训练。

## 目录导航

- [研究问题与范围](#研究问题与范围)
- [已有结果](#已有结果)
- [数据、标签与划分](#数据标签与划分)
- [模型与训练](#模型与训练)
- [安装与快速开始](#安装与快速开始)
- [下载历史模型与数据](#下载历史模型与数据)
- [核验已经保存的结果](#核验已经保存的结果)
- [重新开发与训练](#重新开发与训练)
- [Harvard 评价流程](#harvard-评价流程)
- [文件布局](#文件布局)
- [协作与下一步](#协作与下一步)
- [限制、来源与许可](#限制来源与许可)

## 研究问题与范围

核心问题：只有一个分子的明确结构时，模型能否把在指定 whole-cell assay 中有活性的化合物排在前面，帮助安排有限的实验预算？

科学单位始终是 **compound + assay context → measurement**。同样是 RN4220，不同实验条件、浓度、读数归一化和标签规则仍可能对应不同任务。本项目输出的是排序分数，没有证明它是任意实验条件下已校准的活性概率。

这一仓库覆盖：

- Wong 数据下载、逐行实验数据提取、结构标准化、固定 scaffold 划分。
- Morgan/Tanimoto 近邻、Morgan 指纹随机森林、轻量 Chemprop D-MPNN。
- 加入 200 维固定归一化 RDKit 描述符及多 seed 等权集成。
- CO-ADD 和 SyntheMol 的辅助迁移比较，以及相关数据审计。
- Harvard RN4220 外部数据去重、暴露排查、冻结预测、AP/Top-K 和独立核验。

保留了早期 CheMeleon 冻结 encoder 与 V0 final-test 的代码和摘要，作为历史上下文。主发布数据包围绕 Wong→Harvard 实验，**不包含 CheMeleon 权重/嵌入和全部早期 V0 输出**。后续四菌种训练、NPASS 汇总、天然产物候选筛选、演示材料不属于本仓库的交接范围。

## 已有结果

### Harvard 外部历史评价

所有方法在同一份严格队列上计分：**225,829 个分子，601 个活性，活性率约 0.266%**。AP 为 sklearn 非插值 average precision。

- **近邻：** AP **0.037752**；Top-20/50/100 hits = **11 / 17 / 34**。
- **Morgan-RF，3 个 seed 均值：** AP **0.065656**；Top-20/50/100 = **16 / 30 / 39**。
- **Wong-only D-MPNN + descriptors，3 个 seed 均值：** AP **0.087054**；Top-20/50/100 = **20 / 38 / 44**。
- **CO-ADD→Wong，3 个 seed 均值：** AP **0.084055**；Top-20/50/100 = **17 / 37 / 45**。
- **SyntheMol→Wong，3 个 seed 均值：** AP **0.082095**；Top-20/50/100 = **18 / 38 / 44**。

主 GNN 的 Precision@50 = 38/50 = **76%**，找回全部 601 个活性中的约 **6.3%**。因此很靠前的一小批候选富集明显，但不代表模型找回了多数活性。AP 衡量完整排序；它不是准确率，也不是 76%。近邻 Top-20 的确定性排序有边界同分，按随机打破同分计算的期望 hits 为约 10.43；报告同时保存两种口径。

不确定性采用 **2,000 次 Murcko scaffold 分组、配对 bootstrap**。主 GNN 减 RF 的 AP 差约 **+0.0214**，95% 区间 **[+0.0046, +0.0384]**。迁移模型相对 Wong-only GNN 的 AP 差区间均跨零，没有建立明确的迁移收益。这些区间条件于已固定模型，不包含重新训练的不确定性。

**RF 是查看最初 Harvard 结果后补加的历史对照。** RF 权重在此前已经训练保存，没有用 Harvard 调参；这一比较仍应标为补充比较，不能包装成原先完全预定的盲测。

数字来源：

- [原四组评价 JSON](reports/harvard_rn4220_model_comparison_v1.json)
- [RF 补充比较 JSON](reports/harvard_morgan_rf_comparison_v1.json)
- [中文结果解释](harvard_external_results_zh.md)
- [RF 比较的设计边界](harvard_morgan_rf_comparison_zh.md)

### Wong 开发集与早期 test

固定 Wong validation 有 **3,821 个分子、60 个 hits**。匹配的 3 个 seed 集成：不加描述符 AP **0.3713**，加入描述符 AP **0.4125**；两者 Top-50 都为 **22**。10 成员描述符集成 AP **0.4150**、Top-50 为 **20**，没有建立扩大集成对固定 Top-50 预算的收益。

这些 validation 结果参与了 checkpoint/方案选择，属于开发结果。Harvard 主比较固定使用 seeds **17、29、43**，并非事后从 10 个成员挑选。

更早的 V0 test 有 **3,944 个分子、42 hits**，历史结果保存在 [V0 test 摘要](reports/final_test_comparison.json)。V0 与后来的描述符实验不是同一批模型；不能把 V0 scratch 指标当作描述符 GNN 的最终 test 成绩。V0 test 与 Harvard 都已查看，今后的重要结论需要新的锁定评价。

## 数据、标签与划分

### Wong 训练任务

来源为 Wong 等人的 *Discovery of a structural class of antibiotics with explainable deep learning*，以及作者 [antibioticsai 仓库](https://github.com/felixjwong/antibioticsai)。下载固定到 commit `83403d483def9aa6a90df850b3d0454b3030cf47`，原始文件 SHA-256 见 [source_manifest.json](reports/source_manifest.json)。

作者输入表共有 **39,312** 条记录；本项目经结构和标签审计后形成 **38,823** 个可用母体化合物，共 **484** 个 hits：

- train：**31,058 / 382 hits**。
- validation：**3,821 / 60 hits**。
- historical test：**3,944 / 42 hits**。

任务沿用作者的 RN4220 **50 μM** 标签：平均相对生长 **< 0.2** 为 hit。实验数值、重复测量、原始 SMILES、来源 sheet/行均保留；缺失/非法测量不会自动变成 inactive。阈值附近、重复测量跨越阈值的样本作标记，不擅自修改标签。

结构处理：RDKit Cleanup → FragmentParent → Uncharger；保留立体化学，不额外做互变异构标准化。非法结构、多有机片段、同母体标签冲突和身份歧义进入审查，不能靠多数投票或静默删除掩盖冲突。

划分使用非立体 Bemis–Murcko scaffold 整组分配，split seed 为 **20260908**，目标比例 80/10/10。无环结构归入同一组；根据 seed 与 scaffold 的 SHA-256 排序确定分配，不按标签挑选 split。检查 parent identity、连接结构、InChIKey 与 scaffold 跨集合重叠。

### Harvard 外部任务

“Harvard dataset” 在这里具体指 [PubChem BioAssay AID 1259311](https://pubchem.ncbi.nlm.nih.gov/bioassay/1259311)，不是 Wong 的保留 test。

- 原始审计记录 **329,176** 条。
- 仅排除身份/连接重叠后的较宽清单为 **314,118** 个分子、**922** 个活性；它**没有**用于本次主评价。
- 进一步排除 scaffold 重叠后的主评价队列为 **225,829** 个分子、**601** 个活性。

保护集合包含 Wong train/validation/test、CO-ADD train/validation、SyntheMol train/validation/external，共 **8 个分区**。这些排查只针对本项目声明的数据暴露，不能证明不存在任何外部预训练暴露。

沿用该 assay 的 deposited binary labels，并核查两个重复 OD600 读数及 plate Z-score 的原始 hit 规则。**最终化合物浓度未在已下载表中明确**；不能将它写成 Wong 50 μM 的复现，也不能把结果解释成 MIC 或杀菌性预测。完整审计见 [external audit](reports/harvard_rn4220_external_audit_v1.json)。

### 迁移数据

CO-ADD 是单独的 MRSA 测试背景，源任务使用连续抑制读数；SyntheMol 另有自己的结构去重、数据划分与实验标签。迁移后都在 Wong 目标任务上训练/选择，然后固定用于 Harvard。这里没有把不同 assay 的标签直接拼成一个通用“抗菌”标签。

已保存的 SyntheMol external 结果可作历史参考，但源数据的其他分子参与过相关迁移训练，不能把整套来源都叫完全未见。参考 [CO-ADD 迁移说明](transfer_experiment.md)、[SyntheMol 迁移说明](synthemol_transfer_experiment.md)。

## 模型与训练

主模型是 Chemprop **2.2.1** 的 directed message-passing neural network：

1. 标准化 SMILES 经 RDKit 构造成分子图。
2. 有向键消息传递，hidden dimension **300**、depth **3**。
3. mean pooling 得到分子表示。
4. 拼接 `descriptastorus.RDKit2DNormalized` 的 **200 维**描述符。
5. 一个隐藏层的 FFN 输出二分类排序分数。

描述符来自结构和库内固定参考归一化，不在本项目全数据上重新拟合 scaler。非有限值按冻结规则置 0 并逐项留痕；特征计算失败则停止。

训练使用不加类别权重的 BCE，dropout 0、batch size 32、最多 20 epochs、validation AP early stopping。默认 CPU 两线程；每个 seed 使用相同 split。seeds 反映初始化和训练顺序的变化，不代表换了一份独立数据集。主外部比较采用 17/29/43 的等权平均。

传统基线包括：

- radius-2、2048-bit、保留手性的 Morgan 指纹。
- Tanimoto 相似度加权的 5 近邻，第 5 名同分的邻居全部纳入。
- RF：200 棵树，max depth 20，min samples leaf 2，`balanced_subsample`。

Top-K 按 score 降序、compound key 升序固定同分次序，并另外报告边界同分的期望 precision/enrichment。AP 与梯形 PR-AUC 分开命名，不能混为同一个数。

本模型比 Wong 论文模型轻量；没有完整复现作者的超参数搜索、ensemble、解释算法、毒性模型或实验流程。不要直接用两个项目不同数据/协议上的指标宣称超过论文。

## 安装与快速开始

原环境为 **Python 3.12 / macOS Apple Silicon / CPU**。其他系统可以尝试，但没有完成跨平台或逐位一致验证。

```bash
git clone https://github.com/LucasYL/wong-harvard-gnn.git
cd wong-harvard-gnn
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock.txt
PYTHONPATH=src .venv/bin/python -m pytest -q
```

基础环境支持数据处理、传统基线及结果核验。神经网络使用独立环境：

```bash
python3.12 -m venv .venv-neural
.venv-neural/bin/python -m pip install -r requirements-neural.lock.txt
PYTHONPATH=src .venv-neural/bin/python -m pytest -q
```

`requirements*.txt` 是直接依赖，`requirements*.lock.txt` 保存原完整版本。描述符依赖 `descriptastorus==2.8.0` 已固定在 neural lock 中。不要为解决单个安装问题直接升级 RDKit、Chemprop 或指标库后覆盖原实验；先另建环境并记录变化。Jupyter 是可选工具，不是运行 CLI 的必要条件。

最短上手路径：安装基础环境 → 下载并恢复 release 数据包 → 校验文件哈希 → 重算已保存的 Harvard 指标。无需先训练模型。

## 下载历史模型与数据

大文件放在私有仓库的 [v1.0.0-harvard 发布页](https://github.com/LucasYL/wong-harvard-gnn/releases/tag/v1.0.0-harvard)，避免把权重和逐分子预测塞进 Git 历史。需要仓库访问权限及 GitHub CLI 登录。

```bash
gh auth login
mkdir -p artifacts
gh release download v1.0.0-harvard \
  --repo LucasYL/wong-harvard-gnn \
  --dir artifacts \
  --pattern 'wong-harvard-artifacts-v1.zip' \
  --pattern 'SHA256SUMS'
(cd artifacts && shasum -a 256 -c SHA256SUMS)
python3 -m zipfile -e artifacts/wong-harvard-artifacts-v1.zip .
python3 scripts/verify_release_artifacts.py
```

**在干净 clone 中解压一次。** 解压命令会写入归档中的 `data/` 和 `outputs/`；如果已有自己的实验输出，先使用新的 clone，避免覆盖。

数据包包含：

- Wong 原始下载快照、提取记录、固定 train/validation/test 与描述符缓存。
- Harvard 原始分批表、concise 表、审计记录、严格和较宽队列。
- CO-ADD、SyntheMol 的相关源文件/处理结果和 8 个保护分区。
- 主描述符 GNN、两组迁移模型的 checkpoint、训练记录和 validation 预测。
- 近邻/RF、Harvard 全量成员及集成预测、分块和封存记录。
- 与主实验有关的已有 SyntheMol external 输出。

精确内容和每个文件的 SHA-256 见 [artifact_manifest.json](docs/artifact_manifest.json)。`data/`、`outputs/`、环境、日志和下载包均不进入 Git。归档记录中的旧 PID/机器信息只是历史运行记录，不表示现在有任务在运行。

## 核验已经保存的结果

先做无需科学依赖的逐文件完整性检查，再用基础环境独立重算 AP、ROC-AUC、13 组预测的 ensemble 均值、39 项 Top-K 和同分期望：

```bash
python3 scripts/verify_release_artifacts.py
PYTHONPATH=src .venv/bin/python scripts/verify_harvard_evaluation_v1.py
```

第二条会重写 `outputs/harvard-rn4220-evaluation-v1/verification.json` 为确定性核验回执，不训练、不重跑全量预测。保存的 RF 补充预测也可独立重算：

```bash
PYTHONPATH=src .venv/bin/python scripts/verify_harvard_rf_saved.py
```

需要确认原九个 GNN checkpoint 仍能生成保存的分数时，在 neural 环境运行 **64 个旧 Wong validation 分子**的复核：

```bash
PYTHONPATH=src .venv-neural/bin/python scripts/reproduce_saved_validation.py
```

它检查固定协议、重算描述符、九成员和近邻的旧 validation 分数；不读取 Harvard 标签做模型选择，也不启动训练。跨平台数值或依赖漂移可能触发冻结校验，应调查具体差异，不应删除断言掩盖。

## 重新开发与训练

### 从公开 Wong 原始文件开始

在**未恢复 release 数据包的干净 clone**中运行：

```bash
.venv/bin/python scripts/download_wong.py
.venv/bin/python scripts/extract_wong.py
PYTHONPATH=src .venv/bin/python -m antibacterial.prepare --limit 1000
PYTHONPATH=src .venv/bin/python -m antibacterial.prepare
```

下载器验证固定源文件哈希，已有原文件只验证、不覆盖。重新准备数据的清单可能包含新的运行时间、环境信息，其哈希未必与历史清单相同；“从头构建新运行”和“核验原冻结运行”要分开记录。

### 新的基线与轻量神经网络

```bash
PYTHONPATH=src .venv/bin/python -m antibacterial.baseline \
  --pilot --run-id local-baseline-pilot-001
PYTHONPATH=src .venv/bin/python -m antibacterial.baseline \
  --run-id local-baseline-001
PYTHONPATH=src .venv-neural/bin/python -m antibacterial.neural \
  --pilot --seed 17 --run-id local-gnn-pilot-001
PYTHONPATH=src .venv-neural/bin/python -m antibacterial.neural \
  --seed 17 --run-id local-gnn-seed17-001
```

每次使用新的 run ID；先跑 pilot 并查看时间/内存，再决定完整运行。以上代码只计 Wong validation，不创建用于开发选择的 test loader。

### 描述符 GNN

在新准备数据的 clone 中，先建立描述符缓存：

```bash
PYTHONPATH=src .venv-neural/bin/python -m antibacterial.descriptors --pilot
PYTHONPATH=src .venv-neural/bin/python -m antibacterial.descriptors
PYTHONPATH=src .venv-neural/bin/python -m antibacterial.neural_descriptors \
  --pilot --seed 17 --run-id local-descriptor-pilot-001
PYTHONPATH=src .venv-neural/bin/python -m antibacterial.neural_descriptors \
  --seed 17 --run-id local-descriptor-seed17-001
```

29、43 使用不同 run ID 分别运行，等权平均，不看 test 选 seed。原三成员/十成员 batch 配置与恢复说明见 [descriptor_experiment.md](descriptor_experiment.md)。

`descriptor_batch`、迁移 runner、旧 final-evaluation 和 Harvard 的 `freeze` 入口是原实验编排器，带有固定路径/哈希/拒绝覆盖规则。**不要把 README 中的新 run ID 当成已经替换了这些编排器的依赖。** 新一轮完整多模型比较应在新分支中设计版本化配置和输出前缀，再重新预声明协议。

### 成本和进度

历史 Apple Silicon 上，轻量 GNN 每个约 13–20 分钟，10 个描述符成员原队列约 137 分钟；Harvard 全量九成员推理原先预留约 1.5–2.5 小时。它们只是旧机器上的参考，不是新环境性能保证。默认两线程、batch 32、单进程预算 3 GiB；无需云资源。

epoch/checkpoint/`results.json` 位于 `outputs/<run-id>/`。查看进度应结合 PID、日志和文件更新时间；历史 status 文件不能证明任务正在运行。macOS 的 `caffeinate` 可防普通闲置睡眠，但不能保证合盖后继续计算。附带的 [GitHub Actions 模板](docs/ci/tests.yml)只跑小型测试。当前上传令牌缺少 `workflow` scope，因此没有启用云端 CI；需要具有该权限的维护者将模板放到 `.github/workflows/tests.yml` 后才能启用。模板不会启动完整训练或外部批量评价。

## Harvard 评价流程

保存的流程依次是：

1. 从 PubChem 提取明确 assay 和 source SID，保存原始分批记录及哈希。
2. 核查原读数、作者标签、结构、重复记录和冲突；排除 8 个保护分区的重叠。
3. 在旧 Wong validation 的固定小样本上复现已有分数、测资源。
4. 固定队列、九个 checkpoint、指标、种子、bootstrap、同分规则和代码哈希。
5. 只把 query 结构送入 forward，分块保存并封存所有预测。
6. 封存后按稳定 compound key 连接已知实验标签并计分。
7. 从保存 CSV 独立重算，再增加明确标注为 post-hoc 的已有 RF 对照。

主要入口为 `download_rn4220_external_v1.py` → `audit_rn4220_external_v1.py` → `verify_rn4220_external_v1.py` → `evaluate_harvard_rn4220_v1.py` → `verify_harvard_evaluation_v1.py`。下载器依赖归档中的原 concise SID 清单；只克隆代码还不能从它一键恢复完整 Harvard 原始快照。原始历史文件已经在 release 中保留。

原历史 `pilot`、`freeze`、`run` 入口会拒绝覆盖已经存在的结果。后续需要再运行推理时，先创建版本化路径和新协议；不要删除原结果来绕过保护。把新结果作为在已查看 benchmark 上的比较，另找未参与开发的新队列作最终证据。

## 文件布局

```text
README.md                         项目入口、结果与复现路线
CONTRIBUTING.md                   分支、PR、实验记录和共同开发约定
THIRD_PARTY_NOTICES.md             来源、引用和许可边界
configs/                          固定模型配置与历史冻结协议
src/antibacterial/                标准化、划分、基线、D-MPNN、描述符
scripts/                          数据来源、迁移、外部评价和核验 CLI
tests/                            化学身份、泄漏、指标和神经数据流契约
reports/                          小型审计、比较和独立核验摘要
docs/artifact_manifest.json        发布大文件清单及 SHA-256
docs/export_manifest.json          从研究工作区导出的文件溯源
docs/ROADMAP.md                    合作起点与尚未完成的工作
.github/                          issue/PR 模板
docs/ci/tests.yml                 尚未启用的轻量 GitHub Actions 模板
data/                            通过 release 恢复；Git 忽略
outputs/                         历史权重和预测；Git 忽略
artifacts/                       下载 zip 与校验文件；Git 忽略
```

`link_npass_domain_v1.py` 随仓库保留，是因为 Harvard 审计调用其中 `protected_sets()` 读取 8 个分区；其独立 NPASS 主流程不属于本次数据包。旧 CheMeleon/V0 入口需要其额外权重或输出，不能仅凭主包宣称全部历史分支均可运行。

## 协作与下一步

`LucasYL` 与受邀协作者 `andyyyyaa` 使用同一私有仓库。收到邀请后，先接受 GitHub repository invitation，再克隆、下载 release、运行上面的核验。邀请未接受前，私有链接可能显示 404。

建议从这些小任务开始：

1. 在协作者机器上验证安装、数据包和保存结果，记录平台差异。
2. 把固定路径的历史编排器逐步改造成显式 config/run-id 的新版本，保留原文件。
3. 在查看新结果前定义下一份独立队列、assay 可比性、暴露排查及固定 Top-K。
4. 讨论需要优化 AP、固定预算 Top-K，还是不同 assay 的可迁移性；先确定目标再改模型。

详见 [贡献指南](CONTRIBUTING.md) 和 [合作路线](docs/ROADMAP.md)。改动通过分支和 PR；涉及实验结论要一并附配置、数据角色、代码 commit、指标来源和未解决的限制。不要把 proposed/running/completed/verified 写成同一种状态。

## 限制、来源与许可

- 这是结构已知纯化合物的计算排序，不能直接用于未知结构、提取物、菌株或 BGC 的活性判断。
- scaffold 排重不等于低相似度，更不等于新化学类别或新作用机制。
- Harvard 浓度仍不明确，来源中的天然产物/放线菌身份不应从名字或高分猜测。
- 结果不是 MIC、临床效果、毒性安全性、广谱活性或新抗生素发现的证据。
- 不同任务的 AP 受样本组成和活性率影响，不能把 Wong AP 与 Harvard AP 直接比较成退步/进步。
- 归档用于私人研究协作；没有为第三方数据另行授予再分发许可。公开发布前需要逐来源确认条款。本仓库原创代码目前也未另行选择开源许可证。

主要来源：

- [Wong et al., Nature, DOI: 10.1038/s41586-023-06887-8](https://doi.org/10.1038/s41586-023-06887-8)
- [Wong 作者代码与公开数据](https://github.com/felixjwong/antibioticsai)
- [Harvard / PubChem BioAssay AID 1259311](https://pubchem.ncbi.nlm.nih.gov/bioassay/1259311)
- [Chemprop](https://github.com/chemprop/chemprop)
- [descriptastorus](https://github.com/bp-kelley/descriptastorus)
- [CO-ADD](https://www.co-add.org/)

研究数据保持各自来源和使用条件；详细说明与保留的 Wong 上游 MIT 文本见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。引用本仓库时，请记录 commit、release tag，并同时引用使用的数据源和方法。
