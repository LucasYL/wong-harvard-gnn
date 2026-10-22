# 预训练模型候选：接入前核查

更新：2026-09-09。CheMeleon 的 Wong validation 首轮比较已完成；NaFM 仍为保留候选，放线菌来源域尚未验证。

## 第一候选：CheMeleon 冻结 encoder

- 选择理由：已经可以使用当前 Chemprop 环境；作者公开预训练结构列表，便于核对数据暴露。先提取固定 embedding，再训练简单分类器，符合本项目先测算力的顺序。
- 预训练目标：作者报告为约 100 万个 PubChem 分子的计算所得 Mordred descriptors，而非 Wong 抗菌活性。见[方法](https://arxiv.org/html/2506.15792v2#S3.SS1)。这是依据作者方法记录的标签来源，尚未独立重建其完整预训练过程。
- [固定权重版本](https://zenodo.org/records/15460715)：`chemeleon_mp.pt`，34,859,448 bytes，MD5 `6a80b54fdb7de37ef0374d302f01e8ce`；官方[接入示例](https://chemprop.readthedocs.io/en/latest/chemeleon_foundation_finetuning.html)指定同一文件。是基础 message-passing 权重，不使用下游 bioactivity fine-tuned 模型。
- [公开训练结构列表](https://zenodo.org/records/15733575)：`cleaned_pubchem_1MM.smiles`，43,931,153 bytes，MD5 `1228db3fab5f6dc93ae652eaffa889c1`。无需为结构重叠核查下载另一个约 3 GB descriptors 文件。
- 作者报告 encoder hidden 2048、depth 6，约 8.7M 参数；它比当前 300-dimensional scratch baseline 大。即使更好，也不能把所有差异直接归因于预训练；后续若研究预训练本身的贡献，需要匹配模型大小的对照。
- 已完成：下载哈希确认、基础权重检查、944,296 行预训练结构的标准化身份核查和小批量 embedding 实测。确认母体身份匹配 train 3,260/31,058、validation 299/3,821、test 281/3,944；不能解析或母体含糊的记录另存，不能算作确认未暴露。详见 `chemeleon-audit-full-v1.json`。
- 冻结 encoder pilot 处理 320 个分子约 4.17 秒、峰值 880 MiB；最大分子 batch 检查和推理前后权重不变检查通过。全量 embedding 约 6.29 分钟，三个固定 RF 约 1.68 分钟，最高单进程峰值约 1.0 GiB。
- 首轮 validation：AP 0.3654/0.3775/0.3717，均值 0.3715，Top-50 命中 24/22/22；近邻 AP 0.3210、Top-50 22。三次训练共用一个固定 split，不能把 seed 波动当作跨数据集置信区间。完整比较见 `pretrained_comparison.json`。
- 结构暴露不等于活性标签泄漏；仅凭“未用抗菌标签”也不能宣称“所有测试结构从未出现”。不修改 final test 的成员或读取其标签做选择。后续对所有模型一致报告 validation 的确认身份暴露/未确认匹配子组，作为描述性检查，不据此反复调整模型。

## 保留候选：NaFM

- 科学相关性较强：专门面向天然产物的结构预训练，适合后续 NP/放线菌应用问题；不能假设它一定更擅长当前 Wong 全库任务。见[论文](https://doi.org/10.1038/s42256-026-01226-8)和[作者仓库](https://github.com/TomAIDD/NaFM-Official)。
- [基础 checkpoint](https://zenodo.org/records/15385335)：`NaFM.ckpt`，318,967,349 bytes，MD5 `f2c0c52b816fbc3655c5edba953f0d9a`；[数据存档](https://doi.org/10.6084/m9.figshare.28980254.v1)由作者 README 链接。
- 官方安装示例使用 Python 3.9、Torch/PyG 和 CUDA。Mac CPU 是否可用还需实际适配，不能由示例中的 CUDA 推断它必然不能在本机运行。暂未安装该环境或下载 checkpoint。
- 作者 README 说明 `test.py` 是演示模板，不能直接当作论文评估管线。后续需要明确 checkpoint 是预训练权重还是下游训练权重、输入特征是否完整兼容，并审核结构/标签暴露。

## 本次来源快照

已保存两仓库 README 与 commit 信息、三个 Zenodo API metadata 文件到 `data/raw/pretrained_review/`：NaFM commit `23aa34a56e9addc3059e817819c2600436862a45`；CheMeleon commit `b5e5bd88a7070c3f278e7e65e85b792617bf125c`。目前只准备一个首轮预训练比较模型；NaFM 的专门实验、few-label adaptation 和 ActFound 留待相应问题明确后进行。
