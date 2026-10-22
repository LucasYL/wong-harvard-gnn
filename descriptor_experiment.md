# 描述符与集成：本机开发实验

更新：2026-09-10。十个预定成员已完成，预测与协议完整性复核通过。

## 完成结果

队列于 Toronto 2026-09-09 17:30 启动、19:47 完成，共 137.18 分钟；最高单训练进程峰值内存 1506.17 MiB。验证集 3821 个分子、60 个 hits，未重新评价 test。

三个匹配 seeds 的 ensemble：无描述符 AP 0.3713、Top-50 为 22；有描述符 AP 0.4125、Top-50 为 22。十成员描述符 ensemble AP 0.4150、梯形 PR-AUC 0.4094，Top-20/50/100 命中 13/20/33；近邻相应为 13/22/30。描述符改善整体排序，扩大 ensemble 的额外收益有限，也未改善 Top-50。模型选择与 checkpoint 选择使用同一 validation，不能视为独立泛化证据。

完整数值见 `reports/descriptor_ensemble_comparison.json`。2026-09-10 重新从保存预测核算指标，并校验配置、源代码、数据、旧基线和 checkpoint 哈希，十个成员全部通过。

## 目的与评价边界

在固定 Wong RN4220 train/validation 上，检验「加入 200 个全局分子描述符」和「平均多个模型预测」是否改善候选排序。此次是开发实验；V0 test 已经查看，不作为新的独立评价，也不自动重新计分。没有新增 assay 或实验标签。

## 固定比较

- 保留现有轻量架构：hidden 300、depth 3、mean pooling、FFN 一个隐层、dropout 0、batch 32、CPU 两线程；最多 20 epochs，validation AP 早停。
- 新增 `descriptastorus.RDKit2DNormalized` 的 200 维描述符，拼接到图表示后进入分类器。使用现有 RDKit 2025.9.6 与 descriptastorus 2.8.0，不额外根据本数据拟合缩放参数。该库归一化来自固定参考分布，不应误解为本项目新拟合的训练集统计量。
- 描述符仅从标准化结构计算；train/validation 按 compound key 与顺序对齐。计算失败即停止；个别非有限值按预定规则填 0，并逐化合物记录列名。不得无记录删掉难处理分子。
- 预定 seeds：17、29、43、59、71、89、101、113、127、139。分别报告单模型，以及前 3 个、全部 10 个成员的等权预测均值；不按结果挑成员或权重。
- 同 seed 的三个旧模型作为无描述符对照，也构成一个三成员 ensemble。三对三用于解释描述符效果；十成员仅作为进一步集成规模实验，不冒充完整的十对十消融实验。
- 保留 Morgan-RF 和近邻基线。主指标 AP；补充明确标注的梯形 PR-AUC、Top-20/50/100。旧 V0 报告不覆盖。
- 不是 Wong hidden 1600 / depth 5 / 调参 / 训练内部随机划分的完整复现，不能把改动后的分数直接与论文 0.364 比较。

## 执行和恢复

先完成结构描述符与小批训练资源测试，再冻结配置、源代码、数据、描述符缓存和旧对照的 SHA-256。十个模型顺序训练，每轮保存最新恢复 checkpoint，并保留 AP 最佳 checkpoint。每个成员完成后校验预测、更新比较文件；全部完成后队列自动标记完成。

运行信息在 `outputs/chemprop-descriptors-v1/status.json`；成员日志在 `logs/chemprop-descriptors-v1-seed*.log`；每轮进展在各成员输出目录的 `epoch_progress.json`。运行状态须同时核对 PID 和日志；文件存在不代表进程仍存活。

在项目目录查看状态：

```sh
PYTHONPATH=src .venv-neural/bin/python -m antibacterial.descriptor_batch status
```

仅在进程已经退出且确认需要恢复时执行：

```sh
PYTHONPATH=src /usr/bin/caffeinate -i .venv-neural/bin/python -m antibacterial.descriptor_batch run --resume
```

恢复会校验已完成成员，未完成成员从保存的最新 epoch checkpoint 继续；不承诺故障恢复后与从未中断的运行逐位一致。没有可恢复 checkpoint 的失败尝试必须保留并另定新 run ID，不能直接覆盖。

## 本机使用

已测旧轻量模型约 13–15 分钟/个，跑满 20 epochs 预留约 18–20 分钟，十个模型约 2.5–3.5 小时；新模型资源试跑后更新预算。十个成员顺序执行，不需要十倍同时训练内存。

保持接通电源和开盖。队列使用进程级 `caffeinate -i` 防止空闲睡眠，结束后自动释放，不修改全局电源设置；显示器可以关闭或锁屏。合盖通常仍会睡眠，计算随之暂停，不能指望该命令绕过合盖睡眠。完整 Wong 大模型以及 ActFound 全量预训练的资源尚未测试。
