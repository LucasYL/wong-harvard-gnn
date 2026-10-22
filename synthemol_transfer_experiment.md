# SyntheMol独立assay预训练 → Wong微调

日期：2026-09-10。用户已授权启动训练并规划后续放线菌评价。

## 固定问题

更接近Wong菌株/浓度的新SyntheMol assay，能否通过编码器预训练改善Wong候选排序？只比较本次固定方案，不把结果解释为所有多assay方法的结论，也不声称单独分离了assay选择与训练目标/优化步数的因果效果。

## 数据与设置

- 辅助任务：新源train7897、validation585；沿用该来源作者二分类标签，不使用Wong0.2阈值重新标注。预留external1107不参与训练、早停或此次计分。
- 源模型：D-MPNN hidden300、depth3、mean pooling，分类头300→300→1；不加描述符。batch32，Adam固定学习率1e-4，无加权BCEWithLogits，最多20epochs，至少5epochs，源validation AP连续5轮无改善则早停，保留最佳AP checkpoint。
- 仅迁移编码器至原Wong+200描述符模型。分类头同seed重新初始化，重新创建优化器，编码器与分类头均微调。Wong train31058、validation3821、原学习率日程/最多20epochs/AP早停均保持一致。
- seeds固定17/29/43，依次执行source→target，共6阶段。三个target等权ensemble，不挑seed或成员。对照为原匹配三seed描述符模型及近邻。AP主指标，梯形PR-AUC和Top20/50/100为补充。
- 本次不训练CO-ADD数据，不重新评价Wong test，不读取external标签进行模型选择。

## 资源、检查与监视

source pilot用1024/128个train/validation样本、两轮训练，并检查最大32分子批次；最大分子454个重原子。约3.00秒，峰值698.94 MiB。随后source→target端到端检查通过，target两轮最大进程峰值约687.72 MiB。试跑分数只检查流程，不当作模型性能。

正式三seed队列暂预算60–90分钟，CPU两线程、每训练进程3GiB预算。各阶段串行释放内存；已有磁盘约35GiB可用。当前无需云服务。保持插电开盖；caffeinate防止空闲睡眠，不保证合盖继续计算。

每阶段保存best/last checkpoint、逐epoch记录和validation预测。源任务每100批、目标每200批及末批输出进度，显示真实batch loss；不是validation指标。状态JSON按阶段更新。监视命令：

```bash
python3 scripts/watch_transfer.py --experiment synthemol
```

Ctrl+C只停止监视，不停止训练。监视器每30秒显示队列状态，自动跟随全部seeds与阶段。队列失败会记录错误并停止，没有自动覆盖或完整自动恢复。

入口：`scripts/run_synthemol_transfer.py`；配置：`configs/synthemol_transfer_v1.json`；冻结身份：`configs/synthemol_transfer_v1_frozen.json`。冻结前发现并修正新旧数据manifest字段命名差异，保留未使用的preflight快照；正式pilot及运行只使用修正后的冻结版本。

状态：`outputs/synthemol-transfer-v1/status.json`；汇总：`reports/synthemol_transfer_comparison_v1.json`；日志：`logs/synthemol-transfer-v1-source*.log`、`logs/synthemol-transfer-v1-target*.log`。

## 训练完成之后

先核验三个成员的预测、checkpoint、配置与数据哈希，再计算ensemble并与固定对照比较。无论本次是否改善，都不自动增加seeds或无限调参。完成本轮后，优先冻结外部评价方案并推进来源明确的放线菌数据；具体阶段门槛见 `next_stage_plan_zh.md`。

## 正式启动记录

2026-09-10 18:58:38 UTC（Toronto14:58）队列启动，PID99210；caffeinate辅助进程99211，空闲睡眠保护已实查。开始时0/6阶段完成。source17前20轮约每轮9–10秒，峰值约548MiB；这是中途快照，不是完整队列最终资源。实时状态以队列JSON为准。

## 完成与评价（2026-09-10）

六阶段全部完成，队列54.21分钟，单训练进程最高1245.06MiB，无需云计算。Wong validation三成员AP0.4241、Top20/50/100为13/24/31；原描述符三成员AP0.4125、13/22/33。AP及Top-50略有改善，Top-100下降，不能选择性只报有利指标。

随后按预先指定四组对照完成1107个预留结构的一次评价：新迁移AP0.3690、Top-50为28；原描述符0.3420/25、CO-ADD迁移0.3549/27、近邻0.2618/22。新迁移AP差相对原模型的95%配对bootstrap区间[-0.0004,+0.0561]。详见external_evaluation_zh.md。未重新计分旧Wong test。此后以领域适用性数据工作为主，暂不扩大常规训练。
