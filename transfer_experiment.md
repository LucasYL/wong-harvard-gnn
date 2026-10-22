# CO-ADD 辅助预训练 → Wong 分类：首轮迁移实验

更新：2026-09-10。用户要求继续推进；当前实验在本机执行，不购买云资源。

## 问题与边界

在完全相同的 Wong train/validation 和目标模型设置下，先用额外 assay 学习的图编码器，是否比随机初始化提供更好的候选排序？当前是开发比较，不重新评价已经查看的 V0 test，不证明放线菌或真实实验效果。此首轮检验整个预训练方案的收益，未单独控制额外优化步数等所有因素，也不是 ActFound/meta-learning 复现。

## 辅助数据已经准备

固定 CO-ADD GP_020 / ATCC 43300 MRSA / 32 µg/mL。由 76,887 条源记录保守得到 72,495 个不同母体身份，辅助训练 69,367、辅助验证 3,128。配置先固定再过滤和划分。

按照优先原因互斥统计排除：结构待审 352 行；同 InChIKey 不同母体写法 115 行；其他重复身份 1,041 行；测量摘要 QC 361 行；Wong validation/test 结构或骨架重叠 2,523 行。所有待审记录保留源行指针，未直接平均不同测量或挑最好结果。这个规则会损失部分有效重复测量，是保守第一版选择。

同时检查 InChIKey、parent SMILES、无立体连接结构和 scaffold；包括 ACYCLIC 组。辅助 train/validation 之间以及对 Wong validation/test 的上述重叠均为零。只读 Wong 结构列，不读其 test activity。

辅助 Y 为原始平均 inhibition_percent/100，保留负值和超过100的值，不取对数、不裁剪、不产生作者未提供的 hit 标签。训练范围约−1.99至1.13，因此使用对大残差较稳健的 SmoothL1 损失。原始 assay 的摘要证据不等于重建了 plate-level QC；这项限制保留。数据仍在本地，来源发布/再分发条款尚未完整核实。

## 固定训练方案

- 三个 seeds：17、29、43，不挑 seed 或 ensemble 成员。
- 每个 seed 先做辅助预训练：D-MPNN hidden300、depth3、mean pooling，回归头300→300→1；不加入全局描述符。Adam 学习率固定1e-4，batch32，5 epochs；用辅助 validation SmoothL1 最低的 checkpoint。
- 只把消息传递编码器参数转移到目标模型。目标仍为此前的 D-MPNN +200 descriptors；分类头按同 seed 新初始化，优化器重新初始化，编码器和分类头都在 Wong train 上更新。
- 目标训练的 batch32、学习率日程、最多20 epochs、AP早停等保持原描述符配置一致。原 scratch 三个 seeds 的结果作为对照，并独立从保存预测复算。
- 报告三个单模型、三成员等权 ensemble、scratch描述符三成员和近邻。AP为主，梯形PR-AUC与Top20/50/100补充，不重新打开test计分。

## 资源和执行

资源 pilot：1,024 个辅助训练样本、最大32个分子前后向，以及编码器严格加载检查，约8.45秒，峰值661.19 MiB；最大分子89个重原子。预测精度不是该pilot目的，未保存其模型权重。

初步外推每seed辅助5 epochs约6.7分钟，再加此前目标模型约13–15分钟/seed，总计约60–90分钟，包含启动和较保守余量；这是估计，实际训练速度与早停不同。CPU两线程、每进程3GiB预算，各阶段串行。保持插电和开盖；进程级caffeinate防止空闲睡眠，不能保证合盖后继续。

小规模源任务→目标任务端到端检查已通过，正式队列于 2026-09-10 16:04:03 UTC（Toronto 12:04）启动，初始 PID 88157。目前处于 seed17 辅助预训练，尚无完成的目标模型。首轮实测78.30秒、单训练进程峰值1361.19 MiB；60–90分钟仍为全队列估计。此处是启动时快照，实时状态以以下状态文件为准。

每个阶段分别保存最佳和最新checkpoint、逐化合物validation预测及运行记录；每完成一个目标成员更新比较报告。训练期间不修改冻结代码、配置和输入数据。

冻结配置：`configs/transfer_v1_frozen.json`；入口：`scripts/run_auxiliary_transfer.py`。

队列状态：`outputs/coadd-transfer-v1/status.json`；日志：`logs/coadd-transfer-v1-source*.log` 与 `logs/coadd-transfer-v1-target*.log`；结果：`reports/coadd_transfer_comparison_v1.json`。

统一终端监视：在项目目录运行 `python3 scripts/watch_transfer.py`，自动跟随三个seed的源任务和目标任务日志，显示epoch、指标、耗时、峰值进程内存以及30秒状态提示。首次只展示各现有日志末三行，之后显示新输出；`--once`只看快照，Ctrl+C仅退出监视。该工具只读状态和日志，不更改训练协议。当前训练脚本在epoch结束时写指标，监视工具不会伪造batch进度。

失败会停止并记录错误。本入口不自动覆盖或重开已有队列；须先查看失败阶段和已保存权重，再制定保留原记录的恢复方案。尚未实现完整自动断点恢复，不把 last checkpoint 的存在说成无条件可恢复。

## 完成结果（2026-09-10复核）

正式队列已于UTC17:00:04（Toronto13:00）完成全部六阶段，耗时56.01分钟，最高单训练进程峰值1471.22 MiB。以上运行中描述为历史启动快照。重新执行summarize并核验冻结身份、checkpoint及保存预测成功。

同一Wong validation的3821个分子、60个hits：迁移三成员ensemble AP0.4010、Top-50命中21；原匹配三成员描述符ensemble AP0.4125、Top-50命中22；近邻AP0.3210、Top-50命中22。Top-100迁移34、原描述符33、近邻30。三个seed仅一个的AP相对scratch提高，当前方案没有明确增益，不能因Top-100多一个命中就宣布成功。

这个结果只评价固定的辅助预训练方案；不证明其他assays或meta-learning无效，不能确认下降原因。建议先诊断具体排名变化与结构相似度分层，再筛选条件更匹配的数据；不扩大当前方案训练。仍是validation开发比较，没有新独立或放线菌实验验证。
