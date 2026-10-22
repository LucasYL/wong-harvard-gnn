# V0 最终测试方案

日期：2026-09-09。此方案在首次产生 final-test 性能前确定；机器可执行的模型、输入、代码和环境清单将冻结于 `configs/final_evaluation_v1.json`。

## 要回答的问题

在固定 Wong RN4220 assay、固定训练集和 scaffold-held-out test 上，首轮各模型的抗菌 hit 排序能力如何？这是同一 assay 的回顾性评价，不能直接代表放线菌来源天然产物或未来实验的命中率。

## 固定比较

- 共十份预测：一个相似度加权 5-NN、三个 Morgan-RF、三个 scratch Chemprop、三个 CheMeleon 冻结 encoder + RF；三个训练种子均为 17/29/43。
- 使用已经完成的 train-only 模型。Chemprop 使用此前按 validation AP 选定的 epoch 11/10/13 checkpoint。近邻只查询 train 标签，不加入 validation 标签。所有方法保持原表示、标准化和参数。
- 根据完整 validation 比较，预先选定 CheMeleon + RF 为当前候选方案、近邻为重要参考，同时完整公开其余方法。汇总所有种子，不按 test 选择最好的 seed，不临时加入 ensemble、校准、阈值搜索或重训练。

## 指标与解释

- 主指标：sklearn 非插值 average precision；实用指标：Top-20/50/100 hits、precision 与相对随机抽样的 enrichment，其中 Top-50 为主要筛选预算。
- 分数同分按 compound key 排序；同时保留边界同分组的期望 precision/enrichment。ROC-AUC 与 Brier 作为补充，不把未校准分数当作真实概率。
- 三个种子的均值和样本标准差只表示本 split 的训练波动，不作置信区间或统计显著性声明。validation 与 test 的阳性率不同，AP 的跨集合变化不能单独判断为性能退化。
- 所有方法使用相同描述性子组：到 train 最近邻 Tanimoto <0.3、0.3–0.5、≥0.5；确认预训练母体身份暴露 / 未确认完整身份匹配。子组无两个类别时 AP 记空，阳性数少时保留限制，不据此改变 test 或追加模型选择。
- 已知预训练结构暴露并非活性标签暴露；“未确认匹配”不保证完全没有接触相关结构。

## 执行与核查

1. 新的纯推理路径先在全 validation 上复算，须逐分子复现十份已保存的预测及 AP/top-K；检查通过后冻结方案。
2. 逐文件固定数据、模型、配置、关键代码和软件版本 SHA-256 / 版本号，执行前后校验。旧开发配置的 `final_test_evaluation: false` 保持不变，最终评价使用单独入口。
3. 使用 query 结构生成全部分数并保存带 ID 的矩阵与哈希，然后加载 query 活性标签计算指标。CPU 两线程、进程内存预算 3 GiB，预计数分钟，无需云服务。
4. 统一输出 `outputs/final-test-v1/`；已有目录时拒绝再次运行，不覆盖预测。若工程错误导致未完成，保留全部故障记录；不能将已查看的 test 结果重新称为未知。
5. 独立从保存的 CSV 重算指标、检查身份/结构/标签/来源行，保存所有结果，不因不符合预期而更换模型或切分。

## 后续边界

本次完成后，此 test 成为 V0 的历史对照。额外 assays 的 V1 开发仍用开发数据做选择，先固定 assay 身份、菌株/剂量/时间/endpoint 与分组规则；如果开发受到本次结果启发，不能再把原 test 的 V1 分数包装成完全独立验证。后续重要结论需要新的锁定来源/化学域测试或前瞻性实验。
