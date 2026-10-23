# 共同开发

本仓库从已完成的 Wong→Harvard 实验开始。原来的评价已被查看；后续模型改进需要明确哪些数据用于开发、哪些用于独立结论。

## 初次加入

1. 接受 GitHub 协作邀请，克隆私有仓库。
2. 按 README 安装基础环境，下载 `v1.0.0-harvard` 数据包。
3. 校验 SHA-256，并运行已保存 Harvard 结果的两个核验入口。
4. 需要操作 GNN 时，再装 neural 环境并运行小样本 validation 复现。
5. 在 issue 中记录操作系统、Python、依赖、命令和结果；不要上传 token 或环境凭据。


不要在同一个 PR 里同时改变标签、split、架构、指标与对照定义。如果需要多个变化，把各自的解释和匹配对照写清楚。

## 实验记录

每次新实验创建唯一 run ID，保存：

- 研究问题、预定比较、主指标与 Top-K。
- 数据来源、assay、身份标准化、排除规则、划分和输入哈希。
- Git commit、配置、依赖版本、seed、资源预算。
- checkpoint 选择规则、运行状态、完整预测、指标及校验回执。
- 失败/中断原因、未完成项、置信区间的范围和结论限制。

运行状态用 proposed / running / completed / verified 等明确措辞。保存了预测不等于验证了科学结论。

## 保留历史

`configs/*frozen*.json`、原始 `src/antibacterial/*.py` 及历史报告被哈希关联。要修改时创建新版本；原冻结数据包和 release 不替换。Harvard 与 V0 test 已是历史 benchmark，不能用它们反复调参后继续称独立测试。

`data/`、`outputs/` 和下载包都被 Git 忽略。新成果要先生成来源/哈希清单，再使用单独的版本化 release 或约定存储。上游许可未确认的材料不要改为公开分发。

## 指标和 assay 约定

- 主指标使用非插值 AP；梯形 PR-AUC 另外命名。
- Top-K 固定并保留同分处理；不能事后挑最好 seed 或 K。
- compound + assay context → measurement；缺测不等于 inactive。
- 划分、标签、源暴露与是否已查看结果必须明确。
- 新长任务先做资源 pilot，明确成本和监控入口，再启动全量运行。

## CI 模板

`docs/ci/tests.yml` 保存基础测试工作流。此次 GitHub OAuth 令牌缺少 `workflow` scope，GitHub 拒绝直接上传活动工作流；因此模板尚未启用。维护者可使用具备该权限的凭据将其复制到 `.github/workflows/tests.yml`。当前交接验证来自本地基础和 neural 环境，不能称为 GitHub CI 已通过。
