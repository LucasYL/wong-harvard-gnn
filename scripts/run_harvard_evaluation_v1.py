"""Detached job: evaluate, independently verify, and write the Chinese result record."""
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/harvard-rn4220-evaluation-v1'


def state(value):
    tmp = OUT/'status.json.tmp'
    tmp.write_text(json.dumps({**value, 'updated_at_utc': datetime.now(timezone.utc).isoformat(), 'pid': os.getpid()}, indent=2)+'\n')
    os.replace(tmp, OUT/'status.json')


def main():
    os.chdir(ROOT); OUT.mkdir(exist_ok=True)
    env = dict(os.environ, PYTHONPATH=str(ROOT/'src')+os.pathsep+str(ROOT/'scripts'))
    for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
        env[name] = '2'
    try:
        if not (OUT/'results.json').exists():
            subprocess.run([sys.executable, '-u', 'scripts/evaluate_harvard_rn4220_v1.py', 'run'], env=env, check=True)
        state({'state': 'running', 'stage': 'independent_verification', 'completed': 225829, 'total': 225829})
        subprocess.run([sys.executable, '-u', 'scripts/verify_harvard_evaluation_v1.py'], env=env, check=True)
        result = json.loads((OUT/'results.json').read_text())
        names = {'nearest_neighbour': '近邻基线', 'scratch_descriptors_3': '原始描述符模型（三成员）',
                 'coadd_transfer_3': 'CO-ADD迁移（三成员）', 'synthemol_transfer_3': 'SyntheMol迁移（三成员）'}
        lines = ['# Harvard 外部测试结果\n', '评价完成，独立核查通过。此次没有重新训练模型。\n',
                 '225,829 个分子中，作者标为活性的有 601 个。它们的身份、连接关系及核心骨架均未出现在项目已使用的八个数据分区中。\n',
                 '## 如何看结果\n', 'AP 看整个排序；前50个中的活性数，表示如果按该模型顺序选50个，这个数据表里有多少个被实验标为活性。同分时另给出随机打破同分的期望，避免把ID顺序当模型能力。\n']
        for name, g in result['groups'].items():
            top = g['top_k']['50']
            lines.append(f"- {names[name]}：AP **{g['average_precision']:.4f}**；前50个有 **{top['hits']}** 个活性；考虑边界同分的期望为 **{50*top['tie_averaged_precision']:.2f}** 个。")
        lines += ['\n## 改善有多确定\n']
        for name, d in result['paired_ap_difference_vs_scratch'].items():
            lo,hi=d['percentile_95_interval']
            lines.append(f"- {names[name]}相对原始描述符模型的AP差：{d['observed']:+.4f}，按核心骨架成组重采样的95%区间 [{lo:+.4f}, {hi:+.4f}]。")
        lines += ['\n区间只反映固定模型在这批分子上的样本变化，不包含重新训练的变化。不要只凭某个最高值认定某种方法全面胜出。\n',
                  '## 结论范围\n', '这是另一种实验条件下的广泛化合物排序测试。原液浓度仍未知；不是MIC预测、放线菌专用验证，也没有新实验或新抗生素发现。不同数据集活性比例不同，不能直接拿这里的AP与Wong的AP比较高低。\n',
                  '本集合现在是已经看过结果的历史评价。下一步先解释差异，不能根据这些成绩改模型后继续把它称作从未见过的测试集。\n']
        (ROOT/'harvard_external_results_zh.md').write_text('\n'.join(lines)+'\n')
        marker = '### Harvard RN4220 外部评价完成（自动记录）'
        process = ROOT/'process.md'
        if marker not in process.read_text():
            with process.open('a') as f:
                f.write('\n\n'+marker+'\n\n- 全部225829分子的四组模型排序、骨架分组bootstrap及独立核查完成。结果见harvard_external_results_zh.md。没有重新训练或选择新阈值。本集合自此作为历史评价保存。\n')
        state({'state': 'complete', 'stage': 'verified_and_report_saved', 'completed': 225829, 'total': 225829,
               'report': 'harvard_external_results_zh.md'})
        print('Completed and independently verified. Mandarin report: harvard_external_results_zh.md', flush=True)
    except Exception as exc:
        previous = json.loads((OUT/'status.json').read_text()) if (OUT/'status.json').exists() else {}
        state({**previous, 'state': 'failed', 'error': str(exc)})
        raise


if __name__ == '__main__':
    main()
