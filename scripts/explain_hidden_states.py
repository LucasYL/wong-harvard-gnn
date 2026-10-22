"""用 CCO 对照本项目的 x、键 h、原子 h 和分子 h；使用未训练权重。

在项目目录运行：.venv-neural/bin/python scripts/explain_hidden_states.py
"""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from antibacterial.neural_descriptors import model
from antibacterial.descriptors import vector, rdNormalizedDescriptors
from chemprop import data, featurizers
import torch


def show(name, value):
    """只显示形状和第一行的前 6 个数字，方便阅读。"""
    numbers = value[0, :6].detach().tolist()
    print(f"{name}: shape={list(value.shape)}, 第一行前6项={[round(x, 4) for x in numbers]}")


def main():
    torch.set_num_threads(2)
    torch.manual_seed(17)
    config = json.loads((ROOT / "configs/chemprop_descriptors_v1.json").read_text())
    network = model(config)
    network.eval()
    mp = network.message_passing

    # 1. SMILES → RDKit 分子 → 固定的原子/键特征及连接索引。
    # CCO 的三个节点为 C0—C1—O2；没有提供或假设乙醇的活性标签。
    point = data.MoleculeDatapoint.from_smi("CCO")
    graph = featurizers.SimpleMoleculeMolGraphFeaturizer()(point.mol)
    bmg = data.BatchMolGraph([graph])
    x_atoms = bmg.V                     # [3,72]：原子输入特征
    e_bonds = bmg.E                     # [4,14]：有向键输入特征
    source, target = bmg.edge_index     # 四条边：0→1、1→0、1→2、2→1
    print("CCO；未训练权重；仅用于理解计算。")
    print("有向边：", bmg.edge_index.T.tolist())
    show("x_atoms", x_atoms)
    show("e_bonds", e_bonds)

    with torch.no_grad():
        # 2. 每条有向键：起点原子的72维 + 键的14维 → 86维。
        edge_inputs = torch.cat([x_atoms[source], e_bonds], dim=1)
        z0_edges = mp.W_i(edge_inputs)  # [4,300]：ReLU 之前的初始线性结果
        h_edges = torch.relu(z0_edges) # [4,300]：初始键 h
        show("edge_inputs", edge_inputs)
        show("h_edges 初始化", h_edges)

        # 3. 与本项目 depth=3 一致：初始化后再更新两轮。
        # message() 收集进入起点的消息，并排除目标发来的反向消息。
        for step in range(1, mp.depth):
            messages = mp.message(h_edges, bmg)
            h_edges = torch.relu(z0_edges + mp.W_h(messages))
            show(f"h_edges 更新{step}后", h_edges)

        # 4. 汇总进入每个原子的键消息，再结合原子的原始特征。
        incoming = torch.zeros(len(x_atoms), config["hidden_dim"])
        incoming.index_add_(0, target, h_edges)  # [3,300]
        atom_inputs = torch.cat([x_atoms, incoming], dim=1)  # [3,372]
        h_atoms = torch.relu(mp.W_o(atom_inputs))           # [3,300]
        show("incoming", incoming)
        show("atom_inputs", atom_inputs)
        show("h_atoms", h_atoms)
        print("三个原子的 h 的前6项：")
        for index, row in enumerate(h_atoms):
            print(f"  原子{index}: {[round(x, 4) for x in row[:6].tolist()]}")

        # 5. 这里只有一个分子：逐列平均，保持一行。
        h_molecule = h_atoms.mean(dim=0, keepdim=True)       # [1,300]
        show("h_molecule", h_molecule)

        # 6. 拼接与项目相同的200维描述符，然后计算分数。
        descriptor_values, missing = vector(
            rdNormalizedDescriptors.RDKit2DNormalized(), "CCO"
        )
        descriptors = torch.from_numpy(descriptor_values).unsqueeze(0)
        classifier_input = torch.cat([h_molecule, descriptors], dim=1)
        logits = network.predictor.train_step(classifier_input)
        score = logits.sigmoid()
        show("descriptors", descriptors)
        show("classifier_input", classifier_input)
        print(f"logit={logits.item():.6f}, 分数={score.item():.6f}（未训练，无活性解释）")
        print("描述符非有限值填0位置：", missing)

        # 核对分步展开的计算与项目封装的 forward 完全对应。
        torch.testing.assert_close(h_atoms, mp(bmg))
        torch.testing.assert_close(h_molecule, network.agg(h_atoms, bmg.batch))
        torch.testing.assert_close(classifier_input, network.fingerprint(bmg, None, descriptors))
        torch.testing.assert_close(score, network(bmg, None, descriptors))
        print("核对通过：逐步计算与项目封装输出一致。")


if __name__ == "__main__":
    main()
