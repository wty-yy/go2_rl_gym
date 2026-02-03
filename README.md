<div align="center">
	<h1 align="center">Go2 RL GYM</h1>
	<p align="center">
		<span>🌎 English</span> | <a href="README_zh.md">🇨🇳 中文</a>
	</p>
</div>

<p align="center">
	<strong>This repository builds on <a href="https://github.com/unitreerobotics/unitree_rl_gym">unitree_rl_gym</a> to train the Unitree Go2 quadruped with reinforcement learning.</strong>
</p>

<div align="center">

| <div align="center"> Isaac Gym </div> | <div align="center"> Mujoco </div> |  <div align="center"> Physical </div> |
|--- | --- | --- |
| ![isaacgym eval](https://raw.githubusercontent.com/robogauge/picture-bed/refs/heads/main/go2_rl_gym/isaacgym_eval.gif)  | ![mujoco eval](https://raw.githubusercontent.com/robogauge/picture-bed/refs/heads/main/go2_rl_gym/mujoco_eval.gif) | ![real eval](https://raw.githubusercontent.com/robogauge/picture-bed/refs/heads/main/go2_rl_gym/real_eval.gif) |

</div>

## 📦 Installation

Follow the step-by-step setup guide in [setup.md](doc/setup_en.md).

## 🛠️ Usage Guide

### 1. Train

Run the following command to launch training:

```bash
python legged_gym/scripts/train.py --task=xxx
```

#### ⚙️  Arguments
- `--task`: Required. Options include `go2`, `go2_cts`, `go2_moe_cts`, `go2_moe_ng_cts`, `go2_mcp_cts`, `go2_ac_moe_cts`, `go2_dual_moe_cts`; `go2_moe_cts` is the paper's final version.
- `--headless`: Render viewer by default; set to `true` to disable rendering for higher throughput.
- `--resume`: Resume training from a chosen checkpoint in the logs.
- `--experiment_name`: Experiment folder to save/load from.
- `--run_name`: Run subfolder name to save/load from.
- `--load_run`: Name of the run to load (defaults to the most recent run).
- `--checkpoint`: Checkpoint index to load (defaults to the latest file).
- `--num_envs`: Number of parallel simulated environments.
- `--seed`: Random seed.
- `--max_iterations`: Maximum training iterations.
- `--sim_device`: Physics simulation device. Use `--sim_device=cpu` to force CPU.
- `--rl_device`: RL computation device. Use `--rl_device=cpu` to force CPU.
- `--robogauge`: Enable RoboGauge evaluation tool; disabled by default. Evaluation results are saved as `results_{it}.yaml` in `logs/{exp_name}/{date}/robogauge_results` and logged to TensorBoard.
- `--robogauge_port`: RoboGauge server port; default is 9973.

> RoboGauge evaluation requires a separate server to be started. Refer to the [RoboGauge documentation](https://github.com/wty-yy/RoboGauge).

**Default checkpoint path**: `logs/<experiment_name>/<date_time>_<run_name>/model_<iteration>.pt`

---

### 2. Play

Visualize policies inside Gym with:

```bash
python legged_gym/scripts/play.py --task=xxx
```

**Notes**

- Play launches on randomized terrain with difficulty between 7 and 9.
- It automatically loads the latest checkpoint inside the experiment folder.
- Override via `experiment_name` and `checkpoint`, for example:
	```bash
	python legged_gym/scripts/play.py --task=go2_cts --num_envs 100 --experiment_name go2_cts_hard_terrain --checkpoint 100000
	```

#### 💾 Policy Export

Play exports the Actor network to `logs/{experiment_name}/exported/policies`:
- `policy.pt`: TorchScript model for Sim2Sim.
- `policy.onnx`: ONNX model for Sim2Real.
- `policy.pkl`: Raw weights.
  
#### Demonstration

![isaacgym play](https://raw.githubusercontent.com/robogauge/picture-bed/refs/heads/main/go2_rl_gym/isaacgym_play.gif)

---

### 3. Sim2Sim (Mujoco)

Run policies in the Mujoco simulator:

```bash
python deploy/deploy_mujoco/deploy_go2.py
```

Connect an Xbox-compatible gamepad to enable teleoperation; otherwise, the agent keeps a default forward command.

- **Swap the policy**: The default checkpoint is `deploy/pre_train/go2/go2_cts_150k.pt`. Replace `policy_path` in the YAML config with your own `logs/{experiment_name}/exported/policies/policy.pt`.
- **Swap terrains**: Default terrain is `resources/robots/go2/stairs.xml`. Alternatives include `flat.xml`, `race_track.xml`, `cross_stairs.xml`, and `cross_slope.xml`. Generate new terrains with [windigal - mujoco_terrains](https://github.com/windigal/mujoco_terrains)

#### Results

| Flat | Stairs | Race Track |
|--- | --- | --- |
| <img src="https://raw.githubusercontent.com/robogauge/picture-bed/refs/heads/main/go2_rl_gym/mujoco_eval_flat.gif" width="250"/> | <img src="https://raw.githubusercontent.com/robogauge/picture-bed/refs/heads/main/go2_rl_gym/mujoco_eval.gif" width="250"/> | <img src="https://raw.githubusercontent.com/robogauge/picture-bed/refs/heads/main/go2_rl_gym/mujoco_eval_track.gif" width="250"/> |

---

### 4. Sim2Real

#### 4.1 Python Deployment (requires [unitree_sdk2_python](https://github.com/unitreerobotics/unitree_sdk2_python))

```bash
# Onboard Jetson: pick Python by JetPack version
# JetPack 6: Python 3.10
# JetPack 5: Python 3.8
conda create -n deploy python=3.10
conda activate deploy
# Install the matching PyTorch wheel for your Jetson
# https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048
git clone https://github.com/unitreerobotics/unitree_sdk2_python.git
cd unitree_sdk2_python
pip3 install -e .
```

In the Unitree app, open Device → Service, disable `mcf/*`, and enable the `ota_box` service.

Assuming the interface to the low-level controller is `eth0`:

```bash
cd deploy/deploy_real
python deploy_real_go2.py eth0
```

Press `start` to stand and `A` to engage the controller.

#### 4.2 C++ Deployment (requires unitree_cpp_deploy)

Follow the usage described in [unitree_cpp_deploy](https://github.com/wty-yy-mini/unitree_cpp_deploy).

#### Demonstration

| Python Deploy | C++ Deploy |
| --- | --- |
| ![python deploy](https://raw.githubusercontent.com/robogauge/picture-bed/refs/heads/main/deploy/py_deploy_with_commands.gif) | ![cpp deploy](https://raw.githubusercontent.com/robogauge/picture-bed/refs/heads/main/deploy/cpp_deploy_with_commands.gif) |

---

## 🎉  Acknowledgements

This repository would not exist without the following open-source projects:

- [unitree_rl_gym](https://github.com/unitreerobotics/unitree_rl_gym): Unitree's core RL training framework.
- [legged_gym](https://github.com/leggedrobotics/legged_gym): Base locomotion environment.
- [rsl_rl](https://github.com/leggedrobotics/rsl_rl.git): Reinforcement learning algorithms.
- [mujoco](https://github.com/google-deepmind/mujoco.git): High-performance CPU physics simulator.
- [unitree_sdk2_python](https://github.com/unitreerobotics/unitree_sdk2_python.git): Python hardware interface for deployment.
- [unitree_sdk2](https://github.com/unitreerobotics/unitree_sdk2): C++ hardware interface for deployment.

Related publications implemented in this repo:
- [CTS: Concurrent Teacher-Student Reinforcement Learning for Legged Locomotion](https://arxiv.org/pdf/2405.10830)

Contributors:
- [@windigal](https://github.com/windigal): CTS algorithm reproduction, terrain generation, video editing
- [@wertyuilife2](https://github.com/wertyuilife2): CTS algorithm reproduction

---

## 📄  Citation
If you find our work helpful, please cite:
```bibtex
@article{wu2026robogauge,
      title={Toward Reliable Sim-to-Real Predictability for MoE-based Robust Quadrupedal Locomotion}, 
      author={Tianyang Wu and Hanwei Guo and Yuhang Wang and Junshu Yang and Xinyang Sui and Jiayi Xie and Xingyu Chen and Zeyang Liu and Xuguang Lan},
      year={2026},
      journal={arXiv preprint arXiv:2602.00678},
      url={https://arxiv.org/abs/2602.00678}, 
}
```

## 🔖  License

New contributions follow the [MIT License](LICENSE); the original unitree_rl_gym remains under the [BSD 3-Clause License](LICENSE).

See the complete [LICENSE file](LICENSE) for details.

