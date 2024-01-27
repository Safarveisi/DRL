# Documentation
You can find information about algorithms and technical considerations in `doc.pdf`.

# Reinforcement learning algorithms
## Off-policy algorithms 

| Name | Jupyter Notebook | Link to paper |
| ---- | ---------------- | ----------------- |
| Double Deep Q Network (DDQN) | `ddqn.ipynb` | [Link](https://arxiv.org/abs/1509.06461) |
| Prioritized-Experience-Replay DDQN (PER-DDQN) | `pddqn.ipynb` | [Link](https://arxiv.org/abs/1511.05952) |
| Dual DDQN (D-DDQN) | `dddqn.ipynb` | [Link](https://ieeexplore.ieee.org/abstract/document/8483478) |

## On-policy algorithms
| Name | Jupyter Notebook | Link to paper |
| ---- | ---------------- | ------------- |
| Proximal Policy Optimization (PPO) | `ppo.ipynb` | [Link](https://arxiv.org/abs/1707.06347) |

# Usage
Create a python virtual environment
```bash
python -m venv rl
source rl/bin/activate
pip install -r requirements.txt
```
Now you can set `rl` as your jupyter notebook kernel. **All jupyter notebooks are self-contained**. 

# Reference
* [Git Repository 1](https://github.com/p-christ/Deep-Reinforcement-Learning-Algorithms-with-PyTorch/tree/master)
* [Git repository 2](https://github.com/ericyangyu/PPO-for-Beginners)