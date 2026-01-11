# Maritime MEC via HASAC

**Joint Computation Offloading and Resource Allocation for Uncertain Maritime MEC via Cooperation of AAVs and Vessels**

This repository implements the HASAC (Heterogeneous-Agent Soft Actor-Critic) algorithm for maritime mobile edge computing, based on the HARL framework.

## Overview

This project addresses computation offloading and resource allocation in maritime MEC environments with:
- **MIoT Devices**: Maritime IoT devices generating computation tasks
- **AAVs**: Autonomous Aerial Vehicles as edge servers with limited computing power
- **Vessels**: Ships with powerful computing capabilities

The system uses heterogeneous-agent reinforcement learning where AAVs and Vessels are decision-making agents.

## Key Features

- **HASAC Algorithm**: Off-policy heterogeneous-agent SAC for multi-agent coordination
- **Lyapunov Optimization**: Queue management with stability guarantees
- **Realistic Channel Model**: Path loss and transmission rate based on maritime communication standards
- **Drift-Plus-Penalty Reward**: Balances task delay minimization and queue stability

## Installation

```bash
# Create conda environment
conda create -n maritime-mec python=3.8
conda activate maritime-mec

# Install PyTorch (CUDA 11.0+)
pip install torch>=1.9.0

# Install dependencies
pip install -e .
```

## Quick Start

```bash
# Train HASAC on Maritime MEC
python examples/train.py --algo hasac --env maritime_mec --exp_name maritime_exp

# Or use the provided script
./run_tvt_experiment.sh
```

## Citation

If you find this code useful for your research, please consider citing our papers:

```bibtex
@ARTICLE{11045994,
  author={You, Jiahao and Jia, Ziye and Dong, Chao and Wu, Qihui and Han, Zhu},
  journal={IEEE Transactions on Vehicular Technology}, 
  title={Joint Computation Offloading and Resource Allocation for Uncertain Maritime MEC via Cooperation of AAVs and Vessels}, 
  year={2025},
  volume={74},
  number={11},
  pages={18081-18095},
  doi={10.1109/TVT.2025.3581970}
}

@ARTICLE{11284890,
  author={You, Jiahao and Jia, Ziye and Cui, Can and Dong, Chao and Wu, Qihui and Han, Zhu},
  journal={IEEE Transactions on Cognitive Communications and Networking}, 
  title={Hierarchical Task Offloading and Trajectory Optimization in Low-Altitude Intelligent Networks Via Auction and Diffusion-based MARL}, 
  year={2025},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TCCN.2025.3641588}
}
```

## Acknowledgments

Based on the [HARL](https://github.com/PKU-MARL/HARL) framework.

## License

MIT License
