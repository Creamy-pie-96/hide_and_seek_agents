# RL Snake Project Plan (Advanced Architecture)

## 1. Objective
Develop a Reinforcement Learning agent capable of playing the Snake game, optimizing for maximum length and survival time. The agent will use a Convolutional Neural Network (CNN) to process spatial board data and a Deep Q-Network (DQN) for decision making.

## 2. Environment Setup
- **Simulation Engine**:`Pygame` for the game loop.
- **Grid**: A fixed-size grid (e.g., 20x20).
- **Observation Space (Spatial Grid)**:
    Instead of a flat vector, the agent receives a 3-channel tensor of shape `(20, 20, 3)`:
    - **Channel 1**: Binary mask of the snake's head (1 at head, 0 elsewhere).
    - **Channel 2**: Binary mask of the snake's body (1 at body, 0 elsewhere).
    - **Channel 3**: Binary mask of the food location (1 at food, 0 elsewhere).
    This allows the CNN to learn spatial patterns, such as "pockets" or "traps."

## 3. Action Space
Relative movement to prevent illegal 180-degree turns:
- `0`: Continue Straight
- `1`: Turn Right
- `2`: Turn Left

## 4. Reward Function
| Event | Reward | Reasoning |
| :--- | :--- | :--- |
| **Eating Food** | `+10.0` | Primary goal; strong positive reinforcement. |
| **Collision (Wall/Self)** | `-10.0` | Terminal state; strong negative reinforcement. |
| **Moving Towards Food** | `+0.1` | Reward shaping to guide the agent in early training. |
| **Moving Away from Food** | `-0.1` | Discourage wandering. |
| **Step Penalty** | `-0.01` | Encourages finding the shortest path to food. |
| **Stalling/Looping** | `-0.5` | Penalty for visiting the same coordinates repeatedly without eating. |

## 5. RL Architecture (CNN-DQN)
- **Network**: Convolutional Neural Network (CNN).
    - **Input**: `(20, 20, 3)` tensor.
    - **Conv Layer 1**: 32 filters (3x3), ReLU activation, stride 1.
    - **Conv Layer 2**: 64 filters (3x3), ReLU activation, stride 1.
    - **Flatten Layer**: Converts 2D feature maps to a 1D vector.
    - **Dense Layer 1**: 128 neurons, ReLU activation.
    - **Output Layer**: 3 neurons (Q-values for Straight, Right, Left).
- **Hyperparameters**:
    - **Learning Rate**: $1 \\times 10^{-4}$
    - **Discount Factor ($\\gamma$)**: 0.99
    - **Epsilon ($\\epsilon$)**: Start at 1.0, decay to 0.01.
    - **Experience Replay**: Buffer size of 100,000 transitions.
    - **Batch Size**: 64.

## 6. Implementation Phases
### Phase 1: Environment
- Implement Snake logic and the 3-channel grid observation wrapper.
### Phase 2: Agent Development
- Implement the CNN architecture and the Replay Buffer.
- Implement $\\epsilon$-greedy action selection.
### Phase 3: Training
- Train in "headless" mode for speed.
- Log average score and episode length.
### Phase 4: Visualization
- Enable simulation render to observe spatial reasoning (e.g., avoiding self-trapping).

**Summary: By switching to a CNN with a grid-based observation space, the agent gains spatial awareness, allowing it to recognize complex board configurations that a simple MLP would miss.**