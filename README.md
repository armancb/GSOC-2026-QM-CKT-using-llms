# GSoC 2026 — ML4SCI Evaluation Tasks

**Candidate:** Md Arman Abdullah
**Program:** MSc. Physics, BITS Pilani K.K. Birla Goa Campus

---

## Overview

This repository contains my evaluation tasks for **Google Summer of Code (GSoC) 2026** under the [**ML4SCI**](https://ml4sci.org/) (Machine Learning for Science) organization. The notebook [`GSOC_EVAL_TASKS.ipynb`](./GSOC_EVAL_TASKS.ipynb) demonstrates a **closed-loop agentic pipeline** for quantum-classical machine learning, built using [**OrchestrAI**](https://pypi.org/project/orchestral-ai/) with **Claude (claude-sonnet-4-0)** as the LLM backend.

The core idea is to register domain-specific tools (quantum computing utilities, ML training loops) with an AI agent and let it autonomously reason about, invoke, and iterate on those tools — forming a closed feedback loop between the agent and the scientific computation.

---

## Tech Stack

| Component | Details |
|-----------|---------|
| **Agent Framework** | [OrchestrAI](https://pypi.org/project/orchestral-ai/) (`orchestral-ai`) |
| **LLM Backend** | Anthropic Claude (`claude-sonnet-4-0`) |
| **ML Framework** | PyTorch + TorchVision |
| **Data** | MNIST handwritten digits dataset |
| **Visualization** | Matplotlib, Pandas |
| **Environment** | Python 3.13, Jupyter Notebook |

---

## Tasks

### Task 1 — OrchestrAI Setup & Hilbert Space Calculator Tool

**Goal:** Set up the OrchestrAI agent framework and demonstrate reliable, repeatable tool calling.

A custom `hilbert_space_dimension` tool is registered using OrchestrAI's `@define_tool()` decorator. The tool computes the dimension of the Hilbert space for a given number of qubits (dimension = 2ⁿ). The agent is instructed to **always** delegate calculations to this tool rather than computing answers itself.

**Key Results:**
- The agent correctly invoked the tool across **all 5 test queries** (1, 4, 8, 10, and 16 qubits).
- Responses were accurate (e.g., 16 qubits → 65,536 dimensions) and accompanied by physically meaningful explanations.
- Demonstrates that OrchestrAI's decorator automatically generates a JSON tool schema from Python type hints and docstrings.

---

### Task 2 — MNIST Classifier as an Agent-Callable Training Tool

**Goal:** Wrap a PyTorch neural network training pipeline as an agent-invocable tool, enabling the AI to control hyperparameters and receive performance feedback.

A simple 2-layer MLP (`784 → 128 → 10`) is trained on a 5,000-sample subset of MNIST using Adam optimizer and CrossEntropyLoss. The `train_mnist_classifier` tool exposes two parameters — `epochs` and `learning_rate` — and returns `final_loss`, `final_accuracy`, and `epochs_run`.

**Key Results (3 epochs, lr=0.001):**

| Epoch | Training Loss |
|-------|---------------|
| 1     | 0.6563        |
| 2     | 0.2918        |
| 3     | 0.2153        |

- **Final Test Accuracy:** 89.9%
- Loss decreased monotonically across epochs, confirming stable learning.

---

### Task 3 — Autonomous Hyperparameter Optimization via Agent

**Goal:** Let the agent **autonomously** search for the best learning rate by iteratively training the model, analyzing results, and deciding the next hyperparameter to try.

A dedicated `train_mnist_classifier_hpo` tool is used alongside an HPO agent. The agent was instructed to run **6 iterations** with different learning rates, starting from `lr=0.01`, and to reason about each result before selecting the next value.

**HPO Results:**

| Iteration | Learning Rate | Test Accuracy | Final Loss |
|-----------|---------------|---------------|------------|
| 1         | 0.01          | 89.6%         | 0.2175     |
| 2         | 0.001         | 90.1%         | 0.2262     |
| 3         | 0.1           | 45.8%         | 1.4556     |
| 4         | 0.005         | 89.8%         | 0.1251     |
| 5         | 0.05          | 77.5%         | 0.5461     |
| 6         | **0.003**     | **92.1%**     | **0.1597** |

**Best Learning Rate:** `0.003` — achieving **92.1% accuracy**.

**Agent's Reasoning:**
- Very high learning rates (0.1, 0.05) caused instability and poor performance.
- Among smaller learning rates, `0.003` struck the optimal balance between convergence speed and stability.
- It was large enough to make meaningful progress in only 3 epochs, while small enough to avoid overshooting.

The results are also visualized via dual plots (Accuracy per Iteration & Loss per Iteration), saved as `hpo_results.png`.

---

## How to Run

1. **Clone the repository:**
   ```bash
   git clone https://github.com/armancb/GSOC-2026-QM-CKT-using-llms 
   cd GsoC
   ```

2. **Set up a Python environment** :
   ```bash
   conda create -n gsoc2026 python=3.13 -y
   conda activate gsoc2026
   ```

3. **Install dependencies:**
   ```bash
   pip install orchestral-ai python-dotenv torch torchvision pandas matplotlib
   ```

4. **Set your Anthropic API key** in a `.env` file:
   ```
   ANTHROPIC_API_KEY=your_key_here
   ```

5. **Run the notebook:**
   ```bash
   jupyter notebook GSOC_EVAL_TASKS.ipynb
   ```


