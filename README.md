# UCSD ML Research Prep: From Foundations to Embodied AI
**Target:** Hao-AI Lab (Prof. Hao Su) | **Focus:** 3D Perception & Robotics

This repository documents a high-intensity 5-week sprint to master the mathematical and engineering foundations required for research in Embodied AI.

---

## 📈 High-Intensity Roadmap

### Week 1: The Engine (Optimization & Vectorization)
*Mastering the calculus of learning and hardware-level performance.*
- [x] **Day 1:** Environment setup, GitHub Desktop integration, and Hardware Benchmarking (MPS/CUDA).
- [ ] **Day 2:** **Manual Backprop:** Implement Linear Regression using raw tensors and partial derivatives (No `.backward()`).
- [ ] **Day 3:** **The Optimizer:** Build custom SGD and Adam modules; explore the Loss Landscape.
- [ ] **Day 4:** **Vectorization Challenge:** Eliminate `for` loops in math operations using Broadcasting.
- [ ] **Day 5:** **Softmax & Stability:** Build a multi-class classifier with numerical stability (Log-Sum-Exp).
- [ ] **Weekend:** Refactor foundations into object-oriented `nn.Module` architecture.

### Week 2: The Architect (Vision & Transformers)
*Hierarchical feature extraction and spatial intelligence.*
- [ ] **CNN from Scratch:** Implement Convolution and Pooling layers (D2L Ch. 7).
- [ ] **Modern Vision:** Implement ResNet "Skip Connections" to mitigate vanishing gradients.
- [ ] **Attention Mechanisms:** Build Scaled Dot-Product Attention from scratch.
- [ ] **Vision Transformers (ViT):** Replicate the "An Image is Worth 16x16 Words" architecture.

### Week 3: Perception (3D Point Clouds)
*The core domain of the Hao-AI Lab.*
- [ ] **Geometry 101:** Process `.ply` and `.obj` files; implement Furthest Point Sampling (FPS).
- [ ] **PointNet:** Implement Global Feature Aggregation (Su et al. legacy).
- [ ] **PointNet++:** Implement Set Abstraction (local neighborhood) layers.
- [ ] **Graph Neural Networks:** Introduction to **PyTorch Geometric (PyG)**.

### Week 4: Action (Embodied AI & SAPIEN)
*Physics-rich simulation and robotic interaction.*
- [ ] **SAPIEN Setup:** Initialize joints, links, and controllers in the lab's simulator.
- [ ] **RGB-D Sensing:** Extract depth maps and lift 2D pixels into 3D camera space.
- [ ] **Robot Reach:** Integration of PointNet perception with basic robotic manipulation.

### Week 5: Capstone Project
- [ ] **End-to-End Pipeline:** Point Cloud Input $\rightarrow$ PointNet Classification $\rightarrow$ SAPIEN Robot Action.

---

## 📝 Research Logs

### Feb 5, 2026: Hardware Smoke Test
- **Status:** Completed ✅
- **Summary:** Verified hardware acceleration on local device.
- **Outcome:** MatMul benchmark showed a speedup on GPU/MPS vs CPU. Confirmed environment is ready for large-scale tensor operations.
- **Code:** `scripts/device_check.py`

### Feb 6, 2026: Manual Backprop (Upcoming)
- **Goal:** Deriving gradients for $y = Xw + b$.
- **Math:** - $L = \frac{1}{n} \sum (\hat{y} - y)^2$
  - $\nabla_w L = \frac{2}{n} X^T (Xw + b - y)$

---

## 📂 Structure
- `scripts/`: Production-grade research scripts.
- `notebooks/`: Mathematical derivations and experiments.
- `data/`: Local storage for 3D datasets (Git ignored).
- `models/`: Saved model weights (`.pth`).