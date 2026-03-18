# Diffusion Model for NILM: Upgrades & Mathematical Formulation

## 1. 连续性增强流水线 (Continuity Booster Pipeline)

**问题定义**：NILM 数据集存在极端的阶级不平衡，绝大多数时间为 OFF 状态。基线模型使用硬截断的滑动窗口 $W = \{x_t, x_{t+1}, \dots, x_{t+L-1}\}$，导致大量窗口内全为 0，模型易陷入预测常数 0 的局部最优。

**数学与算法改进**：
定义采样窗口集合为 $S$。对于训练集中的窗口 $x^{(i)} \in S_{\text{train}}$，计算其最大功率 $P_{\max}^{(i)} = \max_{t} x^{(i)}_t$。
定义激活阈值 $\tau$ (例如 $0.2$)。提取所有活跃窗口：
$$A = \{ x^{(i)} \in S_{\text{train}} \mid P_{\max}^{(i)} > \tau \}$$

对于每一个活跃窗口 $x^{(i)}$，应用 **Jitter 重采样 (Jitter Resampling)**。给定拓展因子 $B$ (例如 4) 和时间平移算子 $\mathcal{T}_{\Delta}$，生成增强集合：
$$A' = A \cup \{ \mathcal{T}_{\Delta_j} x^{(i)} \}_{j=1}^{B-1}, \quad \Delta_j \sim \mathcal{U}(-2, 2)$$

**Pipeline 效果**：该过程仅在**训练期**被触发 (`period='train'`)。增强后的长尾数据极大地丰富了模型对过渡态（启停瞬间）的相位捕捉，迫使模型学习从 0 到 1 的上升沿分布。在评估和采样阶段 (`period='test'`)，强制采用无重叠 (Non-overlapping) 窗口保证测试集无数据泄露。

---

## 2. DiT 微架构级别的重构: AdaLN-Zero (自适应零初始层归一化)

**问题定义**：传统的 `LayerNorm` 只能做静态归一化，基线采用的早期 `AdaLayerNorm` 实现仅利用时间步 $t$ 生成一对 Scale (缩放) 和 Shift (平移) パ数，且模型后期的 MLP 仍使用不可条件的 `LayerNorm`，导致扩散前向/后向轨迹不稳定。

**数学改进**：全网络引入门控机制 (Gating Mechanism) 的 AdaLN-Zero。
给定时间步 $t \in [1, T]$ 的正弦位置编码 $e_t$，以及高维全局周期特征 (Month, Day, Hour) 的表征 $c$，首先得到强化条件：
$$h_t = \text{SiLU}(e_t + c)$$
通过一个线性映射，输出**三个**参数（零初始化该线性层权重 $\mathbf{W}=[0], \mathbf{b}=[0]$）：
$$[\gamma, \beta, \alpha] = \text{Linear}(h_t)$$
对输入特征 $x$ 的归一化应用：
$$\text{AdaLN}(x) = \text{LayerNorm}(x) \odot (1 + \gamma) + \beta$$
在 Transformer Block (Self-Attention, Cross-Attention, MLP) 中的门控融合：
$$x_{\text{out}} = x_{\text{in}} + \alpha \odot \text{Module}(\text{AdaLN}(x_{\text{in}}))$$

**Pipeline 效果**：由于零初始化，在 $t=0$ 时 $\gamma, \beta, \alpha$ 全为 0，初始网络完全作为一个恒等映射 (Identity Mapping) 运行，这避免了扩散过程早期的梯度粉碎。

---

## 3. 多维变量的解耦扩散流水线 (Decoupled Diffusion Pipeline)

**问题定义**：基线模型处理多维输入时，会将所有维度盲目加噪。但 NILM 中的“时间条件”（如 小时正弦特征）是绝对的先验，不应被加噪或去噪。

**数学与前向加噪过程 (Forward Process)**：
给定干净的数据 $x_0 = [P_0; C_0]$，其中 $P_0 \in \mathbb{R}^{L \times 1}$ 为功率， $C_0 \in \mathbb{R}^{L \times 8}$ 为 8 维绝对时间特征。
在任意 $t$ 时刻，加噪仅作用于功率维度 $P_0$：
$$P_t = \sqrt{\bar{\alpha}_t} P_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$
时间条件 $C_t$ 保持绝对纯净：
$$C_t = C_0$$
模型输入变为 $x_t = [P_t; C_0]$。

**逆向去噪过程 (Reverse Sampling)**：
从 $P_T \sim \mathcal{N}(0, I)$ 开始，在每个降噪步骤 $t$：
$$ \tilde{P}_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( P_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta([P_t; C_0], t) \right) + \sigma_t z $$
时间维度通过赋值强制锁定：$x_{t-1} = [\tilde{P}_{t-1}; C_0]$。

---

## 4. 训练损失函数的革新 (Loss Formulation)

相比于原始可能单纯的 L1/L2，升级后的模型采用了针对尖锐波形特别定制的混合 Loss 流水线。

1. **时域 Huber 损失 (Huber Loss in Time Domain)**：
为了对异常尖峰（极大功率值）鲁棒，同时在低功率区域保持 L2 的平滑性，取代纯 L1。
$$ \mathcal{L}_{\text{time}} = \begin{cases} 
      \frac{1}{2}(\hat{P}_0 - P_0)^2 & \text{if } |\hat{P}_0 - P_0| \leq \delta \\
      \delta |\hat{P}_0 - P_0| - \frac{1}{2}\delta^2 & \text{otherwise}
   \end{cases}
$$

2. **频域约束损失 (Fourier Regularization)**：
利用快速傅里叶变换 (FFT) 将时域预测转换为频域，强迫模型拟合真实波形的频率成分。
$$ \mathcal{F}(\hat{P}_0) = \text{FFT}(\hat{P}_0) $$
$$ \mathcal{L}_{\text{freq}} = \text{Huber}(\text{Re}(\mathcal{F}(\hat{P}_0)), \text{Re}(\mathcal{F}(P_0))) + \text{Huber}(\text{Im}(\mathcal{F}(\hat{P}_0)), \text{Im}(\mathcal{F}(P_0))) $$

3. **扩散重加权总损失 (Reweighted Total Loss)**：
$$ \mathcal{L}_{\text{total}} = \mathbb{E}_{t, x_0, \epsilon} \left[ w_t \cdot \left( \mathcal{L}_{\text{time}} + \lambda_{\text{ff}} \mathcal{L}_{\text{freq}} \right) \right] $$
其中 $w_t = \frac{\sqrt{\alpha_t}\sqrt{1-\bar{\alpha}_t}}{100\beta_t}$ 动态权衡各个时间梯度的反向传播。

---

## 5. 工程级加速管道 (Engineering Pipeline)

在硬件调度上超越了基线简单的 `model.cuda()`：
*   **Flash Attention 替换**：全面移除了昂贵的显存密集型 `matmul + softmax`，改用内存安全的 `F.scaled_dot_product_attention` (SDPA)。
*   **混合精度加速 (AMP BF16)**：在 `Trainer` (solver.py) 的 `train()` 与 `sample()` 函数中强制包裹 `torch.amp.autocast('cuda', dtype=torch.bfloat16)`，吞吐量相较 FP32 获取数量级的提升，且避免了 FP16 容易出现的梯度下溢。
*   **底层并发驱动**：默认激活 `torch.backends.cudnn.benchmark = True`，并在可用时利用 `TF32` 矩阵混合张量核心（Tensor Core）计算。
