# DiffusionModel_NILM 完整数据流程图（超详细版）

## 📚 完整模块索引

### Models 目录结构
```
Models/
├── diffusion/
│   ├── agent_transformer.py    ← DiT 主模型 (Transformer骨干)
│   ├── gaussian_diffusion.py   ← Diffusion 包装器 (加噪/去噪逻辑)
│   └── model_utils.py          ← 工具函数 (位置编码, AdaLN等)
└── ts2vec/                     ← (未使用在此流程)
```

### 文件功能说明
| 文件 | 作用 | 关键类/函数 |
|------|------|------------|
| `agent_transformer.py` | DiT 核心 | `Transformer`, `DiTBlock`, AdaLN-Zero |
| `gaussian_diffusion.py` | 扩散包装器 | `Diffusion.forward`, `q_sample`, `p_sample` |
| `model_utils.py` | 工具组件 | `SinusoidalPosEmb`, `AdaLayerNorm`, `extract` |
| `real_datasets.py` | 数据预处理 | `load_csv`, `minmax_scaler`, `create_windows` |
| `solver.py` | 训练/采样控制 | `Trainer.train`, `Trainer.sample` |
| `main.py` | 项目入口 | `parse_args`, `load_config` |

---

## 图例说明

- 🟦 **蓝色框**：数据预处理 (`real_datasets.py`)
- 🟩 **绿色框**：模型前向传播 (`agent_transformer.py`)
- 🟨 **黄色框**：训练/采样控制 (`solver.py`)
- 🟪 **紫色框**：AdaLN-Zero 机制
- 🟧 **橙色框**：入口/配置 (`main.py`)

---

```mermaid
flowchart TB
    %% ========== 入口层 ==========
    subgraph ENTRY ["🟧 入口层 (main.py)"]
        A1["运行命令:<br/>python main.py --config Config/microwave.yaml --sample"]
        A2["解析参数:<br/>• config_path<br/>• --train/--sample<br/>• --sample_num<br/>• --device"]
        A3["加载 YAML 配置:<br/>load_config()"]
        A1 --> A2 --> A3
    end

    %% ========== 数据预处理层 ==========
    subgraph PREPROCESS ["🟦 数据预处理 (Utils/Data_utils/real_datasets.py)"]
        direction TB
        
        subgraph P1 ["步骤 1: 读取原始 CSV"]
            P1A["load_csv(csv_path)<br/>↓<br/>np.loadtxt(delimiter=',', skiprows=1)"]
            P1B["输出: np.ndarray<br/>shape: (N, 9)<br/>9 = 1功率 + 8时间特征"]
            P1A --> P1B
        end
        
        subgraph P2 ["步骤 2: Min-Max 归一化"]
            P2A["minmax_scaler(arr)<br/>↓<br/>对每一列: (x - min) / (max - min + 1e-7)"]
            P2B["输出:<br/>• scaled: (N, 9)<br/>• min_val: (1, 9)<br/>• max_val: (1, 9)"]
            P2A --> P2B
        end
        
        subgraph P3 ["步骤 3: 窗口切分"]
            P3A["create_windows(arr, seq_len=512, style='non_overlapping')<br/>↓<br/>每 512 个点为一个窗口"]
            P3B["输出: np.ndarray<br/>shape: (W, 512, 9)<br/>W = 窗口数量"]
            P3A --> P3B
        end
        
        subgraph P4 ["步骤 4: 转 PyTorch Dataset"]
            P4A["NILMDataset(windows)<br/>↓<br/>torch.from_numpy().float()"]
            P4B["DataLoader<br/>↓<br/>batch_size=64, shuffle=True"]
            P4C["输出: batch<br/>shape: (B, L, 9)<br/>B=64, L=512"]
            P4A --> P4B --> P4C
        end
        
        P1 --> P2 --> P3 --> P4
    end

    %% ========== 训练/采样控制层 ==========
    subgraph SOLVER ["🟨 训练/采样控制 (engine/solver.py)"]
        direction TB
        
        subgraph S_TRAIN ["训练模式: Trainer.train()"]
            ST1["从 DataLoader 获取 batch<br/>shape: (B, L, 9)"]
            ST2["随机采样 diffusion step<br/>t ~ Uniform(0, T-1)<br/>shape: (B,)"]
            ST3["加噪: LinearScheduler.q_sample(batch, t)<br/>公式: x_t = √α̅_t · x0 + √(1-α̅_t) · ε<br/>输出: x_t (B,L,9), ε (B,L,9)"]
            ST4["提取条件向量<br/>c = batch[..., 1:].mean(dim=1)<br/>shape: (B, 8)"]
            ST5["📍 调用模型前向传播<br/>eps_pred = DiT.forward(x_t, t, c)<br/>shape: (B, L, 9)"]
            ST6["计算 MSE Loss<br/>loss = ((eps_pred - ε)²).mean()"]
            ST7["反向传播 & 参数更新<br/>optimizer.step()"]
            
            ST1 --> ST2 --> ST3 --> ST4 --> ST5 --> ST6 --> ST7
        end
        
        subgraph S_SAMPLE ["采样模式: Trainer.sample()"]
            SS1["初始化纯噪声<br/>x = torch.randn(N, L, 9)"]
            SS2["逆扩散循环: t = T-1 → 0"]
            SS3["构造目标时间条件<br/>c = build_condition(target_time)<br/>shape: (N, 8)"]
            SS4["📍 调用模型前向传播<br/>eps_pred = DiT.forward(x_t, t, c)<br/>shape: (N, L, 9)"]
            SS5["逆扩散公式<br/>x_{t-1} = (1/√α_t)(x_t - (β_t/√(1-α̅_t))·ε̂) + σ_t·z"]
            SS6["反归一化<br/>x_real = x_0 · (max - min) + min"]
            SS7["保存为 .npy 文件<br/>np.save('sample_i.npy', x_real[i])"]
            
            SS1 --> SS2 --> SS3 --> SS4 --> SS5 --> SS6 --> SS7
        end
    end

    %% ========== 模型层 ==========
    subgraph MODEL ["🟩 DiT 模型 (Models/diffusion/agent_transformer.py)"]
        direction TB
        
        M_INPUT["🔹 输入:<br/>• x_t: (B, L, 9)<br/>• t: (B,)<br/>• c: (B, 8)"]
        
        %% 条件嵌入
        subgraph M_COND ["条件嵌入模块"]
            direction LR
            MC1["TimestepEmbedding(t)<br/>↓<br/>Sinusoidal + MLP<br/>输出: t_emb (B, hidden_dim)"]
            MC2["Linear Projection<br/>c_emb = Linear(c)<br/>输出: (B, hidden_dim)"]
            MC3["融合条件<br/>cond = t_emb + c_emb<br/>shape: (B, hidden_dim)"]
            MC1 --> MC3
            MC2 --> MC3
        end
        
        %% 输入投影
        M_PROJ["输入投影<br/>x = Linear(9 → hidden_dim)(x_t)<br/>输出: (B, L, hidden_dim)"]
        
        M_INPUT --> M_COND
        M_INPUT --> M_PROJ
        
        %% DiT Block 循环
        M_LOOP["进入 N 个 DiT Block 循环<br/>(N = num_layers, 默认 12)"]
        
        M_COND --> M_LOOP
        M_PROJ --> M_LOOP
        
        %% DiT Block 详细结构
        subgraph DIT_BLOCK ["🟪 单个 DiT Block (包含 AdaLN-Zero)"]
            direction TB
            
            DB_IN["输入:<br/>• x: (B, L, hidden_dim)<br/>• cond: (B, hidden_dim)"]
            
            %% Modulation Network
            subgraph ADALN_MOD ["AdaLN-Zero Modulation Network"]
                AM1["输入: cond (B, hidden_dim)"]
                AM2["SiLU 激活"]
                AM3["Linear(hidden_dim → 6×hidden_dim)<br/>⚠️ 权重初始化为 0"]
                AM4["Split 成 6 份:<br/>shift_msa, scale_msa, gate_msa<br/>shift_mlp, scale_mlp, gate_mlp<br/>每份: (B, hidden_dim)"]
                AM1 --> AM2 --> AM3 --> AM4
            end
            
            %% First Path: MSA
            subgraph MSA_PATH ["路径 1: Multi-Head Self-Attention"]
                direction TB
                MSA1["LayerNorm(x)<br/>elementwise_affine=False<br/>x_norm = (x - μ) / σ"]
                MSA2["🟪 AdaLN 调制<br/>x_mod = x_norm · (1 + scale_msa) + shift_msa<br/>⬅️ 使用 scale_msa, shift_msa"]
                MSA3["Multi-Head Attention<br/>attn_out = Attention(x_mod, x_mod, x_mod)<br/>num_heads = 8"]
                MSA4["🟪 Gate 控制<br/>gated_attn = gate_msa · attn_out<br/>⬅️ 使用 gate_msa"]
                MSA5["Residual 连接<br/>x = x + gated_attn"]
                
                MSA1 --> MSA2 --> MSA3 --> MSA4 --> MSA5
            end
            
            %% Second Path: MLP
            subgraph MLP_PATH ["路径 2: Feed-Forward MLP"]
                direction TB
                MLP1["LayerNorm(x)<br/>elementwise_affine=False"]
                MLP2["🟪 AdaLN 调制<br/>x_mod = x_norm · (1 + scale_mlp) + shift_mlp<br/>⬅️ 使用 scale_mlp, shift_mlp"]
                MLP3["Feed-Forward Network<br/>Linear(hidden_dim → hidden_dim×4)<br/>→ GELU<br/>→ Linear(hidden_dim×4 → hidden_dim)"]
                MLP4["🟪 Gate 控制<br/>gated_mlp = gate_mlp · mlp_out<br/>⬅️ 使用 gate_mlp"]
                MLP5["Residual 连接<br/>x = x + gated_mlp"]
                
                MLP1 --> MLP2 --> MLP3 --> MLP4 --> MLP5
            end
            
            DB_IN --> ADALN_MOD
            DB_IN --> MSA_PATH
            MSA_PATH --> MLP_PATH
            ADALN_MOD -.->|提供 6 个调制参数| MSA2
            ADALN_MOD -.->|提供 6 个调制参数| MSA4
            ADALN_MOD -.->|提供 6 个调制参数| MLP2
            ADALN_MOD -.->|提供 6 个调制参数| MLP4
            
            MLP_PATH --> DB_OUT["输出: x (B, L, hidden_dim)"]
        end
        
        M_LOOP --> DIT_BLOCK
        DIT_BLOCK --> M_NEXT{还有下一个 Block?}
        M_NEXT -->|是| DIT_BLOCK
        M_NEXT -->|否| M_FINAL
        
        %% 最终输出
        subgraph M_FINAL ["最终输出层"]
            MF1["Final LayerNorm<br/>x = LayerNorm(x)"]
            MF2["Output Projection<br/>Linear(hidden_dim → 9)"]
            MF3["🔹 输出: ε̂ (预测的噪声)<br/>shape: (B, L, 9)"]
            MF1 --> MF2 --> MF3
        end
    end

    %% ========== 连接各层 ==========
    ENTRY --> PREPROCESS
    PREPROCESS --> SOLVER
    SOLVER --> MODEL
    
    %% ========== 样式 ==========
    style ENTRY fill:#ffe4b5,stroke:#ff8c00,stroke-width:3px
    style PREPROCESS fill:#e6f3ff,stroke:#0066cc,stroke-width:3px
    style SOLVER fill:#fff9e6,stroke:#ccaa00,stroke-width:3px
    style MODEL fill:#e6ffe6,stroke:#00aa00,stroke-width:3px
    style DIT_BLOCK fill:#f3e5f5,stroke:#9c27b0,stroke-width:3px
    style ADALN_MOD fill:#ede7f6,stroke:#673ab7,stroke-width:2px
    style MSA_PATH fill:#e8eaf6,stroke:#3f51b5,stroke-width:2px
    style MLP_PATH fill:#e8eaf6,stroke:#3f51b5,stroke-width:2px
```

---

## 📋 详细文件功能说明

### 1️⃣ **main.py** (橙色框)
- **作用**：项目总入口
- **关键函数**：
  - `parse_args()`: 解析命令行参数
  - `load_config()`: 读取 YAML 配置
  - `get_dataloader()`: 创建数据加载器
  - `Trainer()`: 实例化训练器

### 2️⃣ **Utils/Data_utils/real_datasets.py** (蓝色框)
- **作用**：数据预处理管道
- **关键函数**：
  - `load_csv()`: 读取原始 CSV（N行×9列）
  - `minmax_scaler()`: Min-Max 归一化，返回 `scaled, min_val, max_val`
  - `create_windows()`: 按 `seq_len=512` 切分窗口
  - `NILMDataset`: PyTorch Dataset 封装
  - `DataLoader`: 批量读取，shape `(B, L, 9)`

### 3️⃣ **engine/solver.py** (黄色框)
- **作用**：训练/采样主循环
- **训练模式 (`Trainer.train`)**：
  - 随机采样 diffusion step `t`
  - 加噪：`q_sample(x0, t)` → 得到 `x_t` 和真实噪声 `ε`
  - 提取条件：`c = batch[..., 1:].mean(dim=1)`
  - 调用模型：`eps_pred = DiT.forward(x_t, t, c)`
  - 计算 Loss：`MSE(eps_pred, ε)`
  - 反向传播：`optimizer.step()`
  
- **采样模式 (`Trainer.sample`)**：
  - 初始化噪声：`x_T ~ N(0, I)`
  - 逆扩散循环：`t = T-1 → 0`
  - 构造条件：`build_condition(target_time)`
  - 逐步去噪：使用逆扩散公式
  - 反归一化：恢复真实功率
  - 保存：`.npy` 文件

### 4️⃣ **Models/diffusion/agent_transformer.py** (绿色框)
- **作用**：DiT 模型核心实现
- **主要组件**：

#### A. 条件嵌入模块
```python
class TimestepEmbedding:
    # 将离散 step → 连续向量
    # Input: t (B,)
    # Output: t_emb (B, hidden_dim)
```

#### B. 输入投影
```python
self.input_proj = nn.Linear(9, hidden_dim)
# Input: (B, L, 9)
# Output: (B, L, hidden_dim)
```

#### C. DiT Block（🟪 AdaLN-Zero 核心）
```python
class DiTBlock:
    def __init__(self):
        # Modulation Network (AdaLN-Zero)
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 6*hidden_dim, bias=True)
        )
        # ⚠️ 最后一层权重初始化为 0
        
        # LayerNorm (无可学习参数)
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        
        # Multi-Head Attention
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads=8)
        
        # Feed-Forward MLP
        self.mlp = ...
    
    def forward(self, x, cond):
        # 1️⃣ 生成 6 个调制参数
        mod = self.modulation(cond).chunk(6, dim=-1)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod
        
        # 2️⃣ 第一路径：AdaLN + Attention
        x_norm = self.norm1(x)
        x_mod = x_norm * (1 + scale_msa) + shift_msa  # 🟪 AdaLN 调制
        attn_out = self.attn(x_mod, x_mod, x_mod)
        x = x + gate_msa * attn_out  # 🟪 Gate 控制
        
        # 3️⃣ 第二路径：AdaLN + MLP
        x_norm = self.norm2(x)
        x_mod = x_norm * (1 + scale_mlp) + shift_mlp  # 🟪 AdaLN 调制
        mlp_out = self.mlp(x_mod)
        x = x + gate_mlp * mlp_out  # 🟪 Gate 控制
        
        return x
```

#### D. 最终输出层
```python
self.final_norm = nn.LayerNorm(hidden_dim)
self.out_proj = nn.Linear(hidden_dim, 9)
# Output: ε̂ (预测的噪声) shape (B, L, 9)
```

---

## 🎯 AdaLN-Zero 的 4 个关键位置

| 位置 | 作用 | 公式 |
|------|------|------|
| **1. Modulation Network** | 根据条件生成调制参数 | `mod = MLP(cond)` → 6 个参数 |
| **2. MSA 前的 Scale & Shift** | 调制归一化后的特征 | `x_mod = x_norm · (1 + scale) + shift` |
| **3. MSA 后的 Gate** | 控制注意力信息流 | `x = x + gate · attn_out` |
| **4. MLP 路径（同上）** | 对 MLP 路径做相同处理 | 同上 |

---

## 🔑 为什么要用 Zero-Init？

```python
# 在 __init__ 中
nn.init.zeros_(self.modulation[-1].weight)
nn.init.zeros_(self.modulation[-1].bias)
```

**原因**：
- 初始时：`scale=0`, `shift=0`, `gate=0`
- 此时 AdaLN 退化为普通 LayerNorm + Residual
- 模型训练稳定，不受未训练的条件干扰
- 随着训练进行，模型逐渐学会如何使用条件信息

---

## 📊 数据维度对照表

| 阶段 | 变量名 | Shape | 文件 |
|------|--------|-------|------|
| **原始 CSV** | `data` | `(N, 9)` | `real_datasets.py` |
| **归一化后** | `scaled` | `(N, 9)` | `real_datasets.py` |
| **窗口化** | `windows` | `(W, 512, 9)` | `real_datasets.py` |
| **Batch** | `batch` | `(64, 512, 9)` | `solver.py` |
| **加噪后** | `x_t` | `(64, 512, 9)` | `solver.py` |
| **条件向量** | `c` | `(64, 8)` | `solver.py` |
| **时间步嵌入** | `t_emb` | `(64, 256)` | `agent_transformer.py` |
| **融合条件** | `cond` | `(64, 256)` | `agent_transformer.py` |
| **投影后** | `x` | `(64, 512, 256)` | `agent_transformer.py` |
| **调制参数** | `scale_msa` | `(64, 256)` | `DiTBlock` |
| **预测噪声** | `ε̂` | `(64, 512, 9)` | `agent_transformer.py` |

---

## 使用说明

在 VSCode 中打开此文件，使用 Mermaid Viewer 扩展即可查看完整流程图。
