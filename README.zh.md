# Python 科学计算演化文档

[![文档状态](https://img.shields.io/badge/docs-latest-brightgreen.svg)](./docs/)
[![语言支持](https://img.shields.io/badge/languages-EN%20%7C%20ZH-blue.svg)]()
[![库覆盖](https://img.shields.io/badge/libraries-15+-orange.svg)]()

> 🔬 **Python 科学计算和数据科学生态系统中主要库的演化、架构和 API 的综合知识库。**

## 📖 概述

本仓库包含精心策划的**演化文档**集合，深入介绍了 Python 科学计算生态系统中核心库的历史、架构和 API 发展。每个文档既是历史记录，也是实用的参考指南。

### 🎯 项目目标

- **历史脉络**: 追踪关键 Python 库的演化和重要里程碑
- **架构理解**: 解释核心概念和设计原则
- **API 参考**: 函数、类和方法的全面文档
- **跨库洞察**: 理解库之间的关系和依赖
- **教育资源**: 支持 Python 科学计算的学习和教学

## 🏗️ 仓库结构

```
docs/
├── computer-graphics/ # 几何建模与渲染
│   ├── en/
│   └── zh/
├── gpu-computing/     # GPU 加速计算
│   ├── en/
│   └── zh/
├── concurrency/       # 异步编程模式
│   ├── en/
│   └── zh/
├── data-processing/     # 数据处理和数值计算
│   ├── en/             # 英文文档
│   └── zh/             # 中文文档
├── deep-learning/      # 神经网络和深度学习框架
│   ├── en/
│   └── zh/
├── ml/                 # 传统机器学习库
│   ├── en/
│   └── zh/
├── nlp/                # 自然语言处理工具
│   ├── en/
│   └── zh/
├── visualization/      # 数据可视化和绘图库
│   ├── en/
│   └── zh/
└── context/           # 项目元数据和指导
```

## 📚 涵盖的库

### 🔢 数据处理与数值计算
- **[NumPy](./docs/data-processing/zh/numpy_evolution_document.zh.md)** - 科学计算基础包
- **[Pandas](./docs/data-processing/zh/pandas_evolution_document.zh.md)** - 数据处理和分析
- **[SciPy](./docs/data-processing/zh/scipy_evolution_document.zh.md)** - 科学计算算法
- **[Dask](./docs/data-processing/zh/dask_evolution_document.zh.md)** - 并行计算和大数据
- **[RMSE 和 R 平方](./docs/data-processing/zh/rmse_r_squared_evolution_document.zh.md)** - 回归模型评估指标

### 🧠 深度学习框架
- **[PyTorch](./docs/deep-learning/zh/pytorch_evolution_document.zh.md)** - 动态神经网络和研究
- **[TensorFlow](./docs/deep-learning/zh/tensorflow_evolution_document.zh.md)** - 生产就绪的机器学习
- **[Keras](./docs/deep-learning/zh/keras_evolution_document.zh.md)** - 高级神经网络 API
- **[CNN](./docs/deep-learning/zh/cnn_evolution_document.zh.md)** - 用于计算机视觉的卷积神经网络
- **[Transformer](./docs/deep-learning/zh/transformer_evolution_document.zh.md)** - 用于序列处理的基于注意力的模型
- **[Transformer 入门指南](./docs/deep-learning/zh/transformer_introduction_for_beginners.zh.md)** - 🌟 面向初学者的 Transformer 介绍 - 无需 ML 背景！
- **[残差连接](./docs/deep-learning/zh/residual_connection_evolution_document.zh.md)** - 深度神经网络的基础跳跃连接架构
- **[图解 Transformer](./docs/deep-learning/zh/the_annotated_transformer.zh.md)** - Transformer 模型的带注释实现
- **[RF-DETR](./docs/deep-learning/zh/rf_detr_evolution_document.zh.md)** - 实时检测与分割模型
- **[RF-DETR Seg (预览版)](./docs/deep-learning/zh/rf_detr_seg_preview_evolution_document.zh.md)** - 实时检测与分割模型 (预览版)
- **[马尔可夫链](./docs/deep-learning/zh/markov_chain_evolution_document.zh.md)** - 不确定性下序列决策的数学框架
- **[Reinforcement Learning](./docs/deep-learning/zh/reinforcement_learning_evolution_document.zh.md)** - 通过交互学习最优行为的智能体算法和理论
- **[强化学习入门指南](./docs/deep-learning/zh/reinforcement_learning_introduction_for_beginners.zh.md)** - 🌟 面向初学者的强化学习介绍 - 无需 ML 背景！
- **[Q-learning](./docs/deep-learning/zh/q_learning_evolution_document.zh.md)** - 用于最优策略学习的无模型强化学习算法
- **[TD-learning](./docs/deep-learning/zh/td_learning_evolution_document.zh.md)** - 用于价值估计的时序差分学习方法
- **[贝尔曼方程](./docs/deep-learning/zh/bellman_equation_evolution_document.zh.md)** - 动态规划和强化学习中的基本方程
- **[贝尔曼方程入门指南](./docs/deep-learning/zh/bellman_equation_introduction_for_beginners.zh.md)** - 🌟 面向初学者的贝尔曼方程介绍 - 无需复杂数学背景！

### 🤖 机器学习
- **[Scikit-learn](./docs/ml/zh/scikit-learn_evolution_document.zh.md)** - 通用机器学习
- **[XGBoost](./docs/ml/zh/xgboost_evolution_document.zh.md)** - 梯度提升框架
- **[LightGBM](./docs/ml/zh/lightgbm_evolution_document.zh.md)** - 快速梯度提升

### 📝 自然语言处理
- **[NLTK](./docs/nlp/zh/nltk_evolution_document.zh.md)** - 自然语言工具包
- **[spaCy](./docs/nlp/zh/spacy_evolution_document.zh.md)** - 工业级自然语言处理

### 📊 数据可视化
- **[Matplotlib](./docs/visualization/zh/matplotlib_evolution_document.zh.md)** - 基础绘图库
- **[Plotly](./docs/visualization/zh/plotly_evolution_document.zh.md)** - 交互式可视化
- **[Seaborn](./docs/visualization/zh/seaborn_evolution_document.zh.md)** - 统计数据可视化

### 💻 计算机图形学
- **[G2 融合算法](./docs/computer-graphics/zh/g2_blending_algorithm_evolution_document.zh.md)** - 几何建模中 G2 连续性的方法
- **[李代数方法](./docs/computer-graphics/zh/lie_algebra_method_evolution_document.zh.md)** - 连续对称性的数学方法
- **[NURBS](./docs/computer-graphics/zh/nurbs_evolution_document.zh.md)** - 计算机图形学中曲线和曲面的数学模型
- **[SLAM](./docs/computer-graphics/zh/slam_evolution_document.zh.md)** - 计算机视觉和机器人中的同时定位与地图构建系统
- **[SLAM 初学者介绍](./docs/computer-graphics/zh/slam_introduction_for_beginners.zh.md)** - 🌟 SLAM温和介绍 - 无需机器人背景！

### 🚀 GPU 加速计算
- **[CUDA](./docs/gpu-computing/zh/cuda_evolution_document.zh.md)** - NVIDIA 的并行计算平台和编程模型

### ⚡ 并发
- **[Async/Await](./docs/concurrency/zh/async_await_evolution_document.zh.md)** - Python 中的异步编程模式

## 🌐 语言支持

所有文档提供**英文**和**中文**两个版本：

- **英文版**: `*_evolution_document.md`
- **中文版**: `*_evolution_document.zh.md`

## 📋 文档结构

每个演化文档遵循一致的架构：

### 🏛️ 标准章节

1. **📜 介绍与历史背景**
   - 库的起源和动机
   - 关键里程碑的演化时间线
   - 社区和生态系统影响

2. **🔧 核心架构**
   - 基本概念和抽象
   - 设计原则和模式
   - 与其他库的集成点

3. **🔍 详细 API 概览**
   - 主要功能区域
   - 核心类和方法
   - 重要变更和弃用
   - API 结构的交互式 Mermaid 图表

4. **📈 演化与性能**
   - 性能改进的时间线
   - 生态系统关系
   - 构建系统和开发成熟度

5. **🎯 结论与未来方向**
   - 在生态系统中的当前重要性
   - 持续发展趋势
   - 未来路线图考虑

### 🎨 视觉元素

- **📊 Mermaid 图表**: 时间线、架构图和 API 思维导图
- **💡 心智模型**: 类比和概念解释
- **✅ 代码示例**: 实际使用演示
- **🔗 交叉引用**: 相关库和概念之间的链接

## 🚀 快速开始

### 📖 对于读者

1. **按类别浏览**: 导航到你感兴趣的领域（数据处理、机器学习等）
2. **选择语言**: 选择英文（`.md`）或中文（`.zh.md`）版本
3. **遵循结构**: 从介绍开始，然后深入具体的 API 章节
4. **使用图表**: Mermaid 图表提供复杂概念的视觉理解

### 🛠️ 对于贡献者

1. **一致结构**: 遵循既定的文档架构
2. **双语更新**: 保持英文/中文版本之间的同步
3. **Mermaid 集成**: 使用图表展示时间线、架构和 API 概览
4. **跨库一致性**: 应用"类比思维" - 在相似库之间传播改进

## 🔧 技术要求

### 📋 先决条件

- **Markdown 渲染器**: 确保你的查看器支持 Mermaid 图表
- **无构建系统**: 纯 Markdown 文档（无需编译）
- **无依赖**: 自包含文档

### 🎯 推荐工具

- **GitHub/GitLab**: 原生 Mermaid 支持
- **VS Code**: Markdown Preview Enhanced 扩展
- **Obsidian**: 优秀的交叉链接和图形视图
- **Typora**: 实时 Markdown 渲染与 Mermaid 支持

## 🤝 贡献

我们欢迎贡献来改进和扩展这个知识库！

### 🎯 贡献领域

- **📝 内容更新**: 保持文档与库变更同步
- **🌐 翻译**: 改进或添加语言支持
- **🎨 可视化**: 增强 Mermaid 图表和视觉元素
- **🔗 交叉引用**: 添加相关概念之间的连接
- **📚 新库**: 建议记录其他库

### 📋 指导原则

1. **遵循 SRP**: 每个文档应有单一、明确的职责
2. **保持一致性**: 使用既定的模式和术语
3. **更新配对**: 保持英文/中文版本同步
4. **测试图表**: 验证 Mermaid 语法正确渲染
5. **交叉引用**: 链接文档间的相关概念

## 📄 许可证

本文档集合作为教育资源提供。提及的各个库受其各自许可证约束。

## 🙏 致谢

本项目基于 Python 科学计算社区的卓越工作。特别感谢所有记录库的维护者和贡献者，他们使这个生态系统成为可能。

---

**📚 开始探索**: 从上面的[涵盖的库](#-涵盖的库)部分选择一个库，或浏览 `docs/` 目录结构。

**🔍 快速导航**: 
- [数据处理](./docs/data-processing/zh/) | [深度学习](./docs/deep-learning/zh/) | [机器学习](./docs/ml/zh/) | [自然语言处理](./docs/nlp/zh/) | [数据可视化](./docs/visualization/zh/) | [计算机图形学](./docs/computer-graphics/zh/) | [GPU 加速计算](./docs/gpu-computing/zh/) | [并发](./docs/concurrency/zh/) | [术语表](./docs/glossary.zh.md)
- [English Docs](./README.md) | [中文文档](./README.zh.md)

**💡 需要帮助？** 查看 [CODEBUDDY.md](./CODEBUDDY.md) 了解详细的仓库指导和约定。