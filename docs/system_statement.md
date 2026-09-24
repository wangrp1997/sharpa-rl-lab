# 系统陈述与组件清单

**日期：** 2026-09-24

这是要做的系统，不是已经成立的理论。没有公开代码的论文也算可复现组件。要训练策略或触觉图像模型的论文不复现。

## 陈述

真手上拿一个事先不限定的物体，先主动摸一会，再把它玩起来。不先要求完整几何、完整动力学，也不先训练控制策略。

## 当前任务

科学问题写在 `docs/progress.md`。力只读五块指腹，和真手一致。自碰撞距离、关节限位和已有指腹不分开是高优先级不等式，切向步长随转轴标准差缩短，写在 `rl_isaaclab/scripts/belief_turn_step.py`。求解用 `osqp`。合成例子通过。活手入口是 `rl_isaaclab/scripts/belief_turn.py`，还没有跑出日志。`tangent_turn.py` 在环境 41 上没有这条距离约束。`envelop_explore_step.py` 不采用。

## 组件是否完整

| 组件 | 论文 | 代码 | 状态 |
| --- | --- | --- | --- |
| 接触点回归成形状，并带不确定度 | Suresh et al., ICRA 2021，[Tactile SLAM](https://suddhu.github.io/tactile-slam/)。平面推动，高斯过程隐式曲面加因子图 | [suddhu/gpis-touch-public](https://github.com/suddhu/gpis-touch-public) | 有论文、有代码。场景是平面推动，不是五指手内 |
| 形状不确定时规划接触并柔顺合拢 | Li, Hang, Kragic, Billard, *Robotics and Autonomous Systems*, 2015，[Dexterous grasping under shape uncertainty](https://doi.org/10.1016/j.robot.2015.09.008) | [MiaoLi/GPIS](https://github.com/MiaoLi/GPIS) | 曲面和接触级抓取规划可复现。论文里的手型配置概率模型是离线学的，那一块不采用。任务是抓住，不是摸完再玩 |
| 按不确定度决定下一次探针摸哪里 | [AESRM 项目页](https://aesrm.github.io/)。双高斯过程，局部滑动、全局转向高不确定区、失去接触后再贴上。对象是测量机探针 | [FerryRain/Probe_Exlporation](https://github.com/FerryRain/Probe_Exlporation) 是简化试验台 | 有论文、有简化代码。停的标准是形状测全，不是后面的手内操作 |
| 未知物体、边摸边为了任务少摸，摸够再转 | Pan, Lepert, Yuan, Antonova, Bohg, IROS 2023，[arXiv:2210.13403](https://arxiv.org/abs/2210.13403)。ICP 和图优化拼轮廓，高斯过程估形状，贝叶斯优化选下一次要摸的截面。高置信能插入后，用速度控制转到孔的方向。不训练神经网络 | 无作者官方代码。相近环境是 [jc-bao/roller_grasper_tacto](https://github.com/jc-bao/roller_grasper_tacto)，只有手写滚动和 ICP，重建项未完成 | 按论文复现。载体是滚动夹持器，任务是插入。停的判据是“这个方向能进孔” |
| 已知几何下的手内规划与跟踪 | Jiang et al., [arXiv:2505.04978](https://arxiv.org/abs/2505.04978)。CQDC 加微分动态规划，低层触觉跟踪。几何已知，惯量和摩擦近似一次 | [Director-of-G/in_hand_manipulation_2](https://github.com/Director-of-G/in_hand_manipulation_2) | 有论文、有代码。不从未知外形摸起 |
| 接触隐式手内转动 | Kurtz, Castro, Önol, Lin, [arXiv:2309.01813](https://arxiv.org/abs/2309.01813)。Allegro 转球，不预训练策略 | [ToyotaResearchInstitute/idto](https://github.com/ToyotaResearchInstitute/idto) | 有论文、有代码。球的模型在规划器里 |
| 无互补接触模型上的手内控制 | FREE，[arXiv:2408.07855](https://arxiv.org/abs/2408.07855) | [asu-iris/Complementarity-Free-Dexterous-Manipulation](https://github.com/asu-iris/Complementarity-Free-Dexterous-Manipulation) | 有论文、有代码。几何和接触模型给定 |
| 给定接触序列上的轨迹求解 | Crocoddyl | [loco-3d/crocoddyl](https://github.com/loco-3d/crocoddyl) | 有代码。Jiang 的实现已经调用它 |

## 不复现

| 论文或代码 | 原因 |
| --- | --- |
| [arXiv:2308.00576](https://arxiv.org/abs/2308.00576)，多指滑动摸未知外形 | 高斯过程曲面和贝叶斯优化可参考，但触觉图像到高度图要训练 |
| [facebookresearch/neuralfeels](https://github.com/facebookresearch/neuralfeels) | 触觉深度模型要预训练。论文写明重建不用于接着操作 |
| [jingxixu/tandem-public](https://github.com/jingxixu/tandem-public) | 探索和判别一起训练。任务是认出已知集合里的物体 |
| Sharpa 旋转策略、[WM-Craftnet](https://github.com/sharpa-robotics/WM-Craftnet) | 策略或世界模型要训练 |

本仓库 `docs/switch_trial_status.md` 的单步偏航是已知圆柱上的试验，不是这块清单里的组件。

## 真要自己实现的

当前要补的见 `docs/progress.md` 里「只读指腹力」。自碰撞是连杆距离的硬约束，由关节角算出，不读指节力。物体上的点和法向只来自有力的指腹。姿态协方差与关节增量一起求，标准差变小以后切向步长才加长。Sommer 和 Billard 2016 只覆盖已有接触别被命令松开，不覆盖手指之间的距离。他们的力矩控制器还没有接到这只手上。

后面玩起来时不先要物体质量和惯量。2210.13403 用滚轮的运动学避开了这件事。Jiang 仍要一次近似惯量和摩擦。这一句还没有展开。
