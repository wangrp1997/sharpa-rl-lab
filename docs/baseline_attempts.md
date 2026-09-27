# 非学习基线

环境都是现有的 Isaac 里这只 Sharpa Wave，物体是半径 2 cm 的圆柱。初始化是现有脚本那套：256 个环境里先找到一只向下能握住的手，再冻住。种子 `1999979387` 时冻在环境 108。力只读五块指腹。

## 分支

| 分支 | 内容 | 结果 |
| --- | --- | --- |
| `cursor/pad-force-belief-turn` | 运动学循环。已有接触沿切向走，围不住中心或没有增量就停。提交 `7c3c19f`。 | 手指贴着物体推，或者规划没有增量、圆柱下落就停。没有连续转起来。 |
| `baseline` | 非学习基线。IDTO、DROP、FREE、Jiang。原先叫 `cursor/idto-sharpa`，中间一度叫 `cursor/baseline-drop`。 | 求解器用他们的开源程序。这只手上的指尖抓取还没有连续转完。 |

## DROP

Li、Culbertson、Kurtz、Ames，arXiv:2409.14562。规划器在 [alberthli/mujoco_mpc 的 leap-hardware 分支](https://github.com/alberthli/mujoco_mpc/tree/leap-hardware)。接到这只手上的入口是 `rl_isaaclab/scripts/drop_mjpc_turn.py`。`drop_turn.py` 是自行重写的采样，不作这个结果。交叉熵的采样设置没有改，标准差仍是 0.5 rad。

种子 `1999979387`，环境 108，冻住时拇指和中指有力。模型里手停在当前关节可以托住圆柱。送进 Isaac 的第一步，关节目标最大偏了 0.527 rad，五块指腹力变成 0，圆柱掉到 0.564 m，转角 0.098 rad。日志 `logs/live_diag/drop_mjpc_turn.jsonl`。

## IDTO：失败

Kurtz 等，arXiv:2309.01813。公开例子在 Drake，不在这套 Isaac 里。手仍是这只 Sharpa，圆柱用抓取缓存第 0 行。开环 50 步没有收敛。圆柱贴在掌背外侧，不在指腹中间。仿真放到约 1.2 秒时接触求解发散。录像 `logs/to_delete/idto_sharpa.mp4` 只有前 1 秒，手指几乎不动。

## FREE

Jin，arXiv:2408.07855。开源求解器是 `MPCExplicit`，接到这只手上的入口是 `rl_isaaclab/scripts/free_explicit_turn.py`。`free_turn.py`、`free_explicit.py`、`fingertip_turn.py`、`fingertip_explicit.py` 是自行重写，不作这个结果。代价保持圆球到物体中心、抓取闭合、位置权重 500、控制权重 50。接触指取冻结时力大于 0.5 N 的指腹。

同一种子，环境 108，这一次是拇指和中指。第一步求解成功，看到 2 个接触，关节增量范数 0.019。五块指腹力变成 0，圆柱掉到 0.584 m，转角 0.135 rad。日志 `logs/live_diag/free_explicit_turn.jsonl`。

## Jiang

Jiang 等，arXiv:2505.04978。开源求解器是 Drake qsim 加 Crocoddyl，接到这只手上的入口是 `rl_isaaclab/scripts/jiang_qsim_turn.py`。`jiang_turn.py` 和 `jiang_mpc.py` 是自行重写的前向模型，不作这个结果。代价和关节增量界限 ±0.08 rad 没有改。接触指取冻结时力大于 0.5 N 的指腹，球在开始时压一次。

同一种子，环境 108，这一次是拇指和中指。40 步里转角从 0.123 rad 升到第 23 步的 1.001 rad，高度留在 0.618 m 附近，这两指仍有力，然后退回到 0.513 rad。之后几次把球重新压回当前指腹，峰值更低，物体还掉过，那些改动已撤回。日志 `logs/live_diag/jiang_qsim_turn.jsonl`。
