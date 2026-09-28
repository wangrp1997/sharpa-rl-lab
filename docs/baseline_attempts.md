# 非学习基线

环境都是现有的 Isaac 里这只 Sharpa Wave，物体是半径 2 cm 的圆柱。初始化是现有脚本那套：256 个环境里先找到一只向下能握住的手，再冻住。种子 `1999979387` 时冻在环境 108。力只读五块指腹。

## 分支

| 分支 | 内容 | 结果 |
| --- | --- | --- |
| `cursor/pad-force-belief-turn` | 运动学循环。已有接触沿切向走，围不住中心或没有增量就停。提交 `7c3c19f`。 | 手指贴着物体推，或者规划没有增量、圆柱下落就停。没有连续转起来。 |
| `baseline` | 非学习基线。IDTO、DROP、FREE、Jiang。原先叫 `cursor/idto-sharpa`，中间一度叫 `cursor/baseline-drop`。 | 指尖握持已经形成、圆柱悬空时，三个求解器都没有把圆柱持续转下去。 |

## 科学问题

指尖握持已经形成，圆柱悬空，接触指不预先指定，而由冻结时力大于 0.5 N 的指腹决定。在不改写求解器的前提下，只替换手与物体，DROP、FREE 和 Jiang 能否保持这组接触并把圆柱持续转下去？

已测握持是种子 `1999979387`、环境 108，接触为拇指和中指。关节命令按名称从规划器的 URDF 顺序映到 Isaac 的关节顺序，再加到当前关节目标上。

三个求解器都没有保持这组接触并持续旋转。Jiang 与 FREE 的增量主要是侧摆：侧摆约 0.2 rad 后，指腹离开圆柱，或圆柱被推到另一根手指上，航向退回。DROP 的增量是继续屈曲，航向往反方向变化。将同一对指腹约束在冻结时的径向距离上沿切向滑动，航向约到 0.7 rad 后不再增加；中指抬起并前摆后再闭合，净转角没有继续积累。

## DROP

Li、Culbertson、Kurtz、Ames，arXiv:2409.14562。规划器在 [alberthli/mujoco_mpc 的 leap-hardware 分支](https://github.com/alberthli/mujoco_mpc/tree/leap-hardware)。接到这只手上的入口是 `rl_isaaclab/scripts/drop_mjpc_turn.py`。`drop_turn.py` 是自行重写的采样，不作这个结果。采样标准差改为 0.04。送进 Isaac 的增量限制在 ±0.02 rad，没有加载的手指保持不动，已加载的手指不许张开。

同一种子，环境 108。12 步里转角从 0.112 rad 降到 0.033 rad，高度留在 0.618 m，拇指和中指的力升高。留下的动作是继续屈曲。日志 `logs/live_diag/drop_mjpc_turn.jsonl`。

## IDTO：失败

Kurtz 等，arXiv:2309.01813。公开例子在 Drake，不在这套 Isaac 里。手仍是这只 Sharpa，圆柱用抓取缓存第 0 行。开环 50 步没有收敛。圆柱贴在掌背外侧，不在指腹中间。仿真放到约 1.2 秒时接触求解发散。录像 `logs/to_delete/idto_sharpa.mp4` 只有前 1 秒，手指几乎不动。

## FREE

Jin，arXiv:2408.07855。开源求解器是 `MPCExplicit`，接到这只手上的入口是 `rl_isaaclab/scripts/free_explicit_turn.py`。`free_turn.py`、`free_explicit.py`、`fingertip_turn.py`、`fingertip_explicit.py` 是自行重写，不作这个结果。代价改为保持冻结时各指腹到圆柱轴线的距离，并跟踪转角。只动冻结时有力的手指，已加载的手指不许张开。

同一种子，环境 108。40 步里转角从 0.124 rad 升到第 22 步的 0.260 rad，然后退到 0.026 rad。拇指和中指一直有力，退回时无名指被顶上约 1.5 N，高度留在 0.610 m 附近。每步主要在加大拇指侧摆。日志 `logs/live_diag/free_explicit_turn.jsonl`。

## Jiang

Jiang 等，arXiv:2505.04978。开源求解器是 Drake qsim 加 Crocoddyl，接到这只手上的入口是 `rl_isaaclab/scripts/jiang_qsim_turn.py`。代价和关节增量界限 ±0.08 rad 没有改。接触指取冻结时力大于 0.5 N 的指腹，球在开始时压一次。

同一种子，环境 108。已加载手指的张开分量被去掉。40 步里转角从 0.124 rad 升到 0.520 rad，然后退到 0.046 rad。往上转的增量主要是拇指和中指侧摆。中指侧摆累加约 0.2 rad 后力掉到 0，拇指随后也离开，圆柱往旁边移。日志 `logs/live_diag/jiang_ordered_40.jsonl`。

另外让这两个指腹沿圆柱切向滑，并且不许张开。双指接触还在时，航向大约到 0.7 rad 后停住。把中指抬起再往前摆，净转角没有继续累加。那不是求解器自己的增量。
