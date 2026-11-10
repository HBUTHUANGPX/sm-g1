# SuperMimic 官方代码与当前 ManagerBased G1：逐项审计

审计日期：2026-11-11。**结论：当前工程与官方公开版本共享“学习世界模型，再穿过模型优化策略”的方法框架，但并非训练语义等价的实现。差异不限于 Isaac Lab/Newton 与 mjlab 的接口。** 状态表示、目标窗口、策略输入、预测量、模型损失、优化器、采样分布与 recovery 终止目标都有明确差异。

这份审计没有修改训练实现、恢复正式训练或执行 GPU 作业。报告及结构核查结果是本次唯一工程变更。不能由源代码差异推导哪边训练效果更好，也不能把当前官方 HEAD 自动视为论文实验所用的确切版本。

## 1. 固定版本与证据边界

- 官方：[HybridRobotics/supermimic，5efbf114e11f725c31d83e217be987c13c0e4ce2](https://github.com/HybridRobotics/supermimic/tree/5efbf114e11f725c31d83e217be987c13c0e4ce2)。该提交日期 2026-11-10。
- 本地：[HBUTHUANGPX/sm-g1，83120bf0722e54dce47fc68a146439140d32d8c2](https://github.com/HBUTHUANGPX/sm-g1/tree/83120bf0722e54dce47fc68a146439140d32d8c2)。审计原工程 ManagerBased G1 路径，不使用已移除的 standalone 实现代替。
- 官方主对照：`SuperMimic-GenCoord-G1`、默认 `SuperMimicRunnerCfg`、actor-critic、motion slot 1.0。MaxCoord 和可选 proprio/history 分支单独注明。
- 本地主对照：G1 CLI 的非 small-model、无感知、recovery 路径；已记录的 Newton 训练形状为 2560 env / policy 4096×32 / WM 2048×8×10。**裸 CLI 新建任务默认 64 env、10 iter，TrainingConfig 类默认 4096 env，都不是此前正式运行的 2560 env。** Resume 以 checkpoint 配置为准。本次未重新读取远端 checkpoint；既有运行值来自仓库中的已验证性能报告。
- 官方 `pyproject.toml` 和 `uv.lock` 使用未固定 commit 的相邻 `../mjlab_research`。README 指向 `kevinzakka/mjlab_research`；本次 Git 访问返回 Repository not found，已连接 GitHub API 也返回 404。因此继承的机器人资产、真实仿真参数、MotionLoader、reset 和 PD 参数存在明确证据缺口。**没有用公开 mjlab 的其他版本冒充这个依赖。**

证据分级：**确定差异**＝可见源码/公式证明；**相同框架或参数**＝局部一致，非全链路一致；**有意扩展**＝符合已有需求但偏离官方；**未核实**＝依赖/数据/实际运行缺失。详细双边文件与行号见三个附录。

## 2. 最重要的差异矩阵

| 项目 | 官方当前 GenCoord 默认 | 当前本地 G1 recovery | 判定与影响 |
|---|---|---|---|
| 仿真框架 | mjlab ManagerBased；具体底层参数继承缺失 fork | Isaac Lab 3.0 ManagerBased；已测 Newton MJWarp | 接口适配不等于动力学一致 |
| 状态表示 | 29 关节时 full state 71D（四元数），invariant rep 68D | packed state 73D（6D旋转），解包后旋转为3×3；invariant 71D | 无法仅按同维 tensor 对齐 |
| WM 输入/输出 | 130→35：68 state+29 action+33 context；预测速度增量 | 162→73：71 invariant+当前/上次各29 action+33 context；位姿及速度 residual | **确定算法差异** |
| WM 积分 | 速度增量，经 decimation-aware 积分生成位姿 | 速度积分外另加位置/关节残差、6D 旋转残差 | 假设空间不同 |
| Actor 输入默认 | rep_state 68D×1；含根高度和线速度 | 有噪声本体观测93D×10；无根高度/线速度直接输入 | 信息条件不同，不能直接比较结果 |
| Actor 可选模式 | 提供 proprio 93D 和 history 配置 | 当前固定使用 proprio/history10 | 选项存在不表示官方默认已匹配 |
| 目标帧偏移 | 0,5,…,45 | 1,2,…,10 | 50Hz 假设下 0–0.9s vs .02–.20s |
| Actor goal | 640D：q、qd 和每个目标相对当前机器人根的旋转 | 730D：参考 heading 下的位姿、v/w、q、.1qd；decoder 额外接收当前相对旋转6D | 不只是维度/排列差异 |
| Decoder 输入 | 默认100D＝32 latent+68×1 | 968D＝32 latent+93×10+6 | 第一层参数及可观测性不同 |
| 网络非线性 | WM/actor/critic 全部 GELU+LayerNorm | WM SiLU+LN；actor/critic ELU，无LN | 隐藏层宽度相同也不等价 |
| 策略任务结构 | 多 encoder，共享 decoder；默认仅 motion 槽有效 | 单 motion encoder+decoder | velocity 默认0槽，不能误称默认多任务训练 |
| WM 训练损失 | 按物理通道 nominal error 加权的均值；8个状态监督7次转移 | 本地状态误差 SSE/time-sum；监督8次转移及mask/权重 | 数值尺度、预测长度和各项梯度不同 |
| 优化器 | 内部 Linear 权重 Muon，其余 Adam | 所有参数 Adam，cosine LR | **确定算法差异** |
| 策略梯度裁剪/Adam betas | norm1，另逐元素clip .1；actor beta2=.95 | 同样 norm1、逐元素clip .1；actor beta2=.95 | 这一局部一致；差异在Muon分组及LR调度 |
| Critic | GAE；5×4更新；MSE | GAE；5×4更新；horizon×MSE，并有有效性权重 | H=32时基础损失缩放差32倍，不代表参数步长恰差32倍 |
| Replay/采样 | episode-safe均匀序列；return-based RSI | bin-aware/reweighted replay；MasterySampler RSI | 数据分布和起点分布均有差异 |
| Recovery | 可见官方 overlay/WM 没有本地这套四组概率保护方案 | 15%大姿态reset，4s保护，四组概率终止 | 有意扩展，不能称官方默认 |
| 想象终止 | 两误差归一化max后sigmoid | 四组概率OR、保护gate | 直接改变BPTT存活权重和目标 |
| 动作延迟/TN | 继承依赖不可得，不能下结论 | 0–4物理tick独立delay；静态PD，不启用TN | TN是用户明确排除；非发现的bug |
| 数据集 | 默认具名W&B `lafan_run1_subject5` artifact，可传多个本地文件 | 固定来源的40段G1 LAFAN1 CSV转换 | 未证明相同数据/预处理 |
| 评测 | play/viewer；benchmark是吞吐测试 | all40 strict与full_clean，全原始长度 | 当前指标不能直接称官方benchmark成绩 |

维度按29关节、33D sim context推导。官方具体资产需缺失依赖验证；本地维度通过实际 CPU 构建网络核对。官方网络没有在缺失依赖下实例化运行。

## 3. 哪些确实相同，哪些不能因此推定相同

双方都包含在线仿真采集、replay、监督世界模型、多步想象、穿过冻结世界模型对 actor 求梯度、critic bootstrap/GAE。对应默认配置也有若干一致：WM隐藏层2048×3，actor隐藏层2048/1024/512/256，latent32，critic256/128，WM batch2048、10次更新、horizon8，policy batch4096、horizon32，actor-critic gamma .99、lambda .95，初始LR 2e-4，探索噪声 .1。

但这些参数相同不足以证明同一训练：预测通道、坐标系、目标时间窗、loss归约、优化器和终止权重均不同。奖励的六个指数项、两类惩罚和数值系数接近/相同，也仍需检查 body集合、重定位锚点、参考时序和仿真reward参数是否一致。

## 4. 网络尺寸核查

| 模块 | 本地实际构建参数量 | 官方源码结构推导参数量 |
|---|---:|---:|
| World model | 8,888,393 | 8,744,995 |
| Actor：motion路径 | 9,005,885 | 7,059,261 |
| Actor：所有已构建encoder+decoder | 9,005,885 | 9,837,661 |
| Critic | 267,521 | 231,425 |

官方额外 velocity encoder 为2,778,400参数，即使默认slot为0仍构建；不能把所有参数量当成每步motion路径实际计算量。官方算术按 `build_mlp` 的 Linear+LN结构计算，不包含没有可训练参数的归一化buffer；不是官方模型已成功加载或训练的证据。本地实例来自 `runtime/configuration.py:32-45` 的 `build_g1_models`，开启recovery及相对朝向反馈，没有使用small model。

[结构核查JSON](2026-11-11-official-supermimic-shapes.json)保存尺寸、参数量及验证边界。

## 5. Iteration、UTD与有效训练预算不能直接对应

官方外层 iteration 是一次真实向量环境step，再按Ratio决定是否做一轮学习；本地 iteration 是一次完整学习包，先采集平均2.5个向量step，再做10次WM、1次actor、20次critic更新。

官方学习包/向量step = `num_envs / (10*2048*8) * replay_ratio`，默认 replay_ratio=16：

- 4096 env时为0.4，与本地固定utd=.4的学习包比例相符。
- 2560 env时官方会变成0.25；本地utd仍为0.4。于是本地每轮学习对应6400条真实transition，WM名义采样transition槽位163840（mask后有效监督数可能更少），名义比例25.6；不是官方名义目标16。官方每个8状态窗口实际只监督7次转移，因此其实际被评分的转移消费还需乘7/8；不能把名义ratio当完全相同的监督计数。
- 忽略启动与取整，本地3000次学习包约对应7500次真实向量step，不能把两边“3000 iter”当同等计算/数据预算。

这是调小env数量时必须注意的比较条件，不等于建议立即修改已训练checkpoint。对照应共同报告真实transitions、WM/actor/critic更新数、总wall time和有效motion数据量。

## 6. Recovery与termination的目标差异

本地reset包括普通RSI扰动和15%大姿态分支；大姿态roll/pitch独立均匀采样、速度清零、碰撞包围角点保证初始离地间隙。4秒保护只作用于recovery-role样本。四组概率接受则作用于recovery配置下的所有真实训练样本：height/tilt每组每步dt/3，ankle/wrist每组dt/.5；评测strict不使用这些稀释概率，而按原始阈值接受失败。

官方可见WM终止为：`sigmoid((max(height/threshold_h, tilt/threshold_t)-1)/.05)`。本地为：`gate * (1 - product(1 - p_i*sigmoid_i))`。官方在归一化阈值处为.5，大偏差趋近1；本地即使四组都严重越界，在dt=.02时每步概率上限也只有约.090647，而保护内为0。因此actor面对的是不同的生存目标，不能用“都是soft termination”概括为同一种实现。

真实官方环境的确切硬阈值/reset/概率机制来自缺失的fork，不能据WM可见代码反推其全部真实环境行为。本地方案是先前明确要求的recovery迁移，TN不加入也是明确约束，不应将这些有意差异自动列为错误。

## 7. 数据、运动学与评测

本地CSV转换含30→50Hz插值、四元数SLERP、差分速度、按名字映射29关节，并在load时用URDF双精度FK重算14个tracked body。训练会按25-transition bin补齐末段，正式评测用保留的original end。官方MotionLoader、重采样、资产body/joint顺序及FK来自缺失fork；未拿到它和确切artifact前，无法完成逐帧输入/FK数值等价检验。

本地strict关闭随机化/观测噪声，以完整tracking termination统计成功/失败；full_clean关闭全部termination及delay等干扰，完整走完原轨迹。full_clean覆盖40/40不是跟踪成功40/40；strict提前失败的RMSE只覆盖执行前缀，不能直接和full_clean全轨迹RMSE比较优劣。官方`benchmark.py`报告吞吐，不能据名称作为同一质量评测协议。

另外发现需要单独核查的**参考时序差异**：官方WM用缓存的当前reference给预测next-state评分，然后推进motion_times；本地按next-reference评分。可见调用顺序确有不同，但依赖MotionLoader/真实环境的索引语义尚缺，不直接判某一侧off-by-one错误。

### 已复现的本地内部不一致：reset历史噪声

`learning/collection.py:27-47` 在reset后把同一帧已加噪观测复制到整个history；`learning/replay.py:1069-1075` 先复制缺失历史的原始状态，而 `learning/policy_update.py:122-126` 对每个history槽独立加噪。因此真实与想象输入的padding噪声相关性不同。

使用生产 `MDP.add_noise`、CPU、seed17、128样本×10帧×93维复现：真实式padding各槽最大差为0；想象式padding最大差约0.998397（来自允许±.5噪声的速度通道）。这不是噪声超限，而是本应代表同一帧的重复槽被重新采样。官方可选history warm-start对旧帧padding显式共享噪声，可借鉴其处理，但官方默认history1不触发此多帧问题。应单独加回归测试并修正后做短训验证；本次只报告，没有修改。证据见[CPU探针](2026-11-11-official-supermimic-padding-noise.json)。

## 8. 性能优化与来源边界

官方学习侧显式使用torch.compile、CUDA graph捕获更新和共享内存池；本地最新修改提供批量actuator、lean/bounded FK及checkpoint重算，但默认训练执行仍为reference。未实跑官方，不能用代码结构推断其吞吐一定更高，也不能将本地此前的短测速度与官方benchmark直接作比。

本地NOTICE明确注明来自unofficial supermimic_vibed的部分实现，当前官方为Apache-2.0。本地代码中“paper contract”“reference code”等注释不构成与这个官方commit一致的证据；需要依据实际公式核对。差异可能早于G1迁移，不能在缺少逐项Git溯源时一概归因于G1适配。历史RX-P2/公司私有材料不属于本次审计和发布范围。

## 9. 对后续工作的建议顺序（本次不实施）

1. **先选对齐目标**：官方当前默认GenCoord，还是保留当前proprio/history10/recovery的工程需求。两者不能同时声称原样一致。
2. **补齐证据**：官方准确mjlab_research commit、motion artifact及其哈希、训练run配置；导出双方joint/body映射、dt/decimation、PD/limits、DR/reset/termination参数。
3. **先锁数据接口**：同一段CSV/NPZ逐帧比对状态、goal、动作变换、FK、reward、termination。加上确定性的小样本梯度对照。
4. **再做分组对照**：goal窗口与信息输入；WM预测/积分/loss；优化器与更新预算；sampler；recovery。不要一次整体替换后归因某一项。
5. **统一评测协议**：相同40段与原始长度、相同噪声/delay/termination条件、同样的结果归约。记录训练transitions和更新次数，不能只对齐iter。

当前可以确认的结论是“存在多处实质训练语义差异”；尚不能确认“本地实现错误”“官方必然更强”或“已完成仿真/训练数值等价验收”。

## 10. 文件级证据附录

- [学习算法、网络、损失、优化器、replay、更新预算](2026-11-11-official-supermimic-learning.md)
- [环境、PD、随机化、观测、reset、recovery、termination](2026-11-11-official-supermimic-environment.md)
- [数据、目标窗口、采样、FK、评测及来源](2026-11-11-official-supermimic-data-evaluation.md)

验证范围：双方固定commit源码逐项审阅；本地实际CPU构建网络并核对参数量、复现历史padding噪声差异；官方参数按可见层结构算术核对；依赖可达性双重检查；没有在缺失依赖下声称官方测试通过或运行完成。
