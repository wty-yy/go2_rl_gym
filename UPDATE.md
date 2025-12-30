# 20251230
## v0.1.1
1. 给cts算法加入robogauge异步评估
2. 加入MCP-CTS
Fix Bug: 修复MoE中专家使用了共享权重的问题, 换成Conv1D
# 20251221
1. 修改最大地形速度限制, y在所有地形上最大为1.0, z只有平地最大为2.0, x最大为2.0
2. 上调难度9地形难度 (都是moe-cts 100k能通过的难度):
    1. Slope: 19.8度 -> 29.6度
    2. Stairs height: 0.212m -> 0.257m
    3. Obstacles height: 0.23m -> 0.275m
# 20251204
1. 添加不同地形的最大速度限制
2. 加入`legs_distance`惩罚两腿横向距离过近的问题, 通过robogauge指标验证发现没有提升, 删除
# 20251128
1. go2.xml添加侧向和背面初始化 (需手动解开注释)
2. cts, moe-cts算法中添加norm_layer选项, 将latent_dim提升到512, norm_layer改为simnorm (dim=8)
3. 默认的layer norm还是用l2norm, latent_dim=32, 地形使用偏平地的地形, 因为开悟测评的分数还是更高, 平地也能走的更好
# 20251127
1. 修改deploy_go2.py支持xbox手柄指令输入
2. 修改turn_over开关, 默认关闭
# 20251126
## v0.1.11
1. 加入`turn_over`具体内容如下 (翻身roll为180度, 侧身roll为90度, 随机两个方向):
    1. `init_state.turn_over_proportions`: 支持初始化时, 处于翻身/侧身的概率分别为0.1/0.3 
    2. `init_state.turn_over_init_heights`: 翻身/侧身的机的初始高度范围, 在该范围中随机采样
    3. `commands.turn_over_zero_time`: 翻身和侧身翻转过来预计用时, 在这段时间内给出的command都是全零
    4. `rewards.turn_over_roll_threshold`: 翻身奖励计算的roll阈值, 当roll.abs小于该阈值时转为计算原奖励
    5. `reward.turn_over_scales`: 翻身奖励配置, 目前只使用了`_reward_upright`奖励计算负base的重力投影z分量和z轴单位向量做差, 系数1.0
2. 在环境数为4096时, 如果想达到8192的训练效果(尤其是上楼梯), 需要增大num_steps_per_env, 训4096时先加到48
3. 发现部分地形分数较低: wave, stairs up, down, 修改地形比例配置, 增大这三个地形, 降低flat, slop
4. 一开始从翻身开始是学不到, 只能从侧身开始学
# 20251125
## v0.1.10
1. 添加robot_properties的低频域随机化更新  (后来发现无法使用, 会导致机器人本体信息错误, 删除)
2. 消融修改num_steps_per_env=48,72,96
# 20251124
## v0.1.9
1. 修改action delay从0/1进行延迟, 改为{0,5,10,15}ms上进行延迟
2. 修改motor_offset和pd的域随机化从开始一次, 改为每次reset环境时随机
# 20251120
## v0.1.8
1. friction为0-2有提升, 在瓷砖地上可以上楼梯了
2. 将dynamic_sigma线速度最大值调到1.5, max_sigma等比例增大
3. rsl_rl支持保存全部的env_cfg和train_cfg配置文件, 便于参数检查
4. 加入MoE-CTS算法, 新任务go2_moe_cts, 消融load_balance_loss正则损失
# 20251119
## v0.1.7
1. 修复rsl_rl中剩余训练时间计算的小bug
2. 发现correct_base_height的奖励过小, 只有1e-3级别, 其他都是1e-2的, 可能导致滑倒时的重心偏移, 加入0-1500训练步的奖励课程系数1-10变化
3. 总训练步数调到50k
4. 域随机化: restitutionb不变0-0.5(isaacgym计算弹性系数是(地形0+机械人弹性系数)/2, 所以也不会很大), friction扩大到0.3-1.7 -> 0-2 (isaacgym中的摩擦计算方法是(地形1+机械人摩擦)/2, 因此可以开的范围更大点模仿瓷砖地)
5. 边界速度对向速度变换: 当连续两次切换边界速度时, 直接将边界速度取反
6. exporter加入新的pkl模型参数导出功能
# 20251118
## v0.1.6
1. 修改边界速度采样, 启用x正/负方向和y正/负方向最大值采样
2. 将heading_commands的角速度最大速度设置为当前的角速度范围 (原来是-1,1)
3. 加入command_range_curriculum在20000步时将最大速度范围修改为-1,1
4. 加入dynamic_sigma动态调整追踪奖励系数:
    设动态调整$\sigma$的速度绝对值范围为$[v_{min}, v_{max}]$（角速度同理），第$i$种地形在最大速度下的速度追踪系数记为$\sigma_{max}^{terrain_i}$，则当指令速度为$v_{x}$（y轴方向速度同理）在第$i$种地形下时，当前系数为
  $$\sigma_{vel} = \begin{cases}
    \sigma,&\quad v_x\in[0,v_{min}),\\
    \sigma(v_x-v_{min})+\sigma_{max}^{terrain_i}(v_{max}-v_{xy}),&\quad v_x\in[v_{min},v_{max}),\\
    \sigma_{max}^{terrain_i},&\quad v_x\in(v_{max},\infty).
    \end{cases}$$
  $$\sigma_{now}=\sigma+\min(e^{\frac{\text{level}_i+1}{10}}-1,1)(\sigma_{vel}-\sigma)$$
5. 上真机发现在1.0速度时突变速度可能发生打滑的问题, 参考CTS论文给到`[0.3,1.7]`, 并消融`[0.5, 3.0]`
6. 真机的原地转向效果很差, 没有到1 rad/s, 关闭heading_command, 并将原来的limit_lin_vel改为limit_vel, 其中加入对最大角速度的最大最小和0的组合; 在原地禁止时, 加入20%的概率已边界角速度原地旋转
# 20251117
## v0.1.5
1. 加入动态调整resample command逻辑, 保证xy线速度能够出地形的一半距离:
    设当前是$n_r$次采样指令，第$i$次的xy方向命令为$c_i^{xy}$，$T_r$ 为采样间隔时间，$T$为episode时间，则第$n_r+1$次命令最小xy速度为
    $$v_{min} = \text{clip}\left(\frac{5-||\sum_{i}^{n_r}c_i^{xy}||_2T_r}{T-n_rT_r},0,v_{max}^{x\ or\ y}\right)$$
    第$n_r+1$次有$p_{zero}$概率(课程增加到0.1)使xy零速度，其持续时长为
    $$T^{zero} = \text{clip}\left(T-n_rT_r-\frac{5-||\sum_{i}^{n_r}c_i^{xy}||_2T_r}{0.8\times\max(v^{max}_x,v^{max}_y)},0,T_r\right)$$
2. 移除当翻滚和俯仰导致的异常终止
3. 修改地形等级变化的distance, 为当前episodex距离中心点最大的distance
4. 加入地形之间的terrain_spacing=0.5m
5. 之前训练中将上下楼梯比例反了
6. 重新打开heading_command范围(-pi,pi), 并在reset时随机朝向
7. 调整地形比例降低wave, 更偏向平地斜坡
8. 加入一种边界线速度采样, 对x正方向和y正负方向最大值采样, 概率0.1, 当才到边界速度时取消角速度和heading指令
9. 加入域随机化:
    1. `randomize_restitution=[0, 0.2]`机器人的弹性系数, 仅在初始化机器人时计算
    2. `randomize_motor_strength=[0.8, 1.2]`机器人随机电机力矩系数, 仅在env.reset时计算
    3. `randomize_action_delay`以0.5概率使用last_action, 每个env.step都计算
10. 提高域随机化`friction_range=[1, 5]`能明显提高真机前进效果 (消融了`[0.5, 3]`, 结果接近, 在变速时可能摔倒)
11. obs中加入`feet_contact_forces, dof_torques, dof_acceleration: 4, 12, 12`
# 20251116
## v0.1.4
1. 用平地奖励训练出来的模型会有x速度过大产生高抬腿动作, 同时给xy也会产生高抬腿, 重心偏移摔倒的问题, 说明加入更多的域随机化也没解决
2. 发现用CTS训练的模型能在mujoco中上20cm台阶, 但是由于双腿之间间距过小导致真机迁移效果非常差, 完全无法正常移动, 但是也能爬箱子
3. 考虑用CTS奖励进行训练, 加上similar_to_default或hip_to_default, 消融两种奖励, 以及对应的奖励系数, 0.01还是0.05
4. 真机上有明显的跺脚动作, 加上feet_contact_forces惩罚接触力大小, 设置接触力阈值大小为go2的重量15*9.8=147, 系数为-1 (参考collision)
5. 修复resume时第一个训练step前奖励系数没更新的bug
6. 关闭在x,y速度小于0.2时变为0, 加入zero_commands的课程, 0->1500训练步, 逐渐从0.0->0.1
7. 降低训练的command采样范围`x,y:[-1,1]->[-0.5,0.5]`
8. resample command降低`10->5`
# 20251115
## v0.1.3
1. 修改`terrain_level`计算方法, 取当前全部环境等级取平均
2. 仍然不使用`heading_command`, 虽然能够更稳定的提升环境等级, 并可以采样到更多的角速度指令, 但是无法以一个恒定的角速度进行移动, 和实际操作中不符
3. 修复`torch.jit`导出问题
4. 分别记录每个地形的奖励
5. 加载训练模型时, 支持环境奖励系数课程加载
6. 参考[yusongmin1-My_unitree_go2_gym](https://github.com/yusongmin1/My_unitree_go2_gym)加入域随机化`randomize_link_mass, randomize_base_com, randomize_pd_gains, randomize_motor_zero_offset`, 降低base_mass的最大值`3->1`, 降低`push_robots`xy方向速度`1.0->0.4`, 加入角速度推力`0.6`, 增大摩擦最小值`0.1->0.2`
7. 修改范围`lin_vel_y: 0.5->1.0`, 修改命令采样时间`resampling_time: 30->10`
8. 修改`base_height`计算使用的点云范围, 长宽`0.7x0.5->0.4x0.3`
9. 修改`wave, stairs`训练配置, 其中都加入`0.1`的平地
10. 关闭对`correct_base_height, dof_power`奖励的课程降低
11. 修改obs中`height_measurements`系数`5 -> 2.5`
# 20251114
## v0.1.2
1. 解决CTS`rollout_storage_cts.py`中学生教授数据采样混乱的问题
# 20251113
## v0.1.1
1. 上下楼梯环境训练7h13min完成, 纯PPO无法学到上楼梯动作, 下楼梯基本能完成
2. 测试10k的wave, slope, rough_slope训练, 修复地形提升问题
3. play.py中加入onnx模型导出功能
4. 机身高度稍微有点低, 提升3cm, `base_height_target: 0.35 -> 0.38`
5. 加入CTS算法替代PPO:
    - 新文件: `on_policy_runner_cts.py, actor_critic_cts.py, cts.py, rollout_storage_cts.py`
    - 新配置: `LeggedRobotCfgCTS`
    - 新任务: `go2_cts`
    - 新导出: 修改torch.script和onnx导出代码, onnx模型的输入是按照IsaacLab的按照item堆叠的结果, 部署C++代码的帧堆叠[obsevation_manager.h](https://github.com/unitreerobotics/unitree_rl_lab/blob/61bfba15d35f1a93e3bacab85fe06b31643c83b7/deploy/include/isaaclab/manager/observation_manager.h#L63)
## v0.1
1. 添加地形选择`wave, slope, rough_slope, stairs down, stairs up, obstacles, stepping_stones, gap, flat`
2. 添加高度特征
3. 修改base_height计算方法, 通过高度特征平均值计算
4. 支持课程奖励系数`rewards.curriculum_rewards`
