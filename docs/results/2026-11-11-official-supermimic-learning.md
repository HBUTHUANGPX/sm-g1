# Static learning audit, 2026-11-11

O = `official checkout / src/supermimic` (official HEAD5efbf114).
L = `local checkout / mimic_baseline/source/mimic_baseline/mimic_baseline/tasks/supermimic` (local HEAD83120bf).
Paths below relative to these explicit roots. No code changes, training, SSH or runtime-equivalence claims. Official means current checkout, not necessarily paper-era implementation.

## Effective execution

L `cli/train.py:116-120,179-205,367-373`: original Trainer; G1; perception false, recovery true, delayed_pd/max_delay4 by default. `runtime/configuration.py:4-22` preserves checkpoint settings on resume; `:35-46` builds29-action models. Legacy ACT_DIM27/OBS_DIM91 do NOT describe G1 sizes. New CLI default env64/iterations10 is not production. `learning/config.py:27-44,85-98`: replay1000,UTD.4,H32,goal10,history10,WMwindow8,10×2048 WM updates,actor4096,LR2e-4,cosine250000 outer iterations to.03 base,gamma.99,lambda.95,action noise.1,critic5×4,warmup64. Repo `docs/results/2026-11-09-g1-training-runtime.md:31` documents2560 env production-shaped profiling with those shapes and log10; checkpoint config not inspected here.

O `runners/supermimic_runner.py:48-94`: gencoord,actor_obs=rep_state,history1,buffer1000,WM2048×10×window8,actorH32/B4096,replay_ratio16,max_iterations1000000. `learners/policy_learner.py:23-41` gamma.99/lambda.95. Official can override proprio/history. Comparing its default privileged actor to local proprio deployment without labeling the distinction is misleading.

## 1. WM representation is different (algorithm divergence)

O `representations.py:469-489,505-554`: for29 joints fullstate71,rep68 (z,gravity,root localv/w,q,qd), predicts35 channels (dv3,dw3,dqd29). Pose integrates velocity with substep coefficient `(decimation+1)/(2*decimation)` (`:86-88`); orientation integrates angular velocity; gravity/height derived.
L `core/mdp.py:18-24`; `learning/models.py:165-177,258-274,307-323`: invariant71,input162=71+29+29+33,output73. Independent pose/rot6d/velocity/joint residuals, p'=p+dt*v'+Rheading*dp,q'=q+dt*qd'+dq,R'=R*project(identity6d+dr6). Rotation is not constrained to angular velocity. Previous-action and33 randomized parameters are useful delayed-actuator adaptation. O input rep+currentaction+simcontext and scaled joint targets (`models.py:73-96`), not the same input contract.
Risk: local greater freedom can imagine kinematically inconsistent transitions. This is an exploitation risk, not proof of failure or official superiority.

## 2. WM objective/horizon differs (algorithm divergence)

O `learners/wm_learner.py:23-40,59-63,142-188`: fixed Bryson scales rootpos.01,rootori.02,q.1,qd3.14; weighted mean across scored channels/time/batch. Gencoord blocks rootpose,q,qd only (`representations.py:147-154,569-578`), no direct rootv/w. Eight stored states/actions yield seven scored transitions (`learners/wm_learner.py:150-166`).
L `learning/world_model_update.py:210-264,374-412`: raw SSE across73 features and8 transitions, divide byB only; `models.py:326-355` scores position,6Drot,rootv/w,q,qd. Output standard deviations affect prediction parameterization and diagnostics, NOT optimized weighting. Both autoregressive and retain full interstep gradient. Local comment about reference inverse-variance loss is stale relative to current official fixed Bryson weighting. Raw rad/s errors may dominate; normalization alone does not dimensionlessly weight this loss.

## 3. Optimizer/MLP details materially differ (algorithm divergence)

O `learners/learner_utils.py:13-36`: **Muon on inner Linear weight matrices**, fused capturable Adam on first/last weights,biases,LN; LR2e-4; actor Adam beta2.95, others.999 (`learners/policy_learner.py:23-33,85-87`; `learners/wm_learner.py:56-57`). No cosine scheduler found in inspected runner/learner execution. `models.py:31-43,76-80,150-160`; `runners/supermimic_runner.py:517-524`: all hidden layers Linear/LN/GELU including critic.
L `learning/trainer.py:212-223,430-446`: pure Adam, same beta2 split, cosine250000/floor3%. `models.py:167-177,465-484,585-597`: WM SiLU+LN with zero final layer; policy/critic ELU withoutLN; decoder final weights×.01,bias0. Hidden widths agree WM2048×3,policyencoder/decoder2048/1024/512/256,critic256/128,latent32. These are not dimension-only adaptations.

## 4. Goal information/window differs (algorithm divergence)

O `representations.py:95-111,157-190`: ten targets stride5, startingcurrent,~.9s lookahead. Actor goal q/qd+robot-relative futureorientation=640; critic goal includes30 relativepositions. State-dependent goals let gradients flow from imagined state into conditioning.
L `data/reference.py:62-83`; `data/reference.py:151-157`; `runtime/configuration.py:42-46`: ten consecutive NEXT frames (~.2s),73/frame=730, reference-local dp/rot/rootv/w/q/.1qd. Pure reference goal has no predicted-state gradient, but separate current relativeorientation6 enters decoder (`learning/policy_update.py:213-223`; `models.py:527-537`). Local decoder968=32+93×10+6,critic915=71+113+730+1. O default decoder100=32+68; optional proprio/history10 decoder962, still different goals/feedback. Root independently CPU-instantiated local dimensions WM162→73,encoder730/decoder968→29,critic915. Official numeric dimensions are source arithmetic, not runtime instantiation.

## 5. BPTT skeleton equivalent, critic details different

O `learners/policy_learner.py:93-142,180-226` and L `learning/policy_update.py:137-170,213-260,342-395,481-528`: freeze WM/critic PARAMETERS for actor update, preserve gradients through dynamics/reward/softsurvival/terminalV to actor. Objective sum gamma^t*survival_before_t*r_t + gamma^H*survival*Vtail. No real-simulator gradients. Critic targets detached; lambda-return recursion algebraically GAE-compatible. Normclip1 then actor elementclip.1 (`O learners/learner_utils.py:39-57`; `L trainer.py:411-426`, `learning/policy_update.py:355-362`).
O critic plain MSE,20 updates,same permutation reused across5epochs (`learners/policy_learner.py:206-226`). L **H×MSE** (32×),fresh shuffle each epoch,importance weighting,validity masking (`learning/policy_update.py:510-544`). O raw V goes intoGAE then target clipped; L clamps V before recursion (`:481-528`). Both upper-bound actor tailV at maxreward/(1-gamma). H scaling may change gradient clipping and Adam epsilon effects; not safe to assume cancellation. L critic running input normalization vs O raw MLP (`runners/supermimic_runner.py:517-524`).

## 6. Update/data ratio and iterations differ (algorithm divergence)

O `runners/supermimic_runner.py:663-698`: one outeriteration=one vectorenvstep; update frequency N/(10*2048*8)*16. N4096→.4 groups/vectorstep; N2560→**.25** (one group/4steps after startup).
L `trainer.py:684-703`: one outeriteration=10WM+1actor+20critic updates,collect alternating2/3 vectorsteps at UTD.4. At2560 remains**.4**,1.6×O. Nominal WM frames/newframes25.6 vs16; official actually scores7/8 nominal samples. Checkpoint iteration is not directly comparable. Local warmup64; official only horizon-availability gate.

## 7. Replay/curriculum differ, terminal endpoints matter

O `replay_buffer.py:100-121`: sample complete8-state windows first7done-free; eligibleenv uniformly,thenvalidstart uniformly. Seven scored transitions exclude terminal transition.
L `learning/replay.py:811-905`:8 stored(s,a,s1) transitions,terminal transition valid,mask following steps. Defensible recovery adaptation, learns failure endpoints/short episodes.
O reset ReturnSampler (`runners/supermimic_runner.py:85-94,631-644`;`runners/reset_sampling.py:17-34`) gamma.99,horizon20,memory256,steepness25,explore.5,keep.1; replay stays uniform.
L active manager explicitly MasterySampler (`envs/mdp/commands.py:95-99`,`manager_based.py:356-357`); actor usesp,WM.8p+.2uniform (`data/sampling.py:110-139,199-225`); replayimportance weighting (`learning/policy_update.py:98-114,352-355`;`learning/world_model_update.py:356-384`). Affects resets AND learning distribution. Detailed statistical equivalence not established.

## 8. Clip ends and reset-history noise

O `envs/world_model_env.py:231-239` holds lastreferenceframe and continues imagination. L `learning/policy_update.py:316-349` terminates atclipexpiry and validity-masks reward. Different task boundary semantics, not inherently a defect.
Concrete local mismatch risk: O padding shares one noisy oldest observation (`envs/world_model_env.py:271-281`). L repeats early states (`learning/replay.py:1069-1075`) then independently noises full imagined history (`learning/policy_update.py:122-126`), whereas real collection repeats a single already-noisy reset observation (`learning/collection.py:27-47`). Imagined/real reset-padding noise correlation differs. Impact unmeasured; merits targeted test particularly recovery resets.

## 9. Reward/exploration core aligns, context still differs

O `utils/tracking_reward.py:69-96`;L `core/mdp.py:196-252`:6Gaussian terms,widths body.3/.4/1/3.14,root.3/.4,weights1×4+.5×2; actionrate-.1,jointlimit-2,dt scaling; body targets reanchored to robotxy/yaw. Exact bodysets/robustyaw are separate environment checks; matching formula≠numerical equivalence.
Both collection Gaussianactionnoise.1 (`O runners/supermimic_runner.py:614-615`;L `learning/collection.py:132-136`). L clamps±20 in real/imagination (`core/action_bounds.py:6-11`,`learning/policy_update.py:222`). O imagination passes raw actions (`envs/world_model_env.py:114-129`); do not assert clamp-equivalence without tracing realactionprocessing. Neither inspected actor-imagination path adds Gaussian action noise; proprio noise separate.

## Conclusion and evidence boundary

Local preserves learned-dynamics BPTT actor/critic design and many nominal sizes, but is not an algorithm-equivalent port of official HEAD5efbf114. Major differences explicit choices, not automatically bugs. Priority: investigate raw WM weighting/kinematic consistency,H-scaled critic under clipping,2560-env update/data ratio,reset-padding noise mismatch. No performance conclusion or blanket prescription to replace local with official follows from static inspection. Matched numerical/gradient tests and frozen-checkpoint rollouts would be needed for runtime verdict.


Pinned source roots: [official](https://github.com/HybridRobotics/supermimic/tree/5efbf114e11f725c31d83e217be987c13c0e4ce2), [local](https://github.com/HBUTHUANGPX/sm-g1/tree/83120bf0722e54dce47fc68a146439140d32d8c2). See [audit summary](2026-11-11-official-supermimic-audit.md) for cross-checks and limitations.
