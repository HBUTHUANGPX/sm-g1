# G1 evaluation

Checkpoint SHA256: `c4d4c8188f91c4683464e64bf7ccb5c4c99047386f906950e5e7e4253705cf51`
Source revision: `47055435fbef35d4ca564509cd218fe447abcc14`
Execution: `grouped`
Kinematics execution: `batched`
Scheduling: `parallel`
Evaluation wall seconds: 279.4850000000001

Full-clean coverage is not tracking success. Strict completion uses native tracking termination.

## strict

Profile elapsed seconds: 277.6535527000001; log: `evaluation/strict/evaluation.log`

6/40 completed; mean normalized return: 0.330480.

| Clip | Steps | Covered to end | Strict completed | Normalized return | Root XY RMSE | Joint RMSE |
|---|---:|:---:|:---:|---:|---:|---:|
| dance1_subject1 | 4391 | False | False | 0.493227 | 2.545051 | 0.173694 |
| dance1_subject2 | 1608 | False | False | 0.181416 | 0.453887 | 0.205805 |
| dance1_subject3 | 6573 | True | True | 0.788367 | 1.370525 | 0.187482 |
| dance2_subject1 | 6 | False | False | 0.000181 | 0.002676 | 0.128278 |
| dance2_subject2 | 477 | False | False | 0.038254 | 0.042109 | 0.236019 |
| dance2_subject3 | 6899 | False | False | 0.488212 | 0.881326 | 0.190789 |
| dance2_subject4 | 2956 | False | False | 0.189882 | 1.206054 | 0.191507 |
| dance2_subject5 | 10223 | False | False | 0.688680 | 5.091011 | 0.150601 |
| fallAndGetUp1_subject1 | 7 | False | False | 0.000250 | 0.004090 | 0.142392 |
| fallAndGetUp1_subject4 | 6 | False | False | 0.000161 | 0.007088 | 0.144841 |
| fallAndGetUp1_subject5 | 441 | False | False | 0.042449 | 0.153103 | 0.215047 |
| fallAndGetUp2_subject2 | 686 | False | False | 0.062188 | 0.210557 | 0.333610 |
| fallAndGetUp2_subject3 | 700 | False | False | 0.068010 | 0.109275 | 0.303401 |
| fallAndGetUp3_subject1 | 887 | False | False | 0.135673 | 0.237027 | 0.207947 |
| fight1_subject2 | 341 | False | False | 0.023037 | 0.228478 | 0.167308 |
| fight1_subject3 | 6711 | False | False | 0.409945 | 1.062643 | 0.178451 |
| fight1_subject5 | 6 | False | False | 0.000080 | 0.008002 | 0.137220 |
| fightAndSports1_subject1 | 6 | False | False | 0.000161 | 0.002202 | 0.125881 |
| fightAndSports1_subject4 | 7 | False | False | 0.000144 | 0.008091 | 0.152681 |
| jumps1_subject1 | 6 | False | False | 0.000151 | 0.004224 | 0.138078 |
| jumps1_subject2 | 589 | False | False | 0.037737 | 1.313887 | 0.185270 |
| jumps1_subject5 | 2716 | False | False | 0.149756 | 5.858282 | 0.183129 |
| run1_subject2 | 8821 | False | False | 0.585391 | 1.681973 | 0.119681 |
| run1_subject5 | 5574 | False | False | 0.337641 | 1.881348 | 0.173388 |
| run2_subject1 | 4898 | False | False | 0.287159 | 1.496153 | 0.165577 |
| run2_subject4 | 10456 | False | False | 0.623985 | 1.417427 | 0.190491 |
| sprint1_subject2 | 3823 | False | False | 0.224830 | 0.626265 | 0.137064 |
| sprint1_subject4 | 13244 | False | False | 0.722877 | 3.123121 | 0.167020 |
| walk1_subject1 | 11733 | False | False | 0.750710 | 0.753273 | 0.146583 |
| walk1_subject2 | 8468 | False | False | 0.536869 | 1.488564 | 0.132779 |
| walk1_subject5 | 13065 | True | True | 0.816441 | 2.520991 | 0.128025 |
| walk2_subject1 | 11908 | True | True | 0.772027 | 1.290274 | 0.182039 |
| walk2_subject3 | 9152 | False | False | 0.620643 | 1.122960 | 0.164899 |
| walk2_subject4 | 7 | False | False | 0.000153 | 0.007954 | 0.152661 |
| walk3_subject1 | 10630 | False | False | 0.672464 | 1.394938 | 0.201622 |
| walk3_subject2 | 12330 | True | True | 0.829689 | 0.901006 | 0.142152 |
| walk3_subject3 | 7 | False | False | 0.000161 | 0.007046 | 0.156896 |
| walk3_subject4 | 7 | False | False | 0.000148 | 0.006446 | 0.144181 |
| walk3_subject5 | 12330 | True | True | 0.802392 | 4.898497 | 0.145630 |
| walk4_subject1 | 8195 | True | True | 0.837651 | 0.994774 | 0.165270 |

## full_clean

Profile elapsed seconds: 279.2763395000002; log: `evaluation/full_clean/evaluation.log`

40/40 covered; tracking success not measured; mean normalized return: 0.701851.

| Clip | Steps | Covered to end | Strict completed | Normalized return | Root XY RMSE | Joint RMSE |
|---|---:|:---:|:---:|---:|---:|---:|
| dance1_subject1 | 6573 | True | not measured | 0.606673 | 2.991504 | 0.264912 |
| dance1_subject2 | 6573 | True | not measured | 0.643665 | 1.848937 | 0.220970 |
| dance1_subject3 | 6573 | True | not measured | 0.787198 | 1.311657 | 0.188224 |
| dance2_subject1 | 11283 | True | not measured | 0.668078 | 6.022658 | 0.197065 |
| dance2_subject2 | 11283 | True | not measured | 0.684460 | 1.879977 | 0.214176 |
| dance2_subject3 | 11283 | True | not measured | 0.773568 | 1.375261 | 0.193091 |
| dance2_subject4 | 11283 | True | not measured | 0.707208 | 4.147499 | 0.201800 |
| dance2_subject5 | 11283 | True | not measured | 0.761075 | 3.829602 | 0.152754 |
| fallAndGetUp1_subject1 | 8410 | True | not measured | 0.412722 | 3.177032 | 0.398381 |
| fallAndGetUp1_subject4 | 8410 | True | not measured | 0.582158 | 4.173288 | 0.343153 |
| fallAndGetUp1_subject5 | 8410 | True | not measured | 0.606379 | 4.048999 | 0.281972 |
| fallAndGetUp2_subject2 | 8195 | True | not measured | 0.429363 | 4.051347 | 0.415677 |
| fallAndGetUp2_subject3 | 8195 | True | not measured | 0.512068 | 2.261910 | 0.397231 |
| fallAndGetUp3_subject1 | 5108 | True | not measured | 0.560321 | 8.301795 | 0.338076 |
| fight1_subject2 | 12243 | True | not measured | 0.669947 | 1.769606 | 0.206838 |
| fight1_subject3 | 12243 | True | not measured | 0.719243 | 3.766150 | 0.193309 |
| fight1_subject5 | 12243 | True | not measured | 0.770564 | 2.738417 | 0.162309 |
| fightAndSports1_subject1 | 12256 | True | not measured | 0.707707 | 7.424154 | 0.186782 |
| fightAndSports1_subject4 | 12256 | True | not measured | 0.711910 | 4.659700 | 0.209925 |
| jumps1_subject1 | 12221 | True | not measured | 0.747603 | 1.979257 | 0.157113 |
| jumps1_subject2 | 12221 | True | not measured | 0.701412 | 5.896037 | 0.250796 |
| jumps1_subject5 | 12221 | True | not measured | 0.274612 | 18.635837 | 0.486007 |
| run1_subject2 | 11890 | True | not measured | 0.774899 | 1.672383 | 0.126278 |
| run1_subject5 | 11890 | True | not measured | 0.733486 | 2.282001 | 0.163673 |
| run2_subject1 | 12240 | True | not measured | 0.730585 | 1.658834 | 0.159867 |
| run2_subject4 | 12240 | True | not measured | 0.722703 | 1.269214 | 0.186604 |
| sprint1_subject2 | 13655 | True | not measured | 0.779821 | 1.664162 | 0.141222 |
| sprint1_subject4 | 13655 | True | not measured | 0.736679 | 2.465920 | 0.169237 |
| walk1_subject1 | 13065 | True | not measured | 0.826330 | 1.522328 | 0.166940 |
| walk1_subject2 | 13065 | True | not measured | 0.826964 | 1.011761 | 0.133031 |
| walk1_subject5 | 13065 | True | not measured | 0.812043 | 2.339828 | 0.128372 |
| walk2_subject1 | 11908 | True | not measured | 0.771089 | 1.113460 | 0.182106 |
| walk2_subject3 | 11908 | True | not measured | 0.786778 | 1.669696 | 0.175965 |
| walk2_subject4 | 11908 | True | not measured | 0.813818 | 1.012269 | 0.150157 |
| walk3_subject1 | 12330 | True | not measured | 0.765225 | 1.839798 | 0.212481 |
| walk3_subject2 | 12330 | True | not measured | 0.835069 | 0.871949 | 0.141371 |
| walk3_subject3 | 12330 | True | not measured | 0.695049 | 4.008122 | 0.286187 |
| walk3_subject4 | 12330 | True | not measured | 0.790387 | 1.621220 | 0.209815 |
| walk3_subject5 | 12330 | True | not measured | 0.801008 | 3.549627 | 0.146929 |
| walk4_subject1 | 8195 | True | not measured | 0.834185 | 1.107959 | 0.165075 |
