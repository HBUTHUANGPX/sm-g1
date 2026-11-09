# G1 evaluation

Checkpoint SHA256: `c4d4c8188f91c4683464e64bf7ccb5c4c99047386f906950e5e7e4253705cf51`
Source revision: `47055435fbef35d4ca564509cd218fe447abcc14`
Execution: `grouped`
Kinematics execution: `batched`
Scheduling: `parallel`
Evaluation wall seconds: 280.3130000000001

Full-clean coverage is not tracking success. Strict completion uses native tracking termination.

## strict

Profile elapsed seconds: 277.8332894; log: `evaluation/strict/evaluation.log`

7/40 completed; mean normalized return: 0.327054.

| Clip | Steps | Covered to end | Strict completed | Normalized return | Root XY RMSE | Joint RMSE |
|---|---:|:---:|:---:|---:|---:|---:|
| dance1_subject1 | 2682 | False | False | 0.312214 | 1.449845 | 0.174320 |
| dance1_subject2 | 1608 | False | False | 0.181013 | 0.459102 | 0.205923 |
| dance1_subject3 | 6573 | True | True | 0.788774 | 1.358051 | 0.187489 |
| dance2_subject1 | 6 | False | False | 0.000181 | 0.002676 | 0.128278 |
| dance2_subject2 | 477 | False | False | 0.038254 | 0.042109 | 0.236019 |
| dance2_subject3 | 6899 | False | False | 0.488210 | 0.871587 | 0.190855 |
| dance2_subject4 | 2955 | False | False | 0.190079 | 1.191393 | 0.191323 |
| dance2_subject5 | 11283 | True | True | 0.761547 | 5.533733 | 0.151939 |
| fallAndGetUp1_subject1 | 7 | False | False | 0.000250 | 0.004090 | 0.142392 |
| fallAndGetUp1_subject4 | 6 | False | False | 0.000161 | 0.007088 | 0.144841 |
| fallAndGetUp1_subject5 | 441 | False | False | 0.042449 | 0.153103 | 0.215047 |
| fallAndGetUp2_subject2 | 687 | False | False | 0.062255 | 0.210848 | 0.333773 |
| fallAndGetUp2_subject3 | 700 | False | False | 0.067994 | 0.109364 | 0.303455 |
| fallAndGetUp3_subject1 | 887 | False | False | 0.135691 | 0.236930 | 0.207939 |
| fight1_subject2 | 341 | False | False | 0.023037 | 0.228484 | 0.167308 |
| fight1_subject3 | 6161 | False | False | 0.379172 | 0.909208 | 0.174620 |
| fight1_subject5 | 6 | False | False | 0.000080 | 0.008002 | 0.137220 |
| fightAndSports1_subject1 | 6 | False | False | 0.000161 | 0.002202 | 0.125881 |
| fightAndSports1_subject4 | 7 | False | False | 0.000144 | 0.008091 | 0.152681 |
| jumps1_subject1 | 6 | False | False | 0.000151 | 0.004224 | 0.138078 |
| jumps1_subject2 | 589 | False | False | 0.037737 | 1.313883 | 0.185270 |
| jumps1_subject5 | 2002 | False | False | 0.114944 | 1.917258 | 0.179411 |
| run1_subject2 | 8821 | False | False | 0.585605 | 1.639367 | 0.119667 |
| run1_subject5 | 5574 | False | False | 0.337652 | 1.894479 | 0.173398 |
| run2_subject1 | 4898 | False | False | 0.287389 | 1.483390 | 0.165536 |
| run2_subject4 | 10455 | False | False | 0.624328 | 1.462528 | 0.190446 |
| sprint1_subject2 | 3824 | False | False | 0.224934 | 0.618949 | 0.137129 |
| sprint1_subject4 | 13245 | False | False | 0.721937 | 3.278944 | 0.167128 |
| walk1_subject1 | 12255 | False | False | 0.776539 | 0.806388 | 0.155769 |
| walk1_subject2 | 8468 | False | False | 0.538028 | 1.434650 | 0.132811 |
| walk1_subject5 | 13065 | True | True | 0.816516 | 2.513950 | 0.127977 |
| walk2_subject1 | 11908 | True | True | 0.772333 | 1.201705 | 0.181918 |
| walk2_subject3 | 9145 | False | False | 0.619457 | 1.072590 | 0.164630 |
| walk2_subject4 | 7 | False | False | 0.000153 | 0.007954 | 0.152661 |
| walk3_subject1 | 10629 | False | False | 0.677784 | 1.246373 | 0.201430 |
| walk3_subject2 | 12330 | True | True | 0.833491 | 0.782575 | 0.142078 |
| walk3_subject3 | 7 | False | False | 0.000161 | 0.007046 | 0.156896 |
| walk3_subject4 | 7 | False | False | 0.000148 | 0.006446 | 0.144181 |
| walk3_subject5 | 12330 | True | True | 0.802760 | 4.958213 | 0.145641 |
| walk4_subject1 | 8195 | True | True | 0.838441 | 0.934696 | 0.165356 |

## full_clean

Profile elapsed seconds: 280.10394480000014; log: `evaluation/full_clean/evaluation.log`

40/40 covered; tracking success not measured; mean normalized return: 0.697699.

| Clip | Steps | Covered to end | Strict completed | Normalized return | Root XY RMSE | Joint RMSE |
|---|---:|:---:|:---:|---:|---:|---:|
| dance1_subject1 | 6573 | True | not measured | 0.606672 | 2.886650 | 0.264800 |
| dance1_subject2 | 6573 | True | not measured | 0.641731 | 3.952453 | 0.220988 |
| dance1_subject3 | 6573 | True | not measured | 0.787762 | 1.289061 | 0.188189 |
| dance2_subject1 | 11283 | True | not measured | 0.670397 | 5.119982 | 0.196530 |
| dance2_subject2 | 11283 | True | not measured | 0.686361 | 1.643339 | 0.214047 |
| dance2_subject3 | 11283 | True | not measured | 0.771452 | 1.428793 | 0.193163 |
| dance2_subject4 | 11283 | True | not measured | 0.709012 | 3.089356 | 0.201659 |
| dance2_subject5 | 11283 | True | not measured | 0.760483 | 3.296139 | 0.152748 |
| fallAndGetUp1_subject1 | 8410 | True | not measured | 0.405584 | 4.278082 | 0.415699 |
| fallAndGetUp1_subject4 | 8410 | True | not measured | 0.546474 | 6.739854 | 0.367122 |
| fallAndGetUp1_subject5 | 8410 | True | not measured | 0.402700 | 6.831386 | 0.421950 |
| fallAndGetUp2_subject2 | 8195 | True | not measured | 0.414910 | 4.556570 | 0.427443 |
| fallAndGetUp2_subject3 | 8195 | True | not measured | 0.526927 | 3.191451 | 0.390213 |
| fallAndGetUp3_subject1 | 5108 | True | not measured | 0.622067 | 7.410772 | 0.288550 |
| fight1_subject2 | 12243 | True | not measured | 0.668775 | 1.760283 | 0.209802 |
| fight1_subject3 | 12243 | True | not measured | 0.718035 | 3.426450 | 0.193092 |
| fight1_subject5 | 12243 | True | not measured | 0.771131 | 2.801987 | 0.162247 |
| fightAndSports1_subject1 | 12256 | True | not measured | 0.705850 | 7.405466 | 0.187105 |
| fightAndSports1_subject4 | 12256 | True | not measured | 0.713614 | 3.611563 | 0.209592 |
| jumps1_subject1 | 12221 | True | not measured | 0.747510 | 1.504975 | 0.157164 |
| jumps1_subject2 | 12221 | True | not measured | 0.695251 | 4.501800 | 0.258320 |
| jumps1_subject5 | 12221 | True | not measured | 0.281457 | 20.814726 | 0.474279 |
| run1_subject2 | 11890 | True | not measured | 0.774950 | 1.664141 | 0.126390 |
| run1_subject5 | 11890 | True | not measured | 0.733435 | 2.246363 | 0.163683 |
| run2_subject1 | 12240 | True | not measured | 0.730640 | 2.026974 | 0.159984 |
| run2_subject4 | 12240 | True | not measured | 0.724696 | 1.267016 | 0.186369 |
| sprint1_subject2 | 13655 | True | not measured | 0.779630 | 1.737640 | 0.141282 |
| sprint1_subject4 | 13655 | True | not measured | 0.737919 | 2.481159 | 0.168927 |
| walk1_subject1 | 13065 | True | not measured | 0.824923 | 1.575386 | 0.167202 |
| walk1_subject2 | 13065 | True | not measured | 0.826056 | 1.067608 | 0.132992 |
| walk1_subject5 | 13065 | True | not measured | 0.812069 | 2.346212 | 0.128377 |
| walk2_subject1 | 11908 | True | not measured | 0.770476 | 1.107884 | 0.182019 |
| walk2_subject3 | 11908 | True | not measured | 0.787035 | 1.729649 | 0.175935 |
| walk2_subject4 | 11908 | True | not measured | 0.813782 | 1.051827 | 0.150229 |
| walk3_subject1 | 12330 | True | not measured | 0.776411 | 1.740984 | 0.202856 |
| walk3_subject2 | 12330 | True | not measured | 0.832915 | 0.941226 | 0.141352 |
| walk3_subject3 | 12330 | True | not measured | 0.701506 | 1.515633 | 0.297273 |
| walk3_subject4 | 12330 | True | not measured | 0.791376 | 1.589228 | 0.209689 |
| walk3_subject5 | 12330 | True | not measured | 0.801770 | 3.435470 | 0.146677 |
| walk4_subject1 | 8195 | True | not measured | 0.834207 | 1.148915 | 0.165103 |
