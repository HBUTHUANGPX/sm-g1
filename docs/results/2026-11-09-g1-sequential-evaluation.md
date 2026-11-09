# G1 evaluation

Checkpoint SHA256: `c4d4c8188f91c4683464e64bf7ccb5c4c99047386f906950e5e7e4253705cf51`
Source revision: `47055435fbef35d4ca564509cd218fe447abcc14`
Execution: `grouped`
Kinematics execution: `batched`
Scheduling: `sequential`
Evaluation wall seconds: 298.5160000000001

Full-clean coverage is not tracking success. Strict completion uses native tracking termination.

## strict

Profile elapsed seconds: 150.34573460000024; log: `evaluation/strict/evaluation.log`

7/40 completed; mean normalized return: 0.335632.

| Clip | Steps | Covered to end | Strict completed | Normalized return | Root XY RMSE | Joint RMSE |
|---|---:|:---:|:---:|---:|---:|---:|
| dance1_subject1 | 4391 | False | False | 0.493295 | 2.497989 | 0.173792 |
| dance1_subject2 | 1608 | False | False | 0.181013 | 0.459104 | 0.205923 |
| dance1_subject3 | 6573 | True | True | 0.788277 | 1.374847 | 0.187517 |
| dance2_subject1 | 6 | False | False | 0.000181 | 0.002676 | 0.128278 |
| dance2_subject2 | 477 | False | False | 0.038254 | 0.042109 | 0.236019 |
| dance2_subject3 | 6899 | False | False | 0.488306 | 0.890915 | 0.190778 |
| dance2_subject4 | 2955 | False | False | 0.190375 | 1.205873 | 0.190966 |
| dance2_subject5 | 11283 | True | True | 0.760901 | 5.703182 | 0.152067 |
| fallAndGetUp1_subject1 | 7 | False | False | 0.000250 | 0.004090 | 0.142392 |
| fallAndGetUp1_subject4 | 6 | False | False | 0.000161 | 0.007088 | 0.144841 |
| fallAndGetUp1_subject5 | 441 | False | False | 0.042444 | 0.153341 | 0.214961 |
| fallAndGetUp2_subject2 | 683 | False | False | 0.059027 | 0.279427 | 0.332873 |
| fallAndGetUp2_subject3 | 700 | False | False | 0.068010 | 0.109280 | 0.303402 |
| fallAndGetUp3_subject1 | 887 | False | False | 0.135583 | 0.239272 | 0.207937 |
| fight1_subject2 | 341 | False | False | 0.023037 | 0.228507 | 0.167312 |
| fight1_subject3 | 6711 | False | False | 0.409994 | 1.089506 | 0.178412 |
| fight1_subject5 | 6 | False | False | 0.000080 | 0.008002 | 0.137220 |
| fightAndSports1_subject1 | 6 | False | False | 0.000161 | 0.002202 | 0.125881 |
| fightAndSports1_subject4 | 7 | False | False | 0.000144 | 0.008091 | 0.152681 |
| jumps1_subject1 | 6 | False | False | 0.000151 | 0.004224 | 0.138078 |
| jumps1_subject2 | 589 | False | False | 0.037737 | 1.313968 | 0.185265 |
| jumps1_subject5 | 2002 | False | False | 0.114323 | 2.115928 | 0.179849 |
| run1_subject2 | 8821 | False | False | 0.585556 | 1.609930 | 0.119679 |
| run1_subject5 | 8135 | False | False | 0.501807 | 3.186624 | 0.166594 |
| run2_subject1 | 4898 | False | False | 0.287704 | 1.442346 | 0.165393 |
| run2_subject4 | 10454 | False | False | 0.624140 | 1.451738 | 0.190489 |
| sprint1_subject2 | 3824 | False | False | 0.225031 | 0.615167 | 0.137242 |
| sprint1_subject4 | 13244 | False | False | 0.722211 | 3.315844 | 0.167030 |
| walk1_subject1 | 11661 | False | False | 0.749860 | 0.727804 | 0.145612 |
| walk1_subject2 | 8468 | False | False | 0.538202 | 1.451117 | 0.132970 |
| walk1_subject5 | 13065 | True | True | 0.816687 | 2.565738 | 0.127944 |
| walk2_subject1 | 11908 | True | True | 0.772220 | 1.338955 | 0.181955 |
| walk2_subject3 | 9142 | False | False | 0.618697 | 1.114413 | 0.164571 |
| walk2_subject4 | 7 | False | False | 0.000153 | 0.007954 | 0.152661 |
| walk3_subject1 | 10630 | False | False | 0.678406 | 1.205192 | 0.201925 |
| walk3_subject2 | 12330 | True | True | 0.833170 | 0.835202 | 0.142128 |
| walk3_subject3 | 7 | False | False | 0.000161 | 0.007046 | 0.156896 |
| walk3_subject4 | 7 | False | False | 0.000148 | 0.006446 | 0.144181 |
| walk3_subject5 | 12330 | True | True | 0.802480 | 5.006054 | 0.145730 |
| walk4_subject1 | 8195 | True | True | 0.836951 | 0.994323 | 0.165295 |

## full_clean

Profile elapsed seconds: 147.9527478; log: `evaluation/full_clean/evaluation.log`

40/40 covered; tracking success not measured; mean normalized return: 0.696768.

| Clip | Steps | Covered to end | Strict completed | Normalized return | Root XY RMSE | Joint RMSE |
|---|---:|:---:|:---:|---:|---:|---:|
| dance1_subject1 | 6573 | True | not measured | 0.673059 | 3.226488 | 0.211548 |
| dance1_subject2 | 6573 | True | not measured | 0.644993 | 2.849494 | 0.220681 |
| dance1_subject3 | 6573 | True | not measured | 0.787749 | 1.208402 | 0.188270 |
| dance2_subject1 | 11283 | True | not measured | 0.669397 | 6.374654 | 0.196689 |
| dance2_subject2 | 11283 | True | not measured | 0.683095 | 1.579040 | 0.214149 |
| dance2_subject3 | 11283 | True | not measured | 0.773406 | 1.389325 | 0.193047 |
| dance2_subject4 | 11283 | True | not measured | 0.707930 | 3.860149 | 0.201638 |
| dance2_subject5 | 11283 | True | not measured | 0.760688 | 4.125271 | 0.152862 |
| fallAndGetUp1_subject1 | 8410 | True | not measured | 0.441940 | 2.222176 | 0.370598 |
| fallAndGetUp1_subject4 | 8410 | True | not measured | 0.572001 | 2.833716 | 0.348709 |
| fallAndGetUp1_subject5 | 8410 | True | not measured | 0.604749 | 3.689491 | 0.282131 |
| fallAndGetUp2_subject2 | 8195 | True | not measured | 0.405080 | 4.744402 | 0.433281 |
| fallAndGetUp2_subject3 | 8195 | True | not measured | 0.514270 | 1.681791 | 0.406786 |
| fallAndGetUp3_subject1 | 5108 | True | not measured | 0.619831 | 7.394843 | 0.272402 |
| fight1_subject2 | 12243 | True | not measured | 0.670494 | 1.933238 | 0.206823 |
| fight1_subject3 | 12243 | True | not measured | 0.633457 | 3.193887 | 0.275508 |
| fight1_subject5 | 12243 | True | not measured | 0.771147 | 2.738962 | 0.162227 |
| fightAndSports1_subject1 | 12256 | True | not measured | 0.699922 | 8.805163 | 0.186803 |
| fightAndSports1_subject4 | 12256 | True | not measured | 0.712506 | 4.580751 | 0.209857 |
| jumps1_subject1 | 12221 | True | not measured | 0.747676 | 1.745458 | 0.157270 |
| jumps1_subject2 | 12221 | True | not measured | 0.703450 | 4.415146 | 0.252006 |
| jumps1_subject5 | 12221 | True | not measured | 0.281298 | 16.795976 | 0.468822 |
| run1_subject2 | 11890 | True | not measured | 0.769039 | 1.779595 | 0.127901 |
| run1_subject5 | 11890 | True | not measured | 0.733378 | 2.222981 | 0.163696 |
| run2_subject1 | 12240 | True | not measured | 0.497102 | 2.178121 | 0.317906 |
| run2_subject4 | 12240 | True | not measured | 0.724296 | 1.162430 | 0.186587 |
| sprint1_subject2 | 13655 | True | not measured | 0.779700 | 1.733414 | 0.141211 |
| sprint1_subject4 | 13655 | True | not measured | 0.723199 | 2.766142 | 0.187157 |
| walk1_subject1 | 13065 | True | not measured | 0.828056 | 1.240952 | 0.166472 |
| walk1_subject2 | 13065 | True | not measured | 0.827912 | 0.981554 | 0.133053 |
| walk1_subject5 | 13065 | True | not measured | 0.811780 | 2.266570 | 0.128452 |
| walk2_subject1 | 11908 | True | not measured | 0.772155 | 1.085123 | 0.182123 |
| walk2_subject3 | 11908 | True | not measured | 0.787741 | 1.697220 | 0.176607 |
| walk2_subject4 | 11908 | True | not measured | 0.815412 | 1.011281 | 0.150162 |
| walk3_subject1 | 12330 | True | not measured | 0.765747 | 1.656781 | 0.212173 |
| walk3_subject2 | 12330 | True | not measured | 0.833283 | 0.893549 | 0.141351 |
| walk3_subject3 | 12330 | True | not measured | 0.699795 | 1.816478 | 0.286126 |
| walk3_subject4 | 12330 | True | not measured | 0.787888 | 1.661631 | 0.210140 |
| walk3_subject5 | 12330 | True | not measured | 0.801797 | 3.299554 | 0.146851 |
| walk4_subject1 | 8195 | True | not measured | 0.834305 | 1.154603 | 0.165083 |
