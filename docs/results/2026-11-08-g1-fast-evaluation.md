# G1 evaluation

Checkpoint SHA256: `c4d4c8188f91c4683464e64bf7ccb5c4c99047386f906950e5e7e4253705cf51`
Source revision: `1c50bb87c19f2ffcffa1d737ef1a962670f345e9`
Execution: `grouped`

Full-clean coverage is not tracking success. Strict completion uses native tracking termination.

## strict

7/40 completed; mean normalized return: 0.342636.

| Clip | Steps | Covered to end | Strict completed | Normalized return | Root XY RMSE | Joint RMSE |
|---|---:|:---:|:---:|---:|---:|---:|
| dance1_subject1 | 4391 | False | False | 0.492093 | 2.572568 | 0.173629 |
| dance1_subject2 | 1608 | False | False | 0.180968 | 0.465652 | 0.206020 |
| dance1_subject3 | 6573 | True | True | 0.788458 | 1.330517 | 0.187577 |
| dance2_subject1 | 6 | False | False | 0.000181 | 0.002676 | 0.128278 |
| dance2_subject2 | 477 | False | False | 0.038254 | 0.042109 | 0.236019 |
| dance2_subject3 | 6899 | False | False | 0.488192 | 0.870622 | 0.190876 |
| dance2_subject4 | 2955 | False | False | 0.190472 | 1.193097 | 0.191016 |
| dance2_subject5 | 10215 | False | False | 0.690122 | 4.883869 | 0.150412 |
| fallAndGetUp1_subject1 | 7 | False | False | 0.000250 | 0.004090 | 0.142392 |
| fallAndGetUp1_subject4 | 6 | False | False | 0.000161 | 0.007088 | 0.144841 |
| fallAndGetUp1_subject5 | 441 | False | False | 0.042444 | 0.153340 | 0.214961 |
| fallAndGetUp2_subject2 | 683 | False | False | 0.059019 | 0.279594 | 0.332907 |
| fallAndGetUp2_subject3 | 700 | False | False | 0.067999 | 0.109309 | 0.303383 |
| fallAndGetUp3_subject1 | 887 | False | False | 0.135784 | 0.236528 | 0.207595 |
| fight1_subject2 | 341 | False | False | 0.023037 | 0.228511 | 0.167309 |
| fight1_subject3 | 6711 | False | False | 0.410568 | 1.055633 | 0.178281 |
| fight1_subject5 | 6 | False | False | 0.000080 | 0.008002 | 0.137220 |
| fightAndSports1_subject1 | 6 | False | False | 0.000161 | 0.002202 | 0.125881 |
| fightAndSports1_subject4 | 7 | False | False | 0.000144 | 0.008091 | 0.152681 |
| jumps1_subject1 | 6 | False | False | 0.000151 | 0.004224 | 0.138078 |
| jumps1_subject2 | 589 | False | False | 0.037737 | 1.313896 | 0.185269 |
| jumps1_subject5 | 2047 | False | False | 0.115033 | 2.267275 | 0.181096 |
| run1_subject2 | 8821 | False | False | 0.585447 | 1.671648 | 0.119668 |
| run1_subject5 | 11890 | True | True | 0.729510 | 4.393231 | 0.165221 |
| run2_subject1 | 10913 | False | False | 0.645040 | 1.403350 | 0.163808 |
| run2_subject4 | 10454 | False | False | 0.623721 | 1.728048 | 0.190538 |
| sprint1_subject2 | 3824 | False | False | 0.224448 | 0.696216 | 0.137297 |
| sprint1_subject4 | 11245 | False | False | 0.611228 | 2.987270 | 0.163480 |
| walk1_subject1 | 11662 | False | False | 0.747608 | 0.748698 | 0.145679 |
| walk1_subject2 | 8468 | False | False | 0.538209 | 1.460945 | 0.132948 |
| walk1_subject5 | 13065 | True | True | 0.816477 | 2.522153 | 0.128010 |
| walk2_subject1 | 11908 | True | True | 0.772094 | 1.265153 | 0.181946 |
| walk2_subject3 | 7209 | False | False | 0.503506 | 0.699719 | 0.160309 |
| walk2_subject4 | 7 | False | False | 0.000153 | 0.007954 | 0.152661 |
| walk3_subject1 | 10629 | False | False | 0.675375 | 1.258406 | 0.201614 |
| walk3_subject2 | 12330 | True | True | 0.830783 | 0.856339 | 0.142102 |
| walk3_subject3 | 7 | False | False | 0.000161 | 0.007046 | 0.156896 |
| walk3_subject4 | 7 | False | False | 0.000148 | 0.006446 | 0.144181 |
| walk3_subject5 | 12330 | True | True | 0.802171 | 5.178261 | 0.145741 |
| walk4_subject1 | 8195 | True | True | 0.838064 | 0.981592 | 0.165304 |

## full_clean

40/40 covered; tracking success not measured; mean normalized return: 0.674870.

| Clip | Steps | Covered to end | Strict completed | Normalized return | Root XY RMSE | Joint RMSE |
|---|---:|:---:|:---:|---:|---:|---:|
| dance1_subject1 | 6573 | True | not measured | 0.606330 | 2.828693 | 0.264734 |
| dance1_subject2 | 6573 | True | not measured | 0.646733 | 2.340349 | 0.220443 |
| dance1_subject3 | 6573 | True | not measured | 0.787482 | 1.346267 | 0.188242 |
| dance2_subject1 | 11283 | True | not measured | 0.669390 | 5.238727 | 0.196825 |
| dance2_subject2 | 11283 | True | not measured | 0.685800 | 1.702373 | 0.214369 |
| dance2_subject3 | 11283 | True | not measured | 0.773469 | 1.374266 | 0.192979 |
| dance2_subject4 | 11283 | True | not measured | 0.708358 | 3.434893 | 0.201608 |
| dance2_subject5 | 11283 | True | not measured | 0.759531 | 3.918882 | 0.152846 |
| fallAndGetUp1_subject1 | 8410 | True | not measured | 0.384643 | 5.627417 | 0.429966 |
| fallAndGetUp1_subject4 | 8410 | True | not measured | 0.584659 | 3.104814 | 0.342172 |
| fallAndGetUp1_subject5 | 8410 | True | not measured | 0.592102 | 3.422410 | 0.283864 |
| fallAndGetUp2_subject2 | 8195 | True | not measured | 0.403863 | 3.992871 | 0.420605 |
| fallAndGetUp2_subject3 | 8195 | True | not measured | 0.507871 | 4.422451 | 0.394461 |
| fallAndGetUp3_subject1 | 5108 | True | not measured | 0.402360 | 7.036072 | 0.438566 |
| fight1_subject2 | 12243 | True | not measured | 0.266386 | 3.798476 | 0.453290 |
| fight1_subject3 | 12243 | True | not measured | 0.716232 | 3.876120 | 0.194057 |
| fight1_subject5 | 12243 | True | not measured | 0.771016 | 2.713737 | 0.162289 |
| fightAndSports1_subject1 | 12256 | True | not measured | 0.701616 | 9.012478 | 0.186785 |
| fightAndSports1_subject4 | 12256 | True | not measured | 0.712659 | 4.369505 | 0.209828 |
| jumps1_subject1 | 12221 | True | not measured | 0.747414 | 1.941797 | 0.157164 |
| jumps1_subject2 | 12221 | True | not measured | 0.348081 | 7.863298 | 0.433656 |
| jumps1_subject5 | 12221 | True | not measured | 0.284658 | 15.184635 | 0.443407 |
| run1_subject2 | 11890 | True | not measured | 0.775465 | 1.658569 | 0.126411 |
| run1_subject5 | 11890 | True | not measured | 0.733420 | 2.242657 | 0.163711 |
| run2_subject1 | 12240 | True | not measured | 0.731183 | 1.722630 | 0.159918 |
| run2_subject4 | 12240 | True | not measured | 0.723878 | 1.338319 | 0.186442 |
| sprint1_subject2 | 13655 | True | not measured | 0.779601 | 1.760541 | 0.141303 |
| sprint1_subject4 | 13655 | True | not measured | 0.736360 | 2.569125 | 0.169246 |
| walk1_subject1 | 13065 | True | not measured | 0.829115 | 1.501360 | 0.166161 |
| walk1_subject2 | 13065 | True | not measured | 0.828784 | 0.986672 | 0.132985 |
| walk1_subject5 | 13065 | True | not measured | 0.812231 | 2.267913 | 0.128410 |
| walk2_subject1 | 11908 | True | not measured | 0.771506 | 1.079792 | 0.181948 |
| walk2_subject3 | 11908 | True | not measured | 0.673512 | 2.739406 | 0.296544 |
| walk2_subject4 | 11908 | True | not measured | 0.814283 | 1.073214 | 0.150101 |
| walk3_subject1 | 12330 | True | not measured | 0.766489 | 1.697372 | 0.212281 |
| walk3_subject2 | 12330 | True | not measured | 0.833338 | 0.886857 | 0.141395 |
| walk3_subject3 | 12330 | True | not measured | 0.701164 | 2.840343 | 0.284150 |
| walk3_subject4 | 12330 | True | not measured | 0.787905 | 1.808243 | 0.210100 |
| walk3_subject5 | 12330 | True | not measured | 0.801708 | 3.493689 | 0.146675 |
| walk4_subject1 | 8195 | True | not measured | 0.834214 | 1.141843 | 0.165103 |
