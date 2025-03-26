import numpy as np

# hole_num: 空穴数目, layer_num: 层数, Norb: 轨道数目
hole_num = 5
layer_num = 1
energy_range = (0, 100)  # 限制能量范围, 单位 eV
# 是否进行全部空穴的耦合变换
if_coupled = 0
# 考虑所有Sz, 或者只考虑某个Sz
Sz_list = ['All_Sz']
# Sz_list = [0]
# 是否考虑在Ni上方的Oap
if_Oap = 0
# 改变Ni2O8(Ni2O10)的晶格, 使得tpp的值是正的
if_change_lattice = 1
# 选择单谈三重态的变换类型
if_basis_change_type = 'double'

Norb = 5
pressure_list = (0, 4, 8, 16, 29.5)

A_list = [5., 6., 7.]
A = 6.
B = 0.15
C = 0.58
Upp = 4.
Uoo = 4.
Upps = [4.0]
Uoos = [4.0]

ed_list = ({'d3z2r2': 0.046, 'dx2y2': 0.},
           {'d3z2r2': 0.054, 'dx2y2': 0.},
           {'d3z2r2': 0.060, 'dx2y2': 0.},
           {'d3z2r2': 0.072, 'dx2y2': 0.},
           {'d3z2r2': 0.095, 'dx2y2': 0.})
ep_list = (2.47, 2.56, 2.62, 2.75, 2.9)
eo_list = (2.94, 3.03, 3.02, 3.14, 3.24)

tpd_list = (1.38, 1.43, 1.46, 1.52, 1.58)
tpp_list = (0.537, 0.548, 0.554, 0.566, 0.562)
tdo_list = (1.48, 1.53, 1.55, 1.61, 1.66)
tpo_list = (0.445, 0.458, 0.468, 0.484, 0.487)
tz_a1a1 = 0.028
tz_b1b1 = 0.047

# if_tz_exist决定两层杂化的轨道
if_tz_exist = 2

Neval = 50
val_num = 1

if Norb == 5:
    Ni_orbs = ['d3z2r2', 'dx2y2']
    O1_orbs = ['px']
    O2_orbs = ['py']
    Oap_orbs = ['apz']
elif Norb == 4:
    Ni_orbs = ['d3z2r2', 'dx2y2']
    O1_orbs = ['px']
    O2_orbs = ['py']
    Oap_orbs = []
else:
    Ni_orbs = []
    O1_orbs = []
    O2_orbs = []
    Oap_orbs = []
O_orbs = O1_orbs + O2_orbs
O_orbs.sort()
O1_orbs.sort()
O2_orbs.sort()
Oap_orbs.sort()
