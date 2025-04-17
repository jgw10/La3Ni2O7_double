import numpy as np

# hole_num: 空穴数目, layer_num: 层数, Norb: 轨道数目
hole_num = 5
layer_num = 1
max_energy = 100.
# Sz = 'All_Sz'时, 考虑的所有自旋的情况
Sz_list = ['All_Sz']
# Sz_list = [0, 1, 2]
if_basis_change_type = 'double'
if_coupled = 0
Norb = 5
Mc = 2
pressure_list = (0, 4, 8, 16, 29.5)

A = 6.0
A_list = [5.0, 6.0, 7.0]
B = 0.15
C = 0.58
Upp = 4.0
Uoo = 4.0
# Upps = [4.0]
# Uoos = [4.0]

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
Neval = 20
val_num = 1

if Norb == 5:
    Ni_orbs = ['d3z2r2', 'dx2y2']
    O1_orbs = ['px']
    O2_orbs = ['py']
    Oap_orbs = ['apz']
else:
    Ni_orbs = None
    O1_orbs = None
    O2_orbs = None
    Oap_orbs = None
O_orbs = O1_orbs + O2_orbs
O_orbs.sort()
O1_orbs.sort()
O2_orbs.sort()
Oap_orbs.sort()
