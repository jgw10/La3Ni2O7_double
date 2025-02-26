import parameters as pam

# 将轨道, 自旋与数字一一对应, 用来生成每个态的数字uid
if pam.Norb == 5:
    orb_int = {'d3z2r2': 0,
               'dx2y2': 1,
               'px': 2,
               'py': 3,
               'apz': 4}
    int_orb = {value: key for key, value in orb_int.items()}
spin_int = {'up': 1, 'dn': 0}
int_spin = {value: key for key, value in spin_int.items()}

Ni_position = [(-1, 0, z) for z in range(2*pam.layer_num-1)] + [(1, 0, z) for z in range(2*pam.layer_num-1)]


def get_unit_cell_rep(x, y, z):
    """
    确定需要计算的晶格, 根据坐标确定轨道
    :return:orbs
    """
    # Ni, 层内O和层间O的位置
    Ni_xy = ((-1, 0), (1, 0))
    O1_xy = ((-2, 0), (0, 0), (2, 0))
    O2_xy = ((-1, 1), (-1, -1), (1, 1), (1, -1))
    Oap_xy = ((-1, 0), (1, 0))

    # 确定轨道
    if z == 0 or z == 2:
        if (x, y) in Ni_xy:
            return pam.Ni_orbs
        elif (x, y) in O1_xy:
            return pam.O1_orbs
        elif (x, y) in O2_xy:
            return pam.O2_orbs
        else:
            return ['NotOnSublattice']
    elif z == 1:
        if (x, y) in Oap_xy:
            return pam.Oap_orbs
        else:
            return ['NotOnSublattice']
    else:
        return ['NotOnSublattice']
