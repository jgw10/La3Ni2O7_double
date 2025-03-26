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

Ni_position = [(-1, 0, 0), (1, 0, 0)]
Oap_position = []
for z in [0]:
    Oap_position.append((-1, 0, z))
    Oap_position.append((1, 0, z))


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
    if z == 0:
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


def get_state_type(state):
    """
    按照Ni, 层内O, 层间O每层的数量给态分类
    :param state: state = ((x1, y1, z1, orb1, s1), ...)
    :return:state_type
    """
    # 统计每一层Ni, 层内O, 层间O的数量
    Ni_num = {}
    O_num = {}
    Oap_num = {}
    for hole in state:
        x, _, _, orb, _ = hole
        # Ni
        if orb in pam.Ni_orbs:      # 这里尽量用轨道来判断, 方便以后修改lattice.py
            if x in Ni_num:
                Ni_num[x] += 1
            else:
                Ni_num[x] = 1
        # 层内O
        if orb in pam.O_orbs:
            if x < 0:
                x = -1
            if x > 0:
                x = 1
            if x in O_num:
                O_num[x] += 1
            else:
                O_num[x] = 1
        # 层间O
        if orb in pam.Oap_orbs:
            if x < 0:
                x = -1
            elif x > 0:
                x = 1
            if x in Oap_num:
                Oap_num[x] += 1
            else:
                Oap_num[x] = 1

    # 根据每一层Ni, 层内O和层间O的空穴数量, 生成态的类型
    state_type = {}
    # Ni
    for x, num in Ni_num.items():
        state_type[x] = f'd{10-num}'
    # 层内O
    for x, num in O_num.items():
        if num == 1:
            if x in state_type:
                state_type[x] += f'L'
            else:
                state_type[x] = f'L'
        else:
            if x in state_type:
                state_type[x] += f'L{num}'
            else:
                state_type[x] = f'L{num}'
    # 层外O
    for x, num in Oap_num.items():
        if num == 1:
            if x in state_type:
                state_type[x] += f'O'
            else:
                state_type[x] = f'O'
        else:
            if x in state_type:
                state_type[x] += f'O{num}'
            else:
                state_type[x] = f'O{num}'
    sorted_x = sorted(state_type.keys())
    state_type = [state_type[x] for x in sorted_x]
    state_type = '-'.join(state_type)

    return state_type


def get_orb_type(state):
    """
    将具体的state_type细化, 输出具体的d轨道
    :param state:
    :return:
    """
    simple_orbs = {'d3z2r2': 'dz2', 'dx2y2': 'dx2'} # 简化坐标表示
    Ni_orb = {}
    Oap_orb = {}    # 收集层间O的轨道数目
    L_orb = {}  # 收集层内O的轨道数目
    for hole in state:
        x, y, _, orb, _ = hole
        if orb in pam.Ni_orbs:
            if orb in simple_orbs:
                orb = simple_orbs[orb]
            if x in Ni_orb:
                Ni_orb[x] += [orb]
            else:
                Ni_orb[x] = [orb]
        elif orb in pam.Oap_orbs:
            if x in Oap_orb:
                Oap_orb[x] += 1
            else:
                Oap_orb[x] = 1
        else:
            if x < 0:
                x = -1
            elif x > 0:
                x = 1
            if x in L_orb:
                L_orb[x] += 1
            else:
                L_orb[x] = 1

    orb_type = {}
    for x, orbs in Ni_orb.items():
        orbs.sort()
        orb_type[x] = ''.join(orbs)
    for x, num in L_orb.items():
        if x in orb_type:
            if num == 1:
                orb_type[x] += f'L'
            else:
                orb_type[x] += f'L{num}'
        else:
            if num == 1:
                orb_type[x] = f'L'
            else:
                orb_type[x] = f'L{num}'
    for x, num in Oap_orb.items():
        if x in orb_type:
            if num == 1:
                orb_type[x] += f'apz'
            else:
                orb_type[x] += f'apz{num}'
        else:
            if num == 1:
                orb_type[x] = f'apz'
            else:
                orb_type[x] = f'apz{num}'
    sorted_x = sorted(orb_type.keys())
    orb_type = [orb_type[x] for x in sorted_x]
    orb_type = '_'.join(orb_type)

    return orb_type


def get_Ni_side_num(state):
    """
    得到和Ni同一边的空穴个数
    改动该函数, 对应basis_change.py中的
    create_singlet_triplet_basis_change_matrix也要一起修改
    :param state:
    :return:
    """
    side1_idx = []
    side2_idx = []
    for idx, hole in enumerate(state):
        if hole[0] < 0:
            side1_idx.append(idx)
        elif hole[0] > 0:
            side2_idx.append(idx)
    return side1_idx, side2_idx
