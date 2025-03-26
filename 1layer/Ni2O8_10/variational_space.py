import parameters as pam
import lattice as lat
import time
from itertools import permutations, product, combinations
from itertools import combinations_with_replacement as cwr
import bisect


def get_hole_uid(hole):
    """
    将空穴信息以进制的规则转化为数字信息
    :param hole: hole = ('x', 'y', 'z', 'orb', 's'), orb轨道, s自旋
    :return: hole_uid, 空穴所对应的数字
    """
    x, y, z, orb, s = hole

    # 依次将x, y, z, orb, s的最大个数求出来, 并作为进制
    b_x, b_y, b_z = lat.b_x, lat.b_y, lat.b_z
    delta_x, delta_y = lat.delta_x, lat.delta_y
    b_orb = pam.Norb
    b_s = 2

    # 将x, y, z, orb, s依次转为数字
    # 保证i_x, i_y, i_z一定是非负数
    i_x = x + delta_x
    i_y = y + delta_y
    i_z = z
    # 轨道和自旋转为数字
    i_orb = lat.orb_int[orb]
    i_s = lat.spin_int[s]

    # 按将这些数, 按进制规则转为一个大数, 从低位到高位分别是s, orb, y, x, z
    hole_uid = 0
    b_list = [b_s, b_orb, b_y, b_x, b_z]
    i_list = [i_s, i_orb, i_y, i_x, i_z]
    b = 1
    for idx in range(len(i_list)):
        i = i_list[idx]
        hole_uid += i * b
        b *= b_list[idx]
    assert hole == get_hole(hole_uid), 'check hole and get_hole are tuple'

    return hole_uid


def get_hole(hole_uid):
    """
    将空穴的数字信息转化为空穴信息
    :param hole_uid: 空穴的数字信息
    :return: hole = ('x', 'y', 'z', 'orb', 's'), orb轨道, s自旋
    """
    # 依次将x, y, z, orb, s的最大个数求出来, 并作为进制
    b_x, b_y, b_z = lat.b_x, lat.b_y, lat.b_z
    delta_x, delta_y = lat.delta_x, lat.delta_y
    b_orb = pam.Norb
    b_s = 2

    # 将大数依据进制规则, 提取每个进制上的数
    # 依次提取i_s, i_orb; i_y, i_x, i_z
    i_s = hole_uid % b_s
    hole_uid //= b_s
    i_orb = hole_uid % b_orb
    hole_uid //= b_orb
    i_y = hole_uid % b_y
    hole_uid //= b_y
    i_x = hole_uid % b_x
    hole_uid //= b_x
    i_z = hole_uid % b_z

    # 将这些数转为对应的空穴信息 hole = (x, y, z, orb, s)
    # i_x, i_y减去pam.Mc
    x = i_x - delta_x
    y = i_y - delta_y
    z = i_z
    # 得到轨道和自旋信息
    orb = lat.int_orb[i_orb]
    s = lat.int_spin[i_s]
    hole = (x, y, z, orb, s)

    return hole


def count_inversion(state):
    """
    计算一个态, 需要经过多少次交换才能得到规范化的顺序
    :param state: state是一个嵌套元组, state = (hole1, hole2, ...)
    :return: inversion, 交换次数
    """
    # 将state中的每个空穴转为hole_uid
    uid_state = map(get_hole_uid, state)  # 注意map是惰性求解器
    uid_state = tuple(uid_state)

    inversion = 0
    hole_num = len(uid_state)
    for i in range(1, hole_num):
        behind_uid = uid_state[i]
        for front_uid in uid_state[:i]:
            if front_uid > behind_uid:
                inversion += 1
    return inversion


def make_state_canonical(state):
    """
    将态中的每个空穴按照hole_uid升序排列,
    :param state: state = (hole1, hole2, ...)
    :return: canonical_state and phase
    canonical_state, 按照hole_uid升序后的态
    phase, 每交换一次顺序相位乘以-1
    """
    inversion = count_inversion(state)
    phase = 1.0 if inversion % 2 == 0 else -1.0
    canonical_state = sorted(state, key=get_hole_uid)
    canonical_state = tuple(canonical_state)
    return canonical_state, phase


def get_state_part(position_list, num_dist):
    """
    根据位置和空穴的数量分布生成所有可能的态
    :param position_list: 位置
    :param num_dist: 空穴的数量分布
    :return: 存储所有可能态的元组
    """
    # 去除分布中数量为0
    num_dist = [num for num in num_dist if num != 0]
    if not num_dist:
        return ()
    tot_num = len(num_dist)
    # 枚举空穴数量分布的所有可能排列
    num_dists = permutations(num_dist)
    num_dists = list(set(num_dists))

    state_part_list = []
    # 从position_list中挑出len(num_dist)个位置, 与num_dist相对应
    for positions in combinations(position_list, tot_num):
        for num_dist in num_dists:
            holes_list = []
            for i, position in enumerate(positions):
                num = num_dist[i]

                # 单个空穴的所有可能情况
                hole_list = []
                for orb in lat.get_unit_cell_rep(*position):
                    for s in ['dn', 'up']:
                        hole_list.append((*position, orb, s))

                # num个空穴的所有可能情况
                holes_tuple = tuple(combinations(hole_list, num))
                holes_list.append(holes_tuple)

            # 拼接不同部分的空穴
            for part1 in product(*holes_list):
                state_part = part1[0]
                for part2 in part1[1:]:
                    state_part += part2
                state_part_list.append(state_part)

    return tuple(state_part_list)


def get_state_Sz(state):
    """
    得到态的总自旋z分量
    :param state: state = ((x, y, z, orb, s), ...)
    :return: Sz
    """
    Sz = 0
    for hole in state:
        s = hole[-1]
        Sz += 1 / 2 if s == 'up' else -1 / 2
    return Sz


class VariationalSpace:
    def __init__(self, Sz):
        self.Sz = Sz
        self.b_hole = lat.b_x * lat.b_y * lat.b_z * pam.Norb * 2

        self.lookup_tbl = self.create_lookup_tbl()
        self.dim = len(self.lookup_tbl)
        print(f'Sz is {Sz}, VS.dim = {self.dim}')

    def create_lookup_tbl(self):
        """
        找出所有可能的态，并根据能量的大小，砍去一部分态后存储在列表中
        :return: lookup_tbl
        """
        t0 = time.time()
        hole_num = pam.hole_num

        Ni_position = lat.Ni_position
        Oap_position = lat.Oap_position
        O_position = lat.O_position

        Ni_num = len(Ni_position)
        Oap_num = len(Oap_position)

        lookup_tbl = []
        A = pam.A
        Upp = pam.Upp
        Uoo = pam.Uoo
        ep = pam.ep_list[4]
        eo = pam.eo_list[4]
        # 枚举Ni的所有可能分布: (Ni0_num, Ni1_num, ...)
        for Ni_dist in cwr((0, 1, 2, 3), Ni_num):
            if sum(Ni_dist) > hole_num:
                continue
            # 层间O的数量分布
            for Oap_dist in cwr((0, 1, 2), Oap_num):
                other_num = hole_num - sum(Ni_dist) - sum(Oap_dist)
                if other_num < 0:
                    continue
                # 层内O的数量分布
                for O_dist in cwr((0, 1, 2), other_num):
                    if sum(O_dist) != other_num:
                        continue

                    energy = 0
                    # 以d8为基态能量, 其他dn比d8高 A/2 * abs(n - 2)的能量
                    for num in Ni_dist:
                        energy += A/2 * abs(num-2)
                    # 除去分布中等于0的数
                    Ni_num_dist = [num for num in Ni_dist if num != 0]
                    Oap_num_dist = [num for num in Oap_dist if num != 0]
                    O_num_dist = [num for num in O_dist if num != 0]
                    # 计算Oap, O上的相互作用能和在位能
                    for num in Oap_num_dist:
                        energy += Uoo * num * (num - 1) / 2
                        energy += eo * num
                    for num in O_num_dist:
                        energy += Upp * num * (num - 1) / 2
                        energy += ep * num
                    # 不考虑高能态
                    if energy < pam.energy_range[0] or energy > pam.energy_range[1]:
                        continue

                    # 分别得到在Ni上, 层间O和层内O的空穴
                    Ni_part = get_state_part(Ni_position, Ni_num_dist)
                    Oap_part = get_state_part(Oap_position, Oap_num_dist)
                    O_part = get_state_part(O_position, O_num_dist)

                    # 存储三部分中的非空元组
                    non_empty_part = [part for part in (Ni_part, Oap_part, O_part) if part]
                    # 如果只有一部分非空, 直接输出
                    if len(non_empty_part) == 1:
                        for state in non_empty_part[0]:
                            # 限制Sz
                            if self.Sz != 'All_Sz':
                                Sz = get_state_Sz(state)
                                if Sz != self.Sz:
                                    continue
                            canonical_state, _ = make_state_canonical(state)
                            uid = self.get_state_uid(canonical_state)
                            lookup_tbl.append(uid)
                    # 若有两部分及以上非空, 则进行笛卡尔积, 即(a, b) @ (c, d) = (a, c), (a, d), (b, c), (b, d)
                    else:
                        # 拼接这几部分的空穴
                        for part1 in product(*non_empty_part):
                            state = part1[0]
                            for part2 in part1[1:]:
                                state += part2
                            # 限制Sz
                            if self.Sz != 'All_Sz':
                                Sz = get_state_Sz(state)
                                if Sz != self.Sz:
                                    continue
                            canonical_state, _ = make_state_canonical(state)
                            uid = self.get_state_uid(canonical_state)
                            lookup_tbl.append(uid)

        lookup_tbl.sort()       # 一定要有这一步, 这会影响get_index函数
        t1 = time.time()
        print(f'VS time {(t1-t0)//60//60}h, {(t1-t0)//60%60}min, {(t1-t0)%60}s')

        return lookup_tbl

    def get_state_uid(self, state):
        """
        将态信息转化为数字信息
        :param state: state = (hole1, hole2, ....)
        :return: uid, 态对应的数字
        """
        b_hole = self.b_hole

        # 将每个空穴数字, 按b_hole进制转成一个大数
        uid = 0
        for idx, hole in enumerate(state):
            i_hole = get_hole_uid(hole)
            uid += i_hole * (b_hole ** idx)
        assert state == self.get_state(uid), 'check state and get_state are tuple'

        return uid

    def get_state(self, uid):
        """
        将态的数字信息转为态信息
        :param uid: 态的数字信息
        :return: state = (hole1, hole2, ....)
        """
        b_hole = self.b_hole

        # 将大数uid按照b_hole进制, 提取每个进制上的数
        state = []
        while uid:
            hole_uid = uid % b_hole
            hole = get_hole(hole_uid)
            state.append(hole)
            uid //= b_hole
        state = tuple(state)

        return state

    def get_index(self, state):
        """
        根据lookup_tbl, 找到state对应的索引
        :param state: state = ((x1, y1, z1, orb1, s1)...)
        :return: index, state对应在lookup_tbl中的索引
        """
        uid = self.get_state_uid(state)
        index = bisect.bisect_left(self.lookup_tbl, uid)

        # 判断索引是否超出lookup_tbl
        if index < self.dim:
            if self.lookup_tbl[index] == uid:
                return index
            else:
                return None
        else:
            return None
