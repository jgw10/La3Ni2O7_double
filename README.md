# La3Ni2O7_double
注意事项:
1. 当layer_num = 2, hole_num = 10时, 一定得调节max_energy(cut_off)
2. 让layer_num = 1, hole_num = 4, 5 or 6时, 可以回到单层Ni2O7, 可以和之前Ni2O9的计算结果进行对比
3. 在改变lattice的时候, 要注意d10这种态, 会影响variational_space.py中的get_atomic_energy和get_state_type, 
hamiltonian.py中的create_Esite_matrix
4. 注意顺序问题, 一个是get_interaction_mat中的state_order, ('d3z2r2', 'dx2y2'), d3z2r2的字典序在dx2y2的前面;
另外一个是由于态中空穴的排列方式带来phase(make_state_canonical函数)的问题
5. 另外代码还没加入dxy, dxz, dyz轨道
6. hamiltonian里, 要if, elif, elif, 最后是else; 或者是if if if, 不要else