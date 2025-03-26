# La3Ni2O7, 单层2个Ni, 层外有2个O
注意事项:
1. 在改变lattice的时候, 要注意d10这种态, 会影响variational_space.py中的get_atomic_energy和get_state_type, 
hamiltonian.py中的create_Esite_matrix
2. 注意顺序问题, 一个是get_interaction_mat中的state_order, ('d3z2r2', 'dx2y2'), d3z2r2的字典序在dx2y2的前面;
另外一个是由于态中空穴的排列方式带来phase(make_state_canonical函数)的问题
3. 另外代码还没加入dxy, dxz, dyz轨道
4. hamiltonian里, 要if, elif, elif, 最后是else; 或者是if if if, 不要else