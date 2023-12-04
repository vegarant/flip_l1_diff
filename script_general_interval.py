from solutions_l1_lasso import (
    create_implicit_Euler_matrix,
    compute_LASSO_solution,
    objective_function_LASSO,
    create_h_vec, 
    create_y_vec
)

import numpy as np


def cretate_flipping_grid(a, N, m, h_m, h_N, eps=1e-5):
    
    assert h_m > h_N, f'h_m  must be greater than h_N, h_m = {h_m}, h_N = {h_N}.'
    assert N > m+1, f'N = {N} must be greater than m+1 = {m+1}.'
    # Start by computing h
    tmp_var = ( h_m / (h_N*(1+a*h_m)) )**(1/(N-m-1))
    h = (tmp_var - 1)/a
    assert h > 0, f'h = {h} < 0, try with different values for h_m and h_N.' 
    
    grid1 = np.zeros(N+1);
    grid2 = np.zeros(N+1);
    for grid_idx, grid in enumerate([grid1, grid2]):
        grid[N] = 1
        grid[N-1] = 1-h_N;
        s = 1 - h_N;
        # Uniform grid in a small area
        for i in range(N-2,m-1,-1):
            s -= h
            grid[i] = s
        
        grid[0:m] = np.linspace(0,s-h_m, m); 
        if grid_idx == 0:
            grid[m-1] = s-h_m + eps;
        else:
            grid[m-1] = s-h_m - eps;
        
    print (grid1) 
    print (grid2) 
    return grid1, grid2


if __name__ == "__main__":

    # Choose the problem parameters 
    N = 10
    m = 3
    a = 1
    lam = .1
    d = 3;
    w = np.ones(N) # The weights
    y = create_y_vec(m,d);
    
    h_N = 0.2
    h_m = 0.3
    grid1, grid2 = cretate_flipping_grid(a, N, m, h_m, h_N, eps=1e-5)
    h_vec1 = create_h_vec(grid1)
    A1 = create_implicit_Euler_matrix(m, a, h_vec1)

 
    ######################################
    # Consider the first discretization 
    ######################################

    # Create problem data
    h_vec1 = create_h_vec(grid1)
    print (h_vec1) 
    A1 = create_implicit_Euler_matrix(m, a, h_vec1)
    
    # Solve the problem 
    solution_vector_problem1 = compute_LASSO_solution(A1,y,lam)
    
    ######################################
    # Consider the second discretization 
    ######################################

    # Create problem data
    h_vec2 = create_h_vec(grid2)
    print (h_vec2) 
    A2 = create_implicit_Euler_matrix(m, a, h_vec2)

    # Solve the problem 
    solution_vector_problem2 = compute_LASSO_solution(A2, y, lam)
    print(f'Solution vector problem 1: {solution_vector_problem1}')
    print(f'Solution vector problem 2: {solution_vector_problem2}')
    

    ######################################
    # Evaluate the two solutions
    ######################################

    objective_problem1 = lambda x: objective_function_LASSO(x, A1, y, lam, w)
    objective_problem2 = lambda x: objective_function_LASSO(x, A2, y, lam, w)

    print('\nPROBLEM 1\n')
    value_problem1_solution_problem1 = objective_problem1(solution_vector_problem1) 
    value_problem1_solution_problem2 = objective_problem1(solution_vector_problem2) 
    print('Value of objective function in problem 1:')
    print(f'Solution problem 1: {value_problem1_solution_problem1}')
    print(f'Solution problem 2: {value_problem1_solution_problem2}')

    print('\nPROBLEM 2\n')
    value_problem2_solution_problem1 = objective_problem2(solution_vector_problem1) 
    value_problem2_solution_problem2 = objective_problem2(solution_vector_problem2) 
    print('Value of objective function in problem 2:')
    print(f'Solution problem 1: {value_problem2_solution_problem1}')
    print(f'Solution problem 2: {value_problem2_solution_problem2}')








