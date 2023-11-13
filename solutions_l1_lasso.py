import numpy as np


def create_implicit_Euler_matrix(m, a, h_vec):
    """
    m (int): Number of rows
    a (float): ODE parameter [ u_t = -au + f ], a > 0
    h_vec (ndarray): discretization parameters
    """

    N = len(h_vec)
    A = np.zeros([m,N]);
    h_bar_vec  = 1/(1+a*h_vec);
    #print('h_bar_vec: ', h_bar_vec)
    
    # Create the lower triangular part
    for n in range(m):   # Row index
        for i in range(n+1): # Column index
            prod_entry = np.prod(h_bar_vec[i:n+1])
            #print(f'n: {n}, i: {i}, prod_entry: {prod_entry}')
            A[n,i] = h_vec[i]*prod_entry
    
    # Insert the components in the last row
    for i in range(m-1,N): # Column index 
            prod_entry = np.prod(h_bar_vec[i:N+1])
            #print(f'n: {n}, i: {i}, prod_entry: {prod_entry}')
            A[n,i] = h_vec[i]*prod_entry
        
    return A


def compute_LASSO_solution(A,y, lam,w=None):
    """
    Compute the LASSO solution for the problem considered in the lemma in the paper.

    A (ndarray) : The  m x N matrix computed by the `create_implicit_Euler_matrix`
    y (ndarray) : The vector for the LASSO problem. We assume 
                  that y=[0, ..., 0, d], where d >0 
    lam (float) : The non-negative regularization parameter
    w (ndarray) : Weights for the weighted l^1 norm. If `w = None`, then 
                  unweighted l^1-norm is used.
    """

    m, N = A.shape
    d = y[-1]   

    if w is None:
        w = np.ones(N);
    
    diagA = A.diagonal()[:-1] # Find the diagonal of the upper m-1 x m-1 block
    R_1 = np.amax(diagA/w[:m-1]);
    #print('R_1: ', R_1)
    R_2_vec = A[m-1,m-1:]/w[m-1:]
    R_2 = np.amax(R_2_vec)
    #print('R_2: ', R_2)
    #print('lam/(2*d): ', lam/(2*y[-1]))
    assert R_2 > R_1, f'Must have that R_2 > R_1. Currently these are R_2 = {R_2}, R_1 = {R_1}.'
    assert R_2 > lam/(2*d), f'R_2 > lambda/(2*y[-1]). Currently these values are {R_2} < {lam/(2*d)}'
    assert np.sum(np.abs(y[:-1])) < 1e-16, 'The m-1 first entries of y must be zero'  
     
    support_set_solutions = R_2_vec == R_2 # \mathcal{J}
    number_of_non_zero_elements = np.sum(support_set_solutions)
    if number_of_non_zero_elements > 1:
        print('Multivalued solution')
    
    idx_sol = np.argmax(support_set_solutions) # Pick the first non-zero index.
    idx_sol = m-1 + idx_sol
    
    mat_entry = A[m-1,idx_sol]
    z_hat = np.zeros(N);
    z_hat[idx_sol] = (d/mat_entry) - (lam/(2*R_2*mat_entry));
    return z_hat

def objective_function_LASSO(x,A,y,lam,w=None):
    " Evaluate the LASSO objective function "
    m,N = A.shape
    
    if w is None:
        w = np.ones(N)
    
    obj = lam*np.sum(w*np.abs(x)) + np.sum((A@x-y)**2)
    return obj


if __name__ == "__main__":
    create_h_vec = lambda grid: grid[1:] - grid[:-1];   # x_{i} - x_{i-1} 
    create_y_vec = lambda m,d: np.array((m-1)*[0]+[d]); # y = [0,...,0,d] (length m)
    
    # Choose the problem parameters 
    N = 5
    m = 3
    a = 1
    lam = 1
    d = 3;
    w = np.ones(N) # The weights
    y = create_y_vec(m,d);
   
    ######################################
    # Consider the first discretization 
    ######################################
    # I have handcrafted these grids by trail and error. 
    grid1 = np.array([0, 0.05, 0.2, 0.63,.72,  1]) 

    # Create problem data
    h_vec1 = create_h_vec(grid1)
    A1 = create_implicit_Euler_matrix(m, a, h_vec1)
    
    # Solve the problem 
    solution_vector_problem1 = compute_LASSO_solution(A1,y,lam)
    
    ######################################
    # Consider the second discretization 
    ######################################
    grid2 = np.array([0, 0.05, 0.2, 0.63,0.73, 1]) 

    # Create problem data
    h_vec2 = create_h_vec(grid2)
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




