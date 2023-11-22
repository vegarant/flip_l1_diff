clear('all'); close('all');

% Choose the problem parameters 
N = 5;
m = 3;
a = 1;
lam = 1;
d = 3;
w = ones([N,1]); % The weights
y = zeros([m,1]);
y(end) = d;

create_h_vec = @(grid) grid(2:end) - grid(1:end-1);  % x_{i} - x_{i-1}
objective_function_LASSO = @(x, A, y, lam, w) lam*sum(w.*abs(x)) + sum((A*x - y).^2); % lambda*||x||_{1,w} + ||Ax - y||_{2}^{2}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Consider the first discretization 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% I have handcrafted these grids by trail and error. 
grid1 = [0, 0.05, 0.2, 0.63,.72,  1];

% Create problem data
h_vec1 = create_h_vec(grid1);
h_vec1
A1 = create_implicit_Euler_matrix(m, a, h_vec1);

A1

% Solve the problem 
solution_vector_problem1 = compute_LASSO_solution(A1,y,lam);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Consider the second discretization 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
grid2 = [0, 0.05, 0.2, 0.63,0.73, 1];

% Create problem data
h_vec2 = create_h_vec(grid2);
A2 = create_implicit_Euler_matrix(m, a, h_vec2);

A2

% Solve the problem 
solution_vector_problem2 = compute_LASSO_solution(A2, y, lam);

solution_vector_problem1
solution_vector_problem2


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Evaluate the two solutions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

objective_problem1 = @(x) objective_function_LASSO(x, A1, y, lam, w);
objective_problem2 = @(x) objective_function_LASSO(x, A2, y, lam, w);

fprintf('\nPROBLEM 1\n');
value_problem1_solution_problem1 = objective_problem1(solution_vector_problem1);
value_problem1_solution_problem2 = objective_problem1(solution_vector_problem2); 
fprintf('Value of objective function in problem 1:\n');
fprintf('Solution vector for problem 1: %g.\n', value_problem1_solution_problem1);
fprintf('Solution vector for problem 2: %g.\n', value_problem1_solution_problem2);

fprintf('\nPROBLEM 2\n')
value_problem2_solution_problem1 = objective_problem2(solution_vector_problem1); 
value_problem2_solution_problem2 = objective_problem2(solution_vector_problem2);
fprintf('Value of objective function in problem 2:\n');
fprintf('Solution vector for problem 1: %g.\n', value_problem2_solution_problem1);
fprintf('Solution vector for problem 2: %g.\n', value_problem2_solution_problem2);

