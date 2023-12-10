N = 5;
j = 2;
lam = 1;
y = 1;

objective_function_LASSO = @(x, A, y, lam, w) lam*sum(w*abs(x)) + sum((A*x - y).^2); % lambda*||x||_{1,w} + ||Ax - y||_{2}^{2}

t = linspace(0,1,N+1);
A = zeros([1, N]);

for i = 1:j
    A(1,i) = (1-t(j+1))/(1-t(i+1));
end
for i = j+1:N
    A(1,i) = t(j+1)/t(i+1);
end

w1 = A;


sol_problem1  = compute_LASSO_solution_FEM(A, y, lam, w1)

val_problem1 = objective_function_LASSO(sol_problem1, A,y,lam, w1)


w2 = w1;
w2(1) = w2(1) - 1e-8;

sol_problem2  = compute_LASSO_solution_FEM(A, y, lam, w2)

val_problem2 = objective_function_LASSO(sol_problem1, A,y,lam, w2)


