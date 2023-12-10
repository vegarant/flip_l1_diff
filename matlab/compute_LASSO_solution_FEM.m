% Compute the LASSO solution for the problem considered in the lemma in the paper.
%
% Arguments
% ---------
% A (matrix) : The  m x N matrix computed by the `create_implicit_Euler_matrix`
% y (vector) : The vector for the LASSO problem. We assume 
%              that y=[0, ..., 0, d], where d >0 
% lam (float): The non-negative regularization parameter
% w (vector) : Weights for the weighted l^1 norm. If `w = None`, then 
%              unweighted l^1-norm is used.
%
% Returns
% ------- 
% z_hat (vector) : The solution vector to the LASSO problem
function z_hat = compute_LASSO_solution_FEM(A, y, lam, w)

    [m1, N] = size(A);
    
    if m1 ~= 1
        emsg = sprintf('`A` must be a row matrix with one row. Current `A` has %d rows', m1);
        error(emsg);
    end
    if y ~= 1
        emsg = sprintf('`y` must equal 1');
        error(emsg);
    end
    
    if (nargin < 4)
        w = ones([1,N]);
    end
    if iscolumn(w)
        w = w';
    end
    max_value = max(lam*w./A)
    min_value = min(lam*w./A)
    
    if min_value <= -0 
        emsg = sprintf('Must have `lam*w_j*rho_j^{-1} > 0`. Currently the smalles value is %g', min_value);
        error(emsg);
    end

    if max_value >= 2 
        emsg = sprintf('Must have `lam*w_j*rho_j^{-1} < 2`. Currently the largest value is %g', min_value);
        error(emsg);
    end
     
    support_set_solutions =  min_value ==  lam*w./A; 
    number_of_non_zero_elements = sum(support_set_solutions);
    t_j = 1/number_of_non_zero_elements;
    
    if number_of_non_zero_elements > 1
        disp('Multivalued solution');
    end
    
    idx_sol = find(support_set_solutions); % Pick the first non-zero index.
   
    z_hat = zeros([N,1]);
    z_hat(idx_sol) = (1 - 0.5*min_value)*t_j./A(idx_sol);
end 




