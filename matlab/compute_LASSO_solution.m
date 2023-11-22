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
function z_hat = compute_LASSO_solution(A, y, lam, w)

    [m, N] = size(A);
    d = y(end); 
    
    if (nargin < 4)
        w = ones([1,N]);
    end
    if iscolumn(w)
        w = w';
    end

    diagA = zeros([m-1,1]);
    for i = 1:m-1
        diagA(i) = A(i,i);
    end
    R_1 = max( diagA./w(1:m-1) );
    R_2_vec = A(m,m:end)./w(m:end);
    R_2 = max(R_2_vec(:));
    
    if R_2  <= R_1
        emsg = sprintf('Must have that R_2 > R_1. Currently these values are R_2 = %g, R_1 = %g', R_2, R_1);
        error(emsg);
    end
    if (R_2 <= lam/(2*d))
        emsg = sprintf('Must have R_2 > lambda/(2*y[-1]). Currently these values are R_2 = %g, lam/(2*d) = %g.', R_2, lam/(2*d));   
        error(emsg);
    end
    if norm(y(1:end-1)) > 1e-16 
        emsg = 'The m-1 first entries of y must be zero.'; 
        error(emsg);
    end
     
    support_set_solutions = R_2_vec == R_2; % \mathcal{J}
    number_of_non_zero_elements = sum(support_set_solutions);
    
    if number_of_non_zero_elements > 1
        disp('Multivalued solution');
    end
    
    idx_sol = find(support_set_solutions); % Pick the first non-zero index.
    idx_sol = m-1 + idx_sol(1);
    
    mat_entry = A(m,idx_sol);
    z_hat = zeros([N,1]);
    z_hat(idx_sol) = (d/mat_entry) - (lam/(2*R_2*mat_entry));

end 
