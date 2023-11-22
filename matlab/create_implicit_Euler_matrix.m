% Create system matrix for a discitized implicit Euler system
%
% Arguments
% ---------
% m (int) : Number of rows
% a (float) : ODE parameter [ u_t = -au + f ], a > 0
% h_vec (vector) : Discretization parameters
% 
% Returns
% ---------
% A (matrix) : A m x N matrix with the implcit Euler system matrix
%              Here N = length(h_vec)    
%
function A = create_implicit_Euler_matrix(m, a, h_vec)

    N = length(h_vec)
    A = zeros([m,N]);
    h_bar_vec  = 1./(1+a*h_vec);

    % Create the lower triangular part
    for n = 1:(m-1)     % Row index
        for i = 1:n % Column index
            prod_entry = prod(h_bar_vec(i:n));
            A(n,i) = h_vec(i)*prod_entry;
        end
    end

    % Insert the components in the last row
    for i = 1:N % Column index 
            prod_entry = prod(h_bar_vec(i:N));
            A(m,i) = h_vec(i)*prod_entry;
    end
end
