function A = poisson1DMatrix(hvec)
    n = numel(hvec); % Number of nodes

    % Initialize the sparse matrix A
    A = sparse(n, n);

    % Loop through the interior nodes to fill the matrix
    for i = 2:n-1
        h_ip = hvec(i);
        h_im = hvec(i-1);
        h_ih = 0.5*(h_ip + h_im);

        % Diagonal element
        A(i, i) = 1 / (h_ih*h_im) + 1 / (h_ih*h_ip);

        % Off-diagonal elements
        A(i, i-1) = -1 / (h_ih*h_im);
        A(i, i+1) = -1 / (h_ih*h_ip);
    end
    % check with dense comp for starters 
    % Neumann:  
    % A(1, 1) = 1/h_ip;  
    % A(1, 2) = -1/h_ip;  
    % A(n, n) = 1/h_im;  
    % A(n, n-1) = -1/h_im;  
    % Dirichlet: 
    A(1, 1) = 1; 
    A(n, n) = 1; 
    A = full(A); 
end
