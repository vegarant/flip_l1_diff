

% note grid is of size n while hvec is of size n-1 
grid = [0:0.1:1]
n = size(grid)(2) 
w = grid.*(1-grid) 
w1 = w(1:n-1) 
w2 = w(2:n) 
hvec = geth(grid) 
A = poisson_1D(hvec) 
observations = zeros(size(hvec)) 
k = find( grid == 0.5) 
observations(k) = 1 
data = 0.3 

x = rand(size(hvec))  
a = inverse_poisson_cost(observations, A, w1, x, data) 
a = inverse_poisson_cost(observations, A, w2, x, data) 
