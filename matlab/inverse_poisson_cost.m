function cret = inverse_poisson_cost(O, A, w, x, d)
    c = inv(A)*x';  	
    cc = O * c;  
    cc = cc - d;   
    w = abs(w);  
    x = abs(x);  
    cret = cc*cc + w*x';   
