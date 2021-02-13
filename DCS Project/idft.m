function f_hat = idft(f1)
    n = length(f1);
    f1 = reshape(f1,[n,1]);    
    v = 0:n-1;
    w_n = exp(2*pi*1i/n);
    mat = w_n.^(v'*v);
    disp(size(mat));
    disp(size(f1));
    f_hat = (mat*f1)./n;     
end