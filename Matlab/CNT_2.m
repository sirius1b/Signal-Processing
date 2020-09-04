% Lavanya Verma(2018155)
% Control Theory TUT 2

%********* Question 1
A = [2,1,3;1,2,3;5,1,0];
A_inv_1 = inv(A);
A_inv_2 = A^(-1);
det_A = det(A);



%********* Question 2
A_2 = [1, 2, -1;1, 1, 2;1, -1, -1];
b = [2;0;1];
x1 = inv(A_2)*b; 
x2 = A_2\b; % slightly better than prev approach
x3 = linsolve(A_2,b); %best among 3

%********* Question 3
A3 = [-12, -19, -3, 14, 0; -12, 10, 14, -19, 8; 4, -2, 1, 7,-3; -9, 17, -12, -5, -8; -12, -1, 7, 13, -12];
eig_A3 = eig(A3); 
trace_A3 = trace(A3);
syms x; 
poly_A3 = charpoly(A3,x);

%********* Question 4
A4 = [2, 2, 2; -1, 2 ,1; 1, -2, -1];
[eig_vec_A4,eig_val_A4] = eig(A4);	

%********* Q 5
syms y(x) ;
ode1 = diff(y(x),x) == x*y ;
cond1 = y(1) == 1;
ySol(x) = dsolve(ode1,cond1);

%********* Q 6
Dy = diff(y,x);
ode2 = diff(y,x,2) + 8*diff(y,x) + 2*y == cos(x);
cond2_1 = y(0) == 0;
cond2_2 = Dy(0) == 1;
ySol2(x) = dsolve(ode2,[cond2_1,cond2_2]);

%********* Q 7
ode3 = diff(y,x) == x*y^2 + y;
[V] = odeToVectorField(ode3);
[tSol, ySol3] = ode45(matlabFunction(V, 'vars',{'x','Y'}), [0 0.5],1);

%********* Q 8
syms x(t);
ode4 = diff(x,t) == 3*exp(-t);
[V1] = odeToVectorField(ode4);
[tSol_1, ySol4] = ode45(matlabFunction(V1, 'vars',{'x','t'}), [0, 5], 0); 
