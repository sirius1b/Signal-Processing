A = [0 , 1 ; 0 , 0 ] ; B = [0 ; 1];

% p = [-1 , -2];
p = [-2-5j , -2+5j];



k = place(A, B , p )

x = [1;1]; t= 0 ; tf = 5 ; dt = 0.01;
X = [] ; T = [];
while(t < tf)
    X = [X,x]; T = [T;t];
    x = x+dt*(A-B*k)*x;
    t = t+dt;
end
hold on 
plot(T,X(1,:));

G = ctrb(A,B)
rank(G)