% Controller implemented in state space form, reference is considered to be origin of configuration space.
A = [0 , 1 ; 0 , 0 ] ; B = [0 ; 1];  C = [1,0];
G1 = ctrb(A,B); G2 = obsv(A,C);

assert(rank(G1) == length(G1), "Controllability Error");
assert(rank(G2) == length(G2), "Observability Error");

%************* Desired Poles *******************
% Poles of controllability matrix
% p1 = [-1 , -2];
p1 = [-2-5j , -2+5j];
%--------------------------
% Poles of observability matrix
p2 = [-9, -10];
%***********************************************

%******* Function to track,Sort of external System ********
time = 0:0.01:9;
% des = sin(3*time);
des = time>=0;
%**********************************************************

k = place(A  , B  , p1);
l = (place(A' , C' , p2))';

x = [1;1] ; xh = [0;0] ;t = 0 ; tf = 9; dt = 0.01;
X = [] ; T =[]; XH = [];
while(t<=tf)
	u = -k*(xh);
	y = C*x;
	XH = [XH,xh];
	X=[X,x];
	x = x + dt*(A*x+B*u);
	xh = xh+ dt*(A*xh + B*u + l*(y - C*xh));
	t = t + dt;
end

hold on
% plot(time,des)
plot(time,X(1,:)) 
plot(time,XH(1,:))
