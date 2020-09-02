% Controller implemented in state space form with custom reference tracking.
A = [0 , 1 ; 0 , 0 ] ; B = [0 ; 1];  C = [1,0];
G1 = ctrb(A,B); G2 = obsv(A,C);

assert(rank(G1) == length(G1), "Controllability Error");
assert(rank(G2) == length(G2), "Observability Error");

%************* Desired Poles *******************
% Poles of controllability matrix
% p1 = [-1 , -2];
p1 = [ -20+5j , -20-5j];
%--------------------------
% Poles of observability matrix
p2 = [-20,-19];
%***********************************************

%******* Function to track,Sort of external System ********
time = 0:0.01:9;
des = sin(3*time);
% des = zeros(2,length(time));
% des(1,:) = 10;
des1 = getRand(0, 2 ,time);   


%**********************************************************

k = place(A  , B  , p1);
l = (place(A' , C' , p2))';

x = [1;1] ; xh = [0;0] ;t = 0 ; tf = 9; dt = 0.01;
X = animatedline("Color","r");
XH = animatedline("Color","b");
XD = animatedline("Color","g");
legend("Desired Trajectory","Error","Followed Trajectory")
count = 1;
while(t<=tf)
 	xd = des(:,count);
%     xd = des1(count);
	u  = -k*(xh);
	y  = C*x;
	addpoints(X,time(count),x(1));
	addpoints(XD, time(count),xd(1));
	addpoints(XH, time(count),xh(1));
	drawnow;
	x  = x + dt*(A*x+B*u);
	xh = xh+ dt*(A*xh + B*u + l*(y - C*(xh+xd)));
	t  = t + dt;
	count = count + 1;
end
drawnow;
hold on
% plot(time,XD(1,:));
% plot(time,X(1,:)) ;
% plot(time,XH(1,:));

grid on;

function op = getRand(start,del,time)
    op = [];
    for i= 1:length(time)
        op = [op, start + del*(rand()-0.5)];
        start = op(i);
    end
end
