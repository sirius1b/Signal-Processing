function [m,t1,rcP1,t] = PulseShaping(bits,T_period)
    bits(bits == 0) = -1;
    t = -length(bits)*T_period/2:T_period:(length(bits)-1)*T_period/2;
    rcP1 = rcPulse(T_period*1,0.2,t);
    m = conv(bits,rcP1)*T_period;
    t1 = -length(bits)*T_period:T_period:(length(bits)-2)*T_period;
%     [m,t1] = contconv(bits,rcP1,t,t,T_period);_
end

function x = rcPulse(T,r,t)
    x = sin(pi.*t/T).*cos(pi.*t*r/T)./((1- (2*r.*t/T).^2).*(pi.*t)); 
end

function [out,time]= contconv(s1,s2,t1,t2,dt)
    time = [];
    out = [];
    for i = 1:length(s1)+length(s2)
        time = [time;t1(1)+t2(1)+i*dt];
        var = 0;
        for j = 1:length(s1)+length(s2)
            if (sum(1:length(s1) == j) ~= 0 && sum(1:length(s2) == i-j) ~= 0)
                var = var+ s1(j)*s2(i-j);
            end
        end
        out = [out;var];
    end
    out = dt*out;
end