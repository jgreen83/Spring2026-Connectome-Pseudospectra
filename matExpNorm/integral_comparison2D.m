% A = [-1 5; 0 -2];
% dim = 2;
% A = randn(dim); 
% [Q, ~] = qr(A);       % Q is a random orthogonal/unitary matrix
% R = triu(randn(dim));
% A = Q*R;
% mReal = max(real(eig(A)));
% if (mReal > 0)
%     % A = A*(1-10^(-2))/mReal;
%     A = A - (mReal + 10^-2)*eye(dim);
% end

function [thetas,eAvInts] = integral_comparison2D(A)

tMax = 50;
ts = 0 : .01: tMax;
th_ts = zeros(size(ts));
eAs = zeros(size(ts));
for i = 1:length(ts)
    t = ts(i);
    eAs(i) = norm(expm(t.*A));
    [~,~,v] = svd(expm(t.*A));
    vMax = v(:,1);
    th_ts(i) = atan(vMax(2)/vMax(1)); %EXPLICITLY FOR 2D
end

c = 0;
%from goldman supplemental materials
%EPSILON = smallest epsilon for which sigma_eps touches x = 0

eps = calc_min_eps(A); 
disp("eps: " + eps)
M = max(eAs);
t_star = (1-c)/(eps*M);

tsReal = 0:0.01:t_star;

eAsReal = zeros(size(tsReal));
for i = 1:length(tsReal)
    t = tsReal(i);
    eAsReal(i) = norm(expm(t.*A));
end
eAint = trapz(tsReal,eAsReal);

N_ths = 100;
thetas = 0:pi/N_ths:pi - pi/N_ths;
eAvInts = zeros(size(thetas));
for i = 1:length(thetas)
    theta = thetas(i);
    v = [cos(theta); sin(theta)];
    eAvs = zeros(size(tsReal));
    for j = 1:length(tsReal)
        t = tsReal(j);
        eAvs(j) = norm(expm(t.*A)*v);
    end
    eAvInts(i) = trapz(tsReal,eAvs)/eAint;
end

clf
figure(2);
plot(thetas,eAvInts);

th_ts = mod(th_ts,pi);
figure(3);
plot(ts(1:round(length(ts)/2)),th_ts(1:round(length(ts)/2)));

end


