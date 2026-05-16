%start with a sample matrix that I know is nonnormal
A = [-1 5; 0 -2];
[V,D] = eig(transpose(A)*A);
[U,T] = schur(A);

ts = 0 : .01: 4;
eAs = zeros(size(ts));
for i = 1:length(ts)
    t = ts(i);
    eAs(i) = norm(expm(t.*A));
end

%setting thresh c = 1 just to test
c = 0.5;

%from goldman supplemental materials
%EPSILON = smalles epsilon for which sigma_eps touches x = 0
eps = 10^(-.44); %NEED TO DEFINE NONMANUALLY LOL %use eigtool
M = max(eAs);
t_star = (1-c)/(eps*M);

eA_star = expm(t_star.*A);
[u,s,v] = svd(eA_star);
vbig = v(:,1);

y1 = zeros(size(ts));
y2 = zeros(size(ts));
y3 = zeros(size(ts));
y4 = zeros(size(ts));
y5 = zeros(size(ts));

for i = 1:length(ts)
    t = ts(i);
    eA = expm(t.*A);
    [u,s,v] = svd(eA);
    [u1,s1,v1] = svd(expm(t.*T));
    y1(i) = norm(eA * v(:,1)); %this should be equal to the matrix norm
    %plot lower bd
    y2(i) = 1 - M*eps*t;

    y3(i) = norm(expm(t.*A));
    y4(i) = norm(eA * vbig);

    % y5(i) = norm(eA*U(:,2));

end


figure;
plot(ts,y3,'DisplayName','matrix norm')
hold on
plot(ts,c*ones(size(ts)),'DisplayName','y = c')
plot(t_star*ones(size(0:.1:1.5)),0:.1:1.5,'DisplayName','t = t^*')
%evs
% plot(ts,y1)
plot(ts,y2,'DisplayName','lower bound')
plot(ts,y4,'DisplayName','first rsv of t_star expm')
%schur modes
% plot(ts,y5)
legend