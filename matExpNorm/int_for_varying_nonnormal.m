A = [-1 5; 0 -2];
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

[thetas,eAvInts] = integral_comparison2D(A);

% alphas = 0:0.1:1;
% % alphas = 0;
% alphas = [alphas 1000];
% A = -.1*eye(2);
% [thetas,eAvInts] = integral_comparison2D(A);
% alleAVs = zeros(length(alphas),length(eAvInts));
% alleAVs(1,:) = eAvInts;
% 
% for i = 2:length(alphas)
%     A = -.1*eye(2) + diag(alphas(i) .* ones(1, size(A, 1) - 1), 1);
%     [thetas,eAvInts] = integral_comparison2D(A);
%     alleAVs(i,:) = eAvInts;
% 
% end
% figure(2);
% for i = 1:size(alleAVs,1)
%     plot(thetas,alleAVs(i,:),'DisplayName',"alpha: " + alphas(i))
%     hold on
% end
% legend
% hold off

