function eps = calc_min_eps(A,varargin)
    addpath '/Users/Greencat/Spring2026-Connectome-Pseudospectra/eigtool-master/num_comp/pseudo_abscissa';
    if nargin < 2
        lbd = 0;
        ubd = 10;
    end
    abs_func = @(x) pspa_2way(A,x);
    eps = fzero(abs_func,[lbd ubd]);

end