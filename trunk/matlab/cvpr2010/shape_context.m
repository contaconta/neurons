function S = shape_context(X, E, Rmin, Rmax, Rbins, Tbins)
% 
% X = location to compute shape context from [r; c]
% E = list of all edge points in the image [r r r; c c c]
% T = associated tangents
% Rbins
% Rmin
% Rmax
% Tbins
% returns S, a row vector containing the shape context

if size(E,1) < size(E,2)
    E = E';     % E is Nx2
end
X=X(:)';        % X is 1x2

r_array = real(sqrt(dist2(E, X)));


r_array = r_array(r_array <= Rmax);



r_bin_edges = logspace(log10(Rmin), log10(Rmax), Rbins);


% maybe try to make it scale invariant by normalizing?

r_array_q = zeros(size(r_array));

for m = 1:Rbins
    r_array_q = r_array_q + (r_array < r_bin_edges(m));
end

%theta_array=atan2(E(2,:)'*ones(1,nsamp)-ones(nsamp,1)*E(2,:),E(1,:)'*ones(1,nsamp)-ones(nsamp,1)*E(1,:))';

T = E( r_array <= Rmax, :);
%t_array = atan2( T(:,1) - X(1), T(:,2) - X(2));
t_array = atan2( T(:,2) - X(2), T(:,1) - X(1));


t_array = rem(rem(t_array,2*pi)+2*pi,2*pi);

t_array_q = 1+floor(t_array/(2*pi/Tbins));

Nbins = Rbins*Tbins;
%S = zeros(1, Nbins);

S = sparse(t_array_q, r_array_q, 1, Tbins, Rbins);

S = full(S(:)');