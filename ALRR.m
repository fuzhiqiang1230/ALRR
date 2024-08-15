function [Z,S,U,F,E,W] = ALRR(X,F_ini,Z_ini,c,lambda1,lambda2,lambda3,max_iter,Ctg)
[m,n] = size(X);
% ---------- Initilization -------- %
miu = 0.01;
rho = 1.2;
max_miu = 1e8;
tol  = 1e-5;
tol2 = 1e-2;
C1 = zeros(m,n);
C2 = zeros(n,n);
C3 = zeros(n,n);
E  = zeros(m,n);
W = eye(m);
for iter = 1:max_iter
    if iter == 1
        Z = Z_ini;
        Y = F_ini*F_ini';
        S = Z_ini;
        U = Z_ini;
        clear Z_ini F_ini
    end
    S_old = S;
    U_old = U;
    Z_old = Z;
    E_old = E;
    % -------- Update Z --------- %
    Z = Ctg*(X'*(X-E+C1/miu)+S+U-(C2+C3)/miu);
    Z = Z- diag(diag(Z));

    % -------- Update S --------- %
    D = diag(Y)*ones(1,n)-Y;
    distX = L2_distance_1(W*X,W*X);
%     distF = L2_distance_1(F',F');            % 计算vij
%     dist  = distX+lambda1*distF;
%     S     = Z+(C2-dist)/miu;
    S     = -lambda1/(2*miu)*(D+D')-distX/miu+Z+C2/miu; 
    S     = S - diag(diag(S));
    for ic = 1:n
        idx    = 1:n;
        idx(ic) = [];
        S(ic,idx) = EProjSimplex_new(S(ic,idx));          % 
    end
    % ---------- Update Y ----------- %
    B = (S+S')/2;
    L = diag(sum(B)) - B;
    [F, ~, ev] = eig1(L, c, 0);
    Y = F*F';
    % -------- Update U --------- %
    [AU,SU,VU] = svd(Z+C3/miu,'econ');
    AU(isnan(AU)) = 0;
    VU(isnan(VU)) = 0;
    SU(isnan(SU)) = 0;
    SU = diag(SU);    
    SVP = length(find(SU>lambda2/miu));
    if SVP >= 1
        SU = SU(1:SVP)-lambda2/miu;
    else
        SVP = 1;
        SU = 0;
    end
    U = AU(:,1:SVP)*diag(SU)*VU(:,1:SVP)';    
    % ------- Update E ---------- %
    temp1 = X-X*Z+C1/miu;
    temp2 = lambda3/miu;
    E = max(0,temp1-temp2) + min(0,temp1+temp2);
    % ------- Update W -----------%
%     WZ = (Z+Z')/2;
    WZ = (S+S')/2;
%     WZ = S;
    DZ = diag(sum((WZ)));
    LZ = DZ - WZ;
    M = X*LZ*X';
    m = diag(M);
    if min(m) == 0
        m = m+1e-4;
    end
    w = 1/sum(1./m)./m;
    W = diag(w);
    % -------- Update C1 C2 C3 miu -------- %
    L1 = X-X*Z-E;
    L2 = Z-S;
    L3 = Z-U;
    C1 = C1+miu*L1;
    C2 = C2+miu*L2;
    C3 = C3+miu*L3;
    F = Y;
    LL1 = norm(Z-Z_old,'fro');
    LL2 = norm(S-S_old,'fro');
    LL3 = norm(U-U_old,'fro');
    LL4 = norm(E-E_old,'fro');
    SLSL = max(max(max(LL1,LL2),LL3),LL4)/norm(X,'fro');
    if miu*SLSL < tol2
        miu = min(rho*miu,max_miu);
    end
    % --------- obj ---------- %
    leq1 = max(max(abs(L1(:))),max(abs(L2(:))));
    stopC = max(leq1,max(abs(L3(:))));
    if stopC < tol
        iter
        break;
    end   
end
end