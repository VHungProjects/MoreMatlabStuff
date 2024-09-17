function p1_00000000
% CISC371, Fall 2024, Practice #1: scalar optimization
% Anonymous functions for objective functions, gradients, and second
% derivatives


f1 =@(t) 5*exp(-2*t) + exp(3*t);          
g1 =@(t) 3*exp(3*t) - 10*exp(-2*t);
h1 =@(t) 20*exp(-2*t)+ 9*exp(3*t);

syms t
f2sym = log(t)^2 - 2 + log(10-t)^2-t^0.2;
g2sym = diff(f2sym);
h2sym = diff(g2sym);
f2 = matlabFunction(f2sym);
g2 = matlabFunction(g2sym);
h2 = matlabFunction(h2sym);

f3sym = -3*t*sin(0.75*t)+exp(-2*t);
g3sym = diff(f3sym);
h3sym = diff(g3sym);
f3 = matlabFunction(f3sym);
g3 = matlabFunction(g3sym);
h3 = matlabFunction(h3sym);
clear t

% Unify the above functions for standard calls to optimization code
fg1  =@(t) deal((f1(t)), (g1(t)));
fgh1 =@(t) deal((f1(t)), (g1(t)), (h1(t)));
fg2  =@(t) deal((f2(t)), (g2(t)));
fgh2 =@(t) deal((f2(t)), (g2(t)), (h2(t)));
fg3  =@(t) deal((f3(t)), (g3(t)));
fgh3 =@(t) deal((f3(t)), (g3(t)), (h3(t)));


% Compute the steepest-descent and backtracking estimates

% %
% % STUDENT CODE GOES HERE
% %
%FUNCTION 1
%Quadratic approx one step
[Func1Quad,~] = funcquadapprox(f1,g1,h1,1)
%tv1 = linspace(-1,1,500);
%plot(tv1, f1(tv1), 'k-',tv1, f1quadapprox(tv1), 'r--', 1, f1(1), 'ro', newtf1, f1(newtf1),'bo')

%Fixed Stepsize
[Func1FixedTmin,~, Func1Fixediterations,~] = steepfixed(fg1,1,(1-0)*(1/100))
%Armijo Backtracking
s0 = (1-0)*(1/10);
beta = 0.5;
[Func1ArmijoTmin,~,Func1Armijoiterations] = steepline(fg1,1,s0,beta)

%Function 2
%quadratic
[Func2Quad,~] = funcquadapprox(f2,g2,h2,9.9)
%Fixed
[Func2FixedTmin,~, Func2Fixediterations,~] = steepfixed(fg2,9.9,(9.9-6)*(1/100))
%Armijo
s0 = (9.9-6)*(1/10);
beta = 0.5;
[Func2ArmijoTmin,~,Func2Armijoiterations] = steepline(fg2,9.9,s0,beta)
%Function 3
%quadratic
[Func3Quad,~] = funcquadapprox(f3,g3,h3,2*pi)
%Fixed
[Func3FixedTmin,~, Func3Fixediterations,~] = steepfixed(fg3,2*pi,(2*pi-0)*(1/100))
%Armijo
s0 = (2*pi-0)*(1/10);
beta = 0.5;
[Func3ArmijoTmin,~,Func3Armijoiterations] = steepline(fg3,2*pi,s0,beta)
end

function [tkplus1, quadapprox]=funcquadapprox(f, g, h, t0)
    t0 = t0;
    quadapprox =@(t) f(t0) + g(t0) * (t - t0) + 0.5 * h(t0) * (t - t0).^2;
    tkplus1 = t0 - g(t0) / h(t0);
end

function [tmin,fmin,ix,wmat]=steepfixed(objgradf,w0,s,imax_in,eps_in)
% [WMIN,FMIN,IX]=STEEPFIXED(OBJGRADF,W0,S,IMAX,F)
% estimates the minimum of function and gradient OBJGRADF, beginning
% at point W0 and using constant stepsize S. Optional
% arguments are the limits on the gradient norm, EPS, and the
% number of iterations, IMAX. 
%
% INPUTS:
%         OBJGRADF - Function for objective and gradient
%         W0       - initial estimate of W
%         S        - stepsize, positive scalar value
%         IMAX     - optional, limit on iterations; default is 50000
%         EPS      - optional, limit on gradient norm; default is 10^-6
% OUTPUTS:
%         TMIN     - minimizer, scalar or vector argument for OBJF
%         FMIN     - scalar value of OBJF at, or near, TMIN
%         IX       - Iterations performed
%         WMAT     - Nx(IX) array, each column is the estimate at each step 

    % Set convergence criteria to those supplied, if available
    if nargin >= 4 & ~isempty(imax_in)
        imax = imax_in;
    else
        imax = 50000;
    end

    if nargin >= 5 & ~isempty(eps_in)
        epsilon = eps_in;
    else
        epsilon = 1e-6;
    end

    % Initialize: search vector, objective, gradient
    tmin = w0; %t = t0
    [fmin gval] = objgradf(tmin); %fcurr = f(t)
    ix = 0;
    wmat = [];
    while (norm(gval)>epsilon & ix<imax)
        % %
        % % STUDENT CODE GOES HERE
        % %
        d = -gval; %dk = -sign(f'(tk))
        tmin = tmin + s*d;  %tk+1 = tk + s0*dk
        [fmin, gval] = objgradf(tmin); %Update values
        ix = ix + 1;
        wmat(:,ix) = tmin;
    end
end

function [tmin,fmin,ix]=steepline(objderivf,t0,s0,beta,imax_in,eps_in)
% [TMIN,FMIN]=STEEPLINE(OBJGRADF,W0,S,BETA,IMAX,EPS)
% estimates the minimum of function and gradient OBJDERIVF, beginning
% at point T0 and using constant stepsize S. Backtracking is
% controlled by backtracking reduction ratio BETA. Optional
% arguments are the limits on the gradient norm, EPS, and the
% number of iterations, IMAX. 
%
% INPUTS:
%         OBJGRADF - Function for objective and gradient
%         T0       - initial estimate of "t"
%         S0       - initial stepsize, positive scalar value
%         BETA     - backtracking hyper-parameter, 0<beta<1; typically 0.5
%         IMAX     - optional, limit on iterations; default is 50000
%         EPS      - optional, limit on gradient norm; default is 10^-6
% OUTPUTS:
%         TMIN     - minimizer, scalar or vector, argument for OBJF
%         FMIN     - scalar value of OBJF at, or near, TMIN
%         IX       - Iterations performed


    % Set convergence criteria to those supplied, if available
    if nargin >= 6 & ~isempty(imax_in)
        imax = imax_in;
    else
        imax = 50000;
    end

    if nargin >= 7 & ~isempty(eps_in)
        epsilon = eps_in;
    else
    epsilon = 1e-6;
    end


    % Limit BETA to the interval (0,1)
    beta  = max(1e-6, min(1-(1e-6), beta));

    % Initialize: objective, gradient, unit search vector
    tmin = t0;
    [fmin dofm] = objderivf(tmin);
    dvec = -dofm';
    alpha = dofm/2;
    ix = 0;

    while (norm(dofm)>epsilon & ix<imax)
        % %
        % % STUDENT CODE GOES HERE
        % %
        s = s0;
        [fest, ~] = objderivf(tmin+s*dvec);
        while fest > (fmin+alpha*s*dvec);
            s = beta*s;
            [fest, ~] = objderivf(tmin+s*dvec);
        end
        tmin = tmin + s*dvec;
        [fmin dofm] = objderivf(tmin);
        dvec = -dofm';
        alpha = dofm/2;
        ix = ix + 1;
    end
end
