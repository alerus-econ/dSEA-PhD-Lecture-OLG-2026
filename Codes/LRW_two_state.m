%==========================================================================
%  LRW (2024) — canonical two-state OLG model with limited commitment
%  Clean single-file MATLAB version (refactor of Generate_policy_2s.m)
%
%  Reads:  nothing (everything self-contained)
%  Writes: canonical_policy.mat in the current folder
%
%  All helper functions live at the bottom of this file as LOCAL FUNCTIONS,
%  so no separate .m files are needed and no globals are used. Run with:
%       >> LRW_two_state
%==========================================================================

function LRW_two_state
clear; close all; clc; tic

% ---------------- 1. Parameters ----------------------------------------
par = parameters();

% ---------------- 2. Diagnostic checks ---------------------------------
diagnostics(par);

% ---------------- 3. Boundary objects of the state space ---------------
[par.v1aut, par.v2aut, par.waut] = autarky(par);
fb                               = first_best(par);
par.cy1fb = fb.cy_fb;  par.cy2fb = fb.cy_fb;
par.wfb   = fb.w_fb;   par.vfb   = fb.V_fb;

par.w_st = max_stationary(par);

fprintf('\n--- Boundary objects ---\n');
fprintf('  ω_aut = %8.4f      (autarky promise)\n', par.waut);
fprintf('  ω_st  = %8.4f      (max stationary promise)\n', par.w_st);
fprintf('  ω_fb  = %8.4f      (first-best promise)\n', par.wfb);
fprintf('  V_fb  = %8.4f      (first-best value)\n', par.vfb);

% ---------------- 4. Grid ----------------------------------------------
par.nw    = 300;
par.wmin  = par.waut;
par.wmax  = par.w_st;
wgrid     = cheby_points(par.wmin, par.wmax, par.nw)';

% ---------------- 5. Howard policy iteration ---------------------------
tol  = 1e-6;
fprintf('\nPredicted iterations to tolerance %.0e: ~%d\n', tol, ...
        round(fzero(@(x) tol^(1/x) - par.delta, 3)));

% Warm start: first-best policies (constant on the grid)
cy1 = par.cy1fb * ones(par.nw, 1);
cy2 = par.cy2fb * ones(par.nw, 1);
w1  = par.wfb   * ones(par.nw, 1);
w2  = par.wfb   * ones(par.nw, 1);
P_old = [cy1, cy2, w1, w2];

dist = 100; old_dist = 1; iter = 0;
fprintf('\n--- Policy iteration ---\n');
while dist > tol
    iter = iter + 1;

    [V, cy1, cy2, w1, w2, w_thr, w_thr_approx] = ...
        value_iteration(cy1, cy2, w1, w2, wgrid, par);

    P    = [cy1, cy2, w1, w2];
    dist = max(max(abs(P - P_old)));
    dist = dist + abs(par.cy1fb - interp1(wgrid, cy1, w_thr, 'spline'));

    fprintf('  iter %3d   distance %.3e   improved by %6.1f%%\n', ...
            iter, dist, (old_dist - dist)/old_dist * 100);

    old_dist = dist;
    P_old    = P;
end

% Upper fixed point: ω = ω_2(ω) (state-2 policy crosses 45° line)
opt    = optimoptions('fsolve', 'Display', 'off');
w_upper = fsolve(@(x) x - interp1(wgrid, w2, x, 'spline'), par.waut, opt);

fprintf('\n--- Convergence ---\n');
fprintf('  ω_0     = %8.4f   (FB threshold)\n', w_thr);
fprintf('  ω_upper = %8.4f   (state-2 fixed point)\n', w_upper);
fprintf('  total time: %.1f s\n', toc);

% ---------------- 6. Plots ---------------------------------------------
make_plots(wgrid, V, cy1, cy2, w1, w2, w_thr, w_upper, par);

% ---------------- 7. Save ----------------------------------------------
out = struct('wgrid', wgrid, 'V', V, ...
             'cy1', cy1, 'cy2', cy2, 'w1', w1, 'w2', w2, ...
             'w_thr', w_thr, 'w_upper', w_upper, 'par', par);
save('canonical_policy.mat', '-struct', 'out');
fprintf('\nSaved: canonical_policy.mat\n');
end


%==========================================================================
%  LOCAL FUNCTIONS (all helpers in one file)
%==========================================================================

% --- Parameters ----------------------------------------------------------
function par = parameters()
par.gamma = 1;                          % risk aversion (γ=1 → log utility)
par.pi1   = 0.5;
par.pi2   = 1 - par.pi1;
par.theta = 0;                          % aggregate shock
par.sigma = 0.1;                        % endowment volatility
par.kappa = 3/5;                        % young share of endowment
par.n     = 0;                          % population growth

% endowments per state and per generation
par.e1  = 1 + par.theta;
par.e2  = 1 - (par.pi1/par.pi2) * par.theta;
par.ey1 = (par.kappa - par.sigma*par.pi2/par.pi1) * par.e1;
par.eo1 = (1 - par.kappa + par.sigma*par.pi2/par.pi1) * par.e1;
par.ey2 = (par.kappa + par.sigma) * par.e2;
par.eo2 = (1 - par.kappa - par.sigma) * par.e2;

% discount factors
par.betta = exp(-1/75);                 % individual
par.delta = par.betta;                  % planner (= β here)
end


% --- Diagnostic checks ---------------------------------------------------
function diagnostics(par)
lam1 = (par.eo1/par.ey1)^par.gamma;
lam2 = (par.eo2/par.ey2)^par.gamma;
fprintf('Calibration sanity checks:\n');
fprintf('  S1  = λ1 - β     = %+.4f   (>0 ⇒ state 1 bad for young)\n', ...
        lam1 - par.betta);
fprintf('  S2  = λ2 - β     = %+.4f   (<0 ⇒ state 2 good for young)\n', ...
        lam2 - par.betta);
fprintf('  Sfb = λ1 - β/δ   = %+.4f   (<0 ⇒ FB has positive transfers)\n', ...
        lam1 - par.betta/par.delta);
fprintf('  No-Samuelson      = %+.4f   (>0 ⇒ dynamic efficiency)\n', ...
        (1-par.kappa) - par.betta*par.kappa);
end


% --- CRRA utility --------------------------------------------------------
function F = u(x, gamma)
if gamma == 1
    F = log(x);
else
    F = (x.^(1-gamma) - 1) ./ (1-gamma);
end
end


% --- Chebyshev nodes -----------------------------------------------------
function x = cheby_points(a, b, n)
k = 1:n;
z = -cos((2*k - 1) * pi / (2*n));
x = (z + 1) * (b - a) / 2 + a;
end


% --- Autarky values ------------------------------------------------------
function [v1aut, v2aut, waut] = autarky(par)
% Each generation eats its own endowment in each life stage. The promise
% state ω^aut is the old's expected utility, seen before the shock realises.
Eu_old = par.pi1 * u(par.eo1, par.gamma) + par.pi2 * u(par.eo2, par.gamma);
v1aut  = u(par.ey1, par.gamma) + par.betta * Eu_old;
v2aut  = u(par.ey2, par.gamma) + par.betta * Eu_old;
waut   = Eu_old;
end


% --- First-best (closed form, log utility) ------------------------------
function fb = first_best(par)
% With log utility the FB young consumption share is closed-form:
%   c_y^fb = δ(1+n) / (β + δ(1+n))
fb.cy_fb = par.delta * (1 + par.n) / (par.betta + par.delta * (1 + par.n));
fb.co_fb = 1 - fb.cy_fb;
fb.V_fb  = (par.pi1 * ((1+par.n)*u(fb.cy_fb,par.gamma) ...
                     + (par.betta/par.delta)*u(fb.co_fb,par.gamma)) ...
          + par.pi2 * ((1+par.n)*u(fb.cy_fb,par.gamma) ...
                     + (par.betta/par.delta)*u(fb.co_fb,par.gamma))) ...
          / (1 - par.delta*(1+par.n));
fb.w_fb  = par.pi1 * u(fb.co_fb, par.gamma) + par.pi2 * u(fb.co_fb, par.gamma);
end


% --- Maximum stationary promise -----------------------------------------
function w_st = max_stationary(par)
% Find transfers (τ1, τ2) that make BOTH state-s autarky ICs bind simultaneously.
% At that point, the promise state hits its upper bound: any larger ω
% would require the planner to violate at least one IC.
[v1aut, v2aut, ~] = autarky(par);
res = @(z) [
    u(par.ey1 - z(1), par.gamma) + par.betta * ( ...
        par.pi1 * u(par.eo1 + z(1), par.gamma) + ...
        par.pi2 * u(par.eo2 + z(2), par.gamma)) - v1aut;
    u(par.ey2 - z(2), par.gamma) + par.betta * ( ...
        par.pi1 * u(par.eo1 + z(1), par.gamma) + ...
        par.pi2 * u(par.eo2 + z(2), par.gamma)) - v2aut];
opt = optimoptions('fsolve', 'Display', 'off');
Z   = fsolve(res, [0.2 0.2], opt);
w_st = par.pi1 * u(par.eo1 + Z(1), par.gamma) ...
     + par.pi2 * u(par.eo2 + Z(2), par.gamma);
end


% --- One value-iteration sweep ------------------------------------------
function [V, cy1, cy2, w1, w2, w_thr, w_thr_approx] = ...
    value_iteration(cy1_old, cy2_old, w1_old, w2_old, wgrid, par)
%
% Two-step Howard improvement at every grid point:
%   (a) VALUATION  — given current policies, evaluate V exactly.
%   (b) IMPROVEMENT — re-optimise (c1,c2,ω1,ω2) at every ω with fmincon.
%
nw     = par.nw;
crit   = 1e-10;
opts_f = optimoptions('fmincon', ...
    'Display', 'off', 'TolX', 1e-8, 'TolFun', crit, 'Algorithm', 'sqp');
opts_s = optimoptions('fsolve', 'Display', 'off');

% (a) VALUATION via the linear system V = (I − δT)⁻¹ flow, then a few
%     Bellman polishes to nail down the fixed point.
PI = zeros(nw);
for k = 1:nw
    PI(k,:) = PI(k,:) + weight(wgrid, w1_old(k), par.pi1);
    PI(k,:) = PI(k,:) + weight(wgrid, w2_old(k), par.pi2);
end
flow = par.pi1 * ((par.betta/par.delta)*u(par.e1 - cy1_old, par.gamma) ...
                + u(cy1_old, par.gamma)) ...
     + par.pi2 * ((par.betta/par.delta)*u(par.e2 - cy2_old, par.gamma) ...
                + u(cy2_old, par.gamma));
V = (eye(nw) - par.delta * PI) \ flow;
dV = 1;
while dV > crit
    Vnew = flow + par.delta * (par.pi1 * interp1(wgrid, V, w1_old, 'spline', V(end)) ...
                             + par.pi2 * interp1(wgrid, V, w2_old, 'spline', V(end)));
    dV = max(abs(Vnew - V));
    V  = Vnew;
end

% Threshold ω₀ — lowest promise where V plateaus (FB region)
w_thr_approx = max(wgrid(abs(V(1) - V) < 1e-8));
w_thr = fsolve(@(x) interp1(wgrid, cy1_old, x, 'spline') - par.cy1fb, ...
               w_thr_approx, opts_s);

% (b) IMPROVEMENT — solve constrained problem at each grid point
lb = [0;          0;         w_thr; w_thr];
ub = [par.ey1; par.ey2; par.wmax; par.wmax];

cy1 = zeros(nw,1); cy2 = zeros(nw,1);
w1  = zeros(nw,1); w2  = zeros(nw,1);
Vimp = zeros(nw,1);

for k = 1:nw
    guess = [cy1_old(k); cy2_old(k); w1_old(k); w2_old(k)];
    omega = wgrid(k);
    obj = @(c) -( ...
            par.pi1 * (u(c(1),par.gamma) + (par.betta/par.delta)*u(par.e1-c(1),par.gamma)) + ...
            par.pi2 * (u(c(2),par.gamma) + (par.betta/par.delta)*u(par.e2-c(2),par.gamma)) + ...
            par.delta * (par.pi1 * interp1(wgrid, V, c(3), 'spline') ...
                       + par.pi2 * interp1(wgrid, V, c(4), 'spline')));
    cons = @(c) deal( ...
        [par.v1aut - u(c(1),par.gamma) - par.betta*c(3);          % IC1: ineq ≤ 0
         omega - (par.pi1*u(par.e1-c(1),par.gamma) ...
                + par.pi2*u(par.e2-c(2),par.gamma))], ...         % PK:  ineq ≤ 0
        par.v2aut - u(c(2),par.gamma) - par.betta*c(4));          % IC2: eq = 0
    [out, fval] = fmincon(obj, guess, [], [], [], [], lb, ub, cons, opts_f);
    cy1(k) = out(1); cy2(k) = out(2);
    w1(k)  = out(3); w2(k)  = out(4);
    Vimp(k) = -fval;
end
V = Vimp;
end


% --- Linear interpolation weights for off-grid points --------------------
function v = weight(wgrid, y, prob)
% Distribute probability `prob` between the two grid points bracketing y.
% Used to build the transition matrix in the valuation step.
v = zeros(1, length(wgrid));
if y >= max(wgrid)
    v(end) = prob;
elseif y <= min(wgrid)
    v(1) = prob;
else
    pos = sum(wgrid <= y);
    a   = (y - wgrid(pos+1)) / (wgrid(pos) - wgrid(pos+1));
    v(pos)   = a * prob;
    v(pos+1) = (1 - a) * prob;
end
end


% --- Plots ---------------------------------------------------------------
function make_plots(wgrid, V, cy1, cy2, w1, w2, w_thr, w_upper, par)

NAVY  = [0.118 0.153 0.380];
RED   = [0.753 0.224 0.169];
BLUE  = [0.180 0.525 0.757];

% Figure 1: promised utility transitions
figure('Name','Promised utilities','Position',[100 100 700 500]);
plot(wgrid, w1, 'Color', RED,  'LineWidth', 2); hold on;
plot(wgrid, w2, 'Color', BLUE, 'LineWidth', 2);
plot(wgrid, wgrid, 'k', 'LineWidth', 0.8);
xline(w_upper, '--k', 'LineWidth', 1);
xline(w_thr,   '--r', 'LineWidth', 1);
xlabel('current promise \omega');
ylabel('next-period promise');
title('Promised utility transitions');
legend({'\omega_1(\omega)', '\omega_2(\omega)', '45°', ...
        sprintf('\\omega_{upper}=%.3f', w_upper), ...
        sprintf('\\omega_0=%.3f', w_thr)}, 'Location', 'best');
grid on;

% Figure 2: young consumption shares
figure('Name','Young consumption','Position',[100 100 700 500]);
plot(wgrid, cy1./par.e1, 'Color', RED,  'LineWidth', 2); hold on;
plot(wgrid, cy2./par.e2, 'Color', BLUE, 'LineWidth', 2);
xlim([w_thr w_upper]);
xlabel('current promise \omega');
ylabel('young consumption share');
title('Young consumption (share of state endowment)');
legend({'c_{y,1}/e_1', 'c_{y,2}/e_2'}, 'Location', 'best');
grid on;

% Figure 3: value function
figure('Name','Value function','Position',[100 100 700 500]);
plot(wgrid, V, 'Color', NAVY, 'LineWidth', 2); hold on;
xline(w_thr, ':r', 'LineWidth', 1);
text(w_thr, V(end) - 0.05*(V(1) - V(end)), '\omega_0', ...
     'FontSize', 14, 'Color', RED, 'HorizontalAlignment', 'right');
xlabel('\omega'); ylabel('V(\omega)');
title('Planner value function');
grid on;

% Figure 4: ex-post promise dynamics (matches MATLAB original)
figure('Name','Ex-post promises','Position',[100 100 1000 450]);
omega1 = u(par.e1 - cy1, par.gamma);
omega2 = u(par.e2 - cy2, par.gamma);
cy11 = interp1(wgrid, cy1, w1, 'spline');
cy12 = interp1(wgrid, cy2, w1, 'spline');
cy21 = interp1(wgrid, cy1, w2, 'spline');
cy22 = interp1(wgrid, cy2, w2, 'spline');
omega11 = u(par.e1 - cy11, par.gamma);
omega12 = u(par.e2 - cy12, par.gamma);
omega21 = u(par.e1 - cy21, par.gamma);
omega22 = u(par.e2 - cy22, par.gamma);

subplot(1,2,1);
plot(omega1, omega11, 'Color', RED,  'LineWidth', 2); hold on;
plot(omega1, omega12, 'Color', BLUE, 'LineWidth', 2);
plot(omega1, omega1, 'k:', 'LineWidth', 0.8);
xlabel('realised \omega'); ylabel('next \omega');
title('Starting from state 1');
legend({'next: state 1', 'next: state 2', '45°'}, 'Location', 'best');
grid on;

subplot(1,2,2);
plot(omega2, omega21, 'Color', RED,  'LineWidth', 2); hold on;
plot(omega2, omega22, 'Color', BLUE, 'LineWidth', 2);
plot(omega2, omega2, 'k:', 'LineWidth', 0.8);
xlabel('realised \omega'); ylabel('next \omega');
title('Starting from state 2');
legend({'next: state 1', 'next: state 2', '45°'}, 'Location', 'best');
grid on;

end
