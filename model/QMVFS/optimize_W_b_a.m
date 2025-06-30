function [alpha, beta, W, obj_values] = optimize_W_b_a(X, K, lambda, p, gamma, max_iter, eta, tol)
% Inputs:
%   X       - cell array£¬X{v} represents the data of the v-th view, with a size of n x d_v.
%   K       - cell array£¬K{v}{i} represents the i-th kernel matrix of the v-th view, with a size of n x n.
%   lambda  - regularization parameter
%   p       - view weight exponent parameter
%   gamma   - entropy regularization parameter
%   max_iter - maximum number of iterations
%   tol     - convergence threshold 
%
% Outputs:
%   alpha   - V x 1 view weight vector
%   beta    - k x V kernel combination weight
%   W       - cell array£¬W{v} represents the self-representation coefficient matrix of the v-th view
%   obj_values - The sequence of objective function values for each iteration

[V, k] = size(K);      % number of kernels per view
n = size(X{1}, 1);     % sample size

% initialization
alpha = ones(V, 1) / V;
beta = repmat(ones(k, 1) / k, 1, V);  % each column corresponds to one beta_v
W = cell(V, 1);
obj_values = zeros(max_iter, 1);
obj_prev = Inf;

for v = 1:V
    d = size(X{v}, 2);
    W{v} = randn(d, n);
end

for iter = 1:max_iter
    %% Step 1: update W_v
    for v = 1:V
        Kv_comb = zeros(n);
        for i = 1:k
            Kv_comb = Kv_comb + beta(i, v) * K{v, i};
        end
        Xv = X{v};
        Wv = W{v};
        row_norms = sqrt(sum(Wv.^2, 2)) + 1e-6;
        D = diag(0.5 ./ row_norms);
        A = Xv' * Xv + (lambda / alpha(v)^p) * D;
        B = Xv' * Kv_comb;
        W{v} = A \ B;
    end

    %% Step 2: update beta_v
    for v = 1:V
        T = X{v} * W{v};
        grad = zeros(k, 1);
        for i = 1:k
            Ki = K{v, i};
            trace_KTi = trace(T' * Ki);
            for j = 1:k
                Kj = K{v, j};
                grad(i) = grad(i) + 2 * beta(j, v) * trace(Kj' * Ki);
            end
            grad(i) = grad(i) - 2 * trace_KTi - gamma * (1 + log(beta(i, v) + 1e-12));
        end
        % gradient step
        % eta = 1e-6;
        beta_tilde = beta(:,v) - eta * grad;
        beta(:, v) = project_simplex(beta_tilde);
    end

    %% Step 3: update alpha
    E = zeros(V, 1);
    for v = 1:V
        Kv_comb = zeros(n);
        for i = 1:k
            Kv_comb = Kv_comb + beta(i, v) * K{v, i};
        end
        E(v) = norm(Kv_comb - X{v} * W{v}, 'fro')^2;
    end
    alpha = (E .^ (1 / (1 - p)))';
    alpha = alpha / sum(alpha); 

    %% Step 4: Objective function value + Convergence judgment
    obj_now = 0;
    for v = 1:V
        Kv_comb = zeros(n);
        for i = 1:k
            Kv_comb = Kv_comb + beta(i, v) * K{v, i};
        end
        term1 = norm(Kv_comb - X{v} * W{v}, 'fro')^2;
        term2 = sum(sqrt(sum(W{v}.^2, 2)));
        ent = sum(beta(:, v) .* log(beta(:, v) + 1e-12));
        obj_now = obj_now + alpha(v)^p * term1 + lambda * term2 - gamma * ent;
    end
    obj_values(iter) = obj_now;

    if abs(obj_prev - obj_now)/obj_prev < tol
        fprintf('Converged to the %dth iteration, the objective function changed to %.6f\n', iter, abs(obj_now - obj_prev));
        obj_values = obj_values(1:iter);
        break;
    end
    obj_prev = obj_now;
end
end

function proj = project_simplex(y)
% Project the vector y onto the probability simplex
k = length(y);
u = sort(y, 'descend');
cssv = cumsum(u);
rho = find(u + (1 - cssv) ./ (1:k)' > 0, 1, 'last');
theta = (cssv(rho) - 1) / rho;
proj = max(y - theta, 0);
end