import numpy as np
# ========= Pretty-print =========
def print_matrix(M, title=None, fmt="{:10.6f}"):
    if title:
        print(title)
    for row in M:
        print(" ".join(fmt.format(x) for x in row))
    print()

def print_interval_matrix(A0, eps, fmt="[{:.6f},{:.6f}]", mode="tomo"):
    """
    mode="tomo": radA = eps * [[1,1],[1,1],...]
    mode="regress": radA = eps * [[1,0],[1,0],...]
    """
    for row in A0:
        a, b = row
        if mode == "tomo":
            print(fmt.format(a - eps, a + eps), fmt.format(b - eps, b + eps))
        elif mode == "regress":
            print(fmt.format(a - eps, a + eps), fmt.format(b, b))
        else:
            raise ValueError("Unknown mode")
    print()

def debug_step(A0, eps, eps_range, step_id, phase="", mode="tomo"):
    print(f"step = {step_id}" + (f" ({phase})" if phase else ""))
    if eps_range is not None:
        l, r = eps_range
        print(f"interval epsilon = [{l:.6f}, {r:.6f}]")
    else:
        print("epsilon = (Not defined)")
    print(f"epsilon = {eps:.6f}")
    print("A(eps):")
    print_interval_matrix(A0, eps, mode=mode)

# ====== Interval helpers & collinearity parts
def interval_div_positive(u_min, u_max, v_min, v_max):
    assert v_min > 0, "Знаменатель должна быть положительным."
    cands = [u_min / v_min, u_min / v_max, u_max / v_min, u_max / v_max]
    return min(cands), max(cands)

def interval_scale(lam, v_min, v_max):
    a = lam * v_min; b = lam * v_max
    return (min(a, b), max(a, b))

def interval_intersection(a_min, a_max, b_min, b_max, tol=0.0):
    lo = max(a_min, b_min); hi = min(a_max, b_max)
    if lo <= hi + tol: return (lo, hi)
    return None

# ---- NEW: helper to get (U,V) per-row depending on mode
def row_intervals(a_i, b_i, eps, mode):
    """
    Returns:
      U = interval for column 1 (a_i)
      V = interval for column 2 (b_i)
    """
    if mode == "tomo":
        # both columns vary by ±eps
        U = (a_i - eps, a_i + eps)
        V = (b_i - eps, b_i + eps)
    elif mode == "regress":
        # only first column varies; second column fixed
        U = (a_i - eps, a_i + eps)
        V = (b_i, b_i)
    else:
        raise ValueError("Unknown mode")
    return U, V

def lambda_interval_for_eps(A0, eps, require_pos_den=True, verbose=False, mode="tomo"):
    a = A0[:, 0].astype(float)
    b = A0[:, 1].astype(float)
    # quick positivity check for denominators in "regress" (degenerate V)
    if require_pos_den and mode == "regress":
        # V = [b_i, b_i] => b_i > 0 with each i
        if np.any(b <= 0):
            return None, []
    lam_l, lam_r = -np.inf, np.inf
    steps = []
    for i in range(len(a)):
        U, V = row_intervals(a[i], b[i], eps, mode)
        u_min, u_max = U
        v_min, v_max = V
        if require_pos_den:
            # if V have 0 fail
            if v_min <= 0:
                # mode="tomo" v_min>0 ⇒ (b_i - eps) > 0
                # mode="regress" v_min=v_max=b_i > 0
                return None, steps
            lo, hi = interval_div_positive(u_min, u_max, v_min, v_max)
        else:
            if v_min <= 0 <= v_max:
                return None, steps
            lo, hi = interval_div_positive(u_min, u_max, v_min, v_max)
        lam_l = max(lam_l, lo); lam_r = min(lam_r, hi)
        steps.append({"row": i, "U": (u_min, u_max), "V": (v_min, v_max),
                      "U_div_V": (lo, hi), "lam_running": (lam_l, lam_r)})
        if lam_l > lam_r:
            return None, steps
    return (lam_l, lam_r), steps

def construct_degenerate_matrix(A0, eps, lam, tol=1e-12, verbose=False, mode="tomo"):
    a = A0[:, 0].astype(float); b = A0[:, 1].astype(float)
    m = len(a); u = np.zeros(m); v = np.zeros(m)
    for i in range(m):
        U, V = row_intervals(a[i], b[i], eps, mode)
        WV = interval_scale(lam, V[0], V[1]) # lam * V interval
        I = interval_intersection(U[0], U[1], WV[0], WV[1], tol=1e-12)
        if I is None: return None
        u_i = 0.5 * (I[0] + I[1])
        if abs(lam) > tol:
            v_i = u_i / lam
        else:
            v_i = 0.0
        # clamp v_i into V interval (even if V is degenerate)
        v_i = min(max(v_i, V[0]), V[1])
        u[i], v[i] = u_i, v_i
    return np.column_stack([u, v])

def cols_are_collinear(A0, eps, verbose=False, mode="tomo"):
    lam_rng, steps = lambda_interval_for_eps(A0, eps, require_pos_den=True, verbose=verbose, mode=mode)
    if lam_rng is None: return False, None, steps
    return True, lam_rng, steps

def epsilon_star_3x2(A0, delta=1e-6, eps_init=0.05, verbose=True, mode="tomo"):  # Убрал eps_max, улучшил delta
    """
    mode="tomo": radA = eps * [[1,1],[1,1],...]
    mode="regress": radA = eps * [[1,0],[1,0],...]
    Находит минимальный δ (eps*), где столбцы могут быть коллинеарными (det(A^T A) может быть 0).
    """
    step_id = 0
    # Check eps=0
    ok0, lam_rng0, _ = cols_are_collinear(A0, 0.0, verbose=verbose, mode=mode)
    if ok0:
        if verbose:
            debug_step(A0, 0.0, (0.0, 0.0), step_id, phase="initial", mode=mode)
        lam_star = lam_rng0[0] if lam_rng0 else 1.0
        M0 = construct_degenerate_matrix(A0, 0.0, lam_star, verbose=verbose, mode=mode)
        return 0.0, M0, lam_star
    # expand right bound
    left, right = 0.0, eps_init
    ok, lam_rng, _ = cols_are_collinear(A0, right, verbose=verbose, mode=mode)
    while not ok:
        left = right
        right *= 2
        ok, lam_rng, _ = cols_are_collinear(A0, right, verbose=verbose, mode=mode)
        if verbose:
            step_id += 1
            debug_step(A0, right, (left, right), step_id, phase="expand", mode=mode)
    # bisection
    while right - left > delta:
        mid = (left + right) / 2.0
        ok, lam_rng, _ = cols_are_collinear(A0, mid, verbose=verbose, mode=mode)
        if ok:
            right = mid
        else:
            left = mid
        if verbose:
            step_id += 1
            debug_step(A0, mid, (left, right), step_id, phase="bisection", mode=mode)
    lam_rng_final, _ = lambda_interval_for_eps(A0, right, verbose=verbose, mode=mode)
    lam_star = lam_rng_final[0]
    M_star = construct_degenerate_matrix(A0, right, lam_star, verbose=verbose, mode=mode)
    if M_star is None and lam_rng_final is not None:
        lam_star = lam_rng_final[1]
        M_star = construct_degenerate_matrix(A0, right, lam_star, verbose=verbose, mode=mode)
    return right, M_star, lam_star

# ===== Demo =====
if __name__ == "__main__":
    A0 = np.array([
        [0.95, 1.00],
        [1.05, 1.00],
        [1.10, 1.00]
    ], dtype=float) 
    print("=== TOMO (томография) ===")
    delta_star, M_star, lam_star = epsilon_star_3x2(A0, delta=1e-6, eps_init=0.05, verbose=True, mode="tomo")
    if delta_star is None:
        print("Не найден δ.")
    else:
        print("\n1. Диапазон δ, при которых det A > 0: [0, {:.6f})".format(delta_star))
        print("2. Минимальное δ ≈ {:.6f}, λ* ≈ {:.6f}".format(delta_star, lam_star))
        print("Интервальная матрица A при min δ:")
        print_interval_matrix(A0, delta_star, mode="tomo")
        if M_star is not None:
            print_matrix(M_star, title="Точечная матрица A' (tomo):")
            print("Проверка: det((A')^T A') = {:.6e}".format(np.linalg.det(M_star.T @ M_star)))
    print("\n=== REGRESS (линейная регрессия) ===")
    delta_star_r, M_star_r, lam_star_r = epsilon_star_3x2(A0, delta=1e-6, eps_init=0.05, verbose=True, mode="regress")
    if delta_star_r is None:
        print("Не найден δ (regress).")
    else:
        print("\n1. Диапазон δ, при которых det A > 0: [0, {:.6f})".format(delta_star_r))
        print("2. Минимальное δ ≈ {:.6f}, λ* ≈ {:.6f}".format(delta_star_r, lam_star_r))
        print("Интервальная матрица A при min δ:")
        print_interval_matrix(A0, delta_star_r, mode="regress")
        if M_star_r is not None:
            print_matrix(M_star_r, title="Точечная матрица A' (regress):")
            print("Проверка: det((A')^T A') = {:.6e}".format(np.linalg.det(M_star_r.T @ M_star_r)))
    