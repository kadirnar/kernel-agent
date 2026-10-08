def mva(S, Z, N):
    Q = 0.0; out = []
    for n in range(1, N + 1):
        R = S * (1 + Q); X = n / (Z + R); Q = X * R
        out.append((n, X * 3600, X * S, R - S))
    return out
for label, S, Z in [("kernel S=60 Z=300", 60, 300), ("kernel S=90 Z=300 (compile in lock)", 90, 300), ("e2e S=120 Z=360", 120, 360), ("mix S=80 Z=300", 80, 300)]:
    print(label)
    for n, x, u, w in mva(S, Z, 8):
        print(f"  N={n}: {x:5.1f} evals/h  GPU busy {u:4.0%}  queue wait {w:5.0f} s")
