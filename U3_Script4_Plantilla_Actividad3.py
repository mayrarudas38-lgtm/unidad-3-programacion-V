# =============================================================================
# LENGUAJE DE PROGRAMACIÓN V  |  UMECIT  |  Unidad III
# Script 4: PLANTILLA — Actividad N.° 3
#           Simplex + Análisis de Sensibilidad + Gráficas
# =============================================================================
#
# INSTRUCCIONES PARA EL EQUIPO:
#   1. Lee la sección "CONFIGURA TU PROBLEMA" y reemplaza los valores
#      con los datos de tu problema real.
#   2. Ejecuta: python U3_Script4_Plantilla_Actividad3.py
#   3. Comprueba que la solución es correcta.
#   4. Sube los PNG generados a tu presentación interactiva.
#   5. Sube este archivo a GitHub o Google Colab y genera el QR con:
#      https://qr-code-generator.com/
#
# El ejemplo pre-cargado usa los datos del curso (Fábrica de Muebles)
# para que puedas verificar que tu instalación funciona antes de
# adaptar el script a tu propio problema.
#
# INSTALACIÓN:  pip install pulp scipy matplotlib numpy
# EJECUCIÓN:    python U3_Script4_Plantilla_Actividad3.py
# =============================================================================

from pulp import *
from scipy.optimize import linprog
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# ╔══════════════════════════════════════════════════════════════════════╗
# ║  CONFIGURA TU PROBLEMA AQUÍ  (edita SOLO esta sección)              ║
# ╠══════════════════════════════════════════════════════════════════════╣
# ║  Ejemplo pre-cargado: Fábrica de Mesas y Sillas (datos del curso)   ║
# ║  Para tu problema: reemplaza los valores marcados con ←             ║
# ╚══════════════════════════════════════════════════════════════════════╝

# Nombre de tu empresa o problema
NOMBRE = "Fabrica_Mesas_y_Sillas"          # ← Cambia aquí

# Coeficientes de la función objetivo (utilidad por unidad)
C_OBJ  = [50.0, 30.0]                      # ← [c₁, c₂]

# Matriz de coeficientes de las restricciones
A_UB   = [
    [4.0, 2.0],                             # ← Restricción 1
    [2.0, 3.0],                             # ← Restricción 2
]

# Lado derecho de cada restricción (RHS)
B_UB   = [240.0, 180.0]                    # ← [b₁, b₂]

# Nombres descriptivos (para tablas y gráficas)
NOMBRE_R = ["Carpintería (h/sem)",          # ← nombre restricción 1
            "Pintura (h/sem)"]              # ← nombre restricción 2

NOMBRE_X = ["Mesas",  "Sillas"]            # ← nombre variable 1 y 2
UNIDAD_X = ["u/sem",  "u/sem"]             # ← unidad de medida

# ── Configuración del análisis de sensibilidad ────────────────────────
# Índice de la restricción a analizar en el RHS (0 = R1, 1 = R2)
IDX_RHS    = 0                             # ← 0 = analizar b₁

# Rango de variación del RHS seleccionado
DELTA_RHS  = np.arange(-120, 181, 20)      # ← ajusta según tu problema

# Rango de variación de c₁ para la sensibilidad de coeficientes
C1_RANGO   = np.arange(5, 155, 5)          # ← ajusta según tu problema

# =============================================================================
# A PARTIR DE AQUÍ EL SCRIPT ES AUTOMÁTICO
# =============================================================================
SEP  = "=" * 70
SEP2 = "─" * 70

print(SEP)
print(f"  Actividad N.° 3  |  {NOMBRE}")
print(f"  Max Z = {' + '.join(f'${c:.0f}·{n}' for c,n in zip(C_OBJ,NOMBRE_X))}")
print(SEP)

c_neg  = [-c for c in C_OBJ]
BOUNDS = [(0, None)] * len(C_OBJ)

# ── SOLUCIÓN ÓPTIMA con PuLP ──────────────────────────────────────────
print("\n── SOLUCIÓN ÓPTIMA (PuLP) ───────────────────────────────")

prob = LpProblem(NOMBRE, LpMaximize)
vars_lp = [LpVariable(f"x{i+1}", lowBound=0) for i in range(len(C_OBJ))]

prob += sum(C_OBJ[i] * vars_lp[i] for i in range(len(C_OBJ))), "FO"
for j, (row, b) in enumerate(zip(A_UB, B_UB)):
    tag = NOMBRE_R[j].replace(" ","_").replace("(","").replace(")","").replace("/","")
    prob += sum(row[k] * vars_lp[k] for k in range(len(C_OBJ))) <= b, tag

prob.solve(PULP_CBC_CMD(msg=0))

x_opt = [v.value() for v in vars_lp]
z_opt = value(prob.objective)

print(f"  Estado: {LpStatus[prob.status]}")
for i, (xi, ni, ui) in enumerate(zip(x_opt, NOMBRE_X, UNIDAD_X)):
    print(f"  x{i+1}* ({ni}) = {xi:.4f} {ui}")
print(f"  Z*     = $ {z_opt:,.4f}")

# ── PRECIOS SOMBRA ────────────────────────────────────────────────────
print("\n── PRECIOS SOMBRA ───────────────────────────────────────")
print(f"  {'Restricción':<24} {'Uso':>8} {'Límite':>8} "
      f"{'PS (yᵢ*)':>12}  Estado")
print(f"  {SEP2[:55]}")

pis = {}
for nombre_c, restr in prob.constraints.items():
    limite  = -restr.constant
    uso     = limite + restr.value()
    holgura = limite - uso
    pi      = restr.pi if hasattr(restr,"pi") and restr.pi else 0.0
    pis[nombre_c] = pi
    estado  = "★ ACTIVA" if abs(holgura) < 0.01 else f"holgada (slack={holgura:.1f})"
    print(f"  {nombre_c:<24} {uso:>8.2f} {limite:>8.0f} {pi:>12.4f}  {estado}")
    if abs(pi) > 0.001:
        print(f"    → Cada unidad adicional de '{nombre_c}' aumenta Z* en ${pi:.4f}")
print()

# ── HOLGURA COMPLEMENTARIA ────────────────────────────────────────────
print("── HOLGURA COMPLEMENTARIA ───────────────────────────────")
print("  yᵢ* × slackᵢ = 0 (condición de optimalidad)\n")
todos_ok = True
for nombre_c, restr in prob.constraints.items():
    limite  = -restr.constant
    uso     = limite + restr.value()
    holgura = limite - uso
    pi      = pis.get(nombre_c, 0.0)
    prod    = abs(pi * holgura)
    ok      = prod < 1e-5
    if not ok:
        todos_ok = False
    print(f"  {nombre_c}: y*={pi:.4f} × slack={holgura:.4f} "
          f"= {prod:.2e}  {'✓' if ok else '✗'}")
print(f"\n  {'✓ Verificado' if todos_ok else '✗ Revisar formulación'}")

# ── SENSIBILIDAD DEL RHS ──────────────────────────────────────────────
b_base = B_UB[IDX_RHS]
print(f"\n── SENSIBILIDAD RHS: {NOMBRE_R[IDX_RHS]} (actual={b_base:.0f}) ───")
print(f"  {'Valor':>8}  {'Δ':>6}  " +
      "  ".join(f"{'x'+str(i+1)+'*':>7}" for i in range(len(C_OBJ))) +
      f"  {'Z*':>10}  {'ΔZ*':>9}")
print(f"  {SEP2[:65]}")

b_list, z_rhs = [], []
for delta in DELTA_RHS:
    b_n = B_UB.copy()
    b_n[IDX_RHS] = b_base + delta
    r = linprog(c_neg, A_ub=A_UB, b_ub=b_n, bounds=BOUNDS, method="highs")
    if r.success:
        z_n = -r.fun
        dz  = z_n - z_opt
        marca = "  ← BASE" if delta == 0 else ""
        print(f"  {b_base+delta:>8.0f}  {delta:>+6}  " +
              "  ".join(f"{r.x[i]:>7.1f}" for i in range(len(C_OBJ))) +
              f"  {z_n:>10,.2f}  {dz:>+9,.2f}{marca}")
        b_list.append(b_base + delta)
        z_rhs.append(z_n)

# ── SENSIBILIDAD DE c₁ ────────────────────────────────────────────────
print(f"\n── SENSIBILIDAD c₁: {NOMBRE_X[0]} (actual=${C_OBJ[0]:.0f}) ─────────")
print(f"  {'c₁($/u)':>8}  " +
      "  ".join(f"{'x'+str(i+1)+'*':>7}" for i in range(len(C_OBJ))) +
      f"  {'Z*':>10}  Observación")
print(f"  {SEP2[:65]}")

c1_list, z_c1, x1_c1, x2_c1 = [], [], [], []
x_prev = x_opt.copy()
for c1 in C1_RANGO:
    c_n = [c1] + C_OBJ[1:]
    r   = linprog([-c for c in c_n], A_ub=A_UB, b_ub=B_UB, bounds=BOUNDS, method="highs")
    if r.success:
        z_n    = -r.fun
        cambio = " ← CAMBIA PLAN" if any(abs(r.x[i]-x_prev[i])>0.5 for i in range(len(x_prev))) else ""
        marca  = "  ← ACTUAL"     if abs(c1 - C_OBJ[0]) < 3 else ""
        print(f"  {c1:>8.0f}  " +
              "  ".join(f"{r.x[i]:>7.1f}" for i in range(len(C_OBJ))) +
              f"  {z_n:>10,.2f}{cambio}{marca}")
        c1_list.append(c1); z_c1.append(z_n)
        x1_c1.append(r.x[0]); x2_c1.append(r.x[1])
        x_prev = list(r.x)

# ── GRÁFICAS ──────────────────────────────────────────────────────────
print(f"\n── GENERANDO GRÁFICAS ───────────────────────────────────")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(
    f"Análisis de Sensibilidad — {NOMBRE}\n"
    f"Max Z = {' + '.join(f'${c:.0f}·{n}' for c,n in zip(C_OBJ,NOMBRE_X))}",
    fontsize=12, fontweight="bold", color="#1F3864"
)

# Panel izquierdo: Z* vs RHS seleccionado
ax1 = axes[0]
ax1.set_facecolor("#F7F9FC")
ax1.plot(b_list, z_rhs, color="#1F3864", lw=2.5, marker="o", ms=6)
ax1.axvline(b_base, color="#C9A227", lw=2.2, ls="--",
            label=f"{NOMBRE_R[IDX_RHS]}  actual={b_base:.0f}")
ax1.axhline(z_opt, color="#1E7145", lw=1.5, ls=":",
            label=f"Z* base = ${z_opt:,.0f}")
ax1.set_xlabel(NOMBRE_R[IDX_RHS], fontsize=11, color="#1F3864")
ax1.set_ylabel("Z* ($)",           fontsize=11, color="#1F3864")
ax1.set_title(f"Z* vs {NOMBRE_R[IDX_RHS]}",
              fontsize=11, fontweight="bold", color="#1F3864")
ax1.legend(fontsize=9)
ax1.grid(alpha=0.3, ls="--")
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

# Panel derecho: Z* y plan vs c₁
ax2  = axes[1]
ax2r = ax2.twinx()
ax2.set_facecolor("#F7F9FC")
ax2.plot(c1_list, z_c1, color="#2E75B6", lw=2.5, label="Z*")
ax2r.plot(c1_list, x1_c1, color="#C00000", lw=2, ls="--",
          label=f"{NOMBRE_X[0]}*", alpha=0.85)
ax2r.plot(c1_list, x2_c1, color="#1E7145", lw=2, ls=":",
          label=f"{NOMBRE_X[1]}*", alpha=0.85)
ax2.axvline(C_OBJ[0], color="#C9A227", lw=2.2, ls="--",
            label=f"c₁ actual = ${C_OBJ[0]:.0f}")
ax2.set_xlabel(f"c₁ — Utilidad {NOMBRE_X[0]} ($/u)",
               fontsize=11, color="#1F3864")
ax2.set_ylabel("Z* ($)",             fontsize=11, color="#1F3864")
ax2r.set_ylabel("Unidades óptimas",  fontsize=10, color="#C00000")
ax2.set_title(f"Z* y Plan vs c₁ ({NOMBRE_X[0]})",
              fontsize=11, fontweight="bold", color="#1F3864")
lns = ax2.get_legend_handles_labels()[0] + ax2r.get_legend_handles_labels()[0]
lbs = ax2.get_legend_handles_labels()[1] + ax2r.get_legend_handles_labels()[1]
ax2.legend(lns, lbs, fontsize=8, loc="upper left")
ax2.grid(alpha=0.3, ls="--")
ax2.spines["top"].set_visible(False)

plt.tight_layout()
img = f"U3_{NOMBRE}_Sensibilidad.png"
plt.savefig(img, dpi=150, bbox_inches="tight")
plt.show()
print(f"  Gráfica guardada: {img}")

print(f"\n{SEP}")
print(f"  Script 4 completado — {NOMBRE}")
print()
print("  PARA ADAPTAR A TU PROBLEMA:")
print("  ① Cambia NOMBRE, C_OBJ, A_UB, B_UB, NOMBRE_R, NOMBRE_X")
print("  ② Ajusta IDX_RHS y los rangos DELTA_RHS / C1_RANGO")
print("  ③ Sube el .py a GitHub/Colab → genera el QR para la actividad")
print(SEP)
