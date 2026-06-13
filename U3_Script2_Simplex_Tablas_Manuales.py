# =============================================================================
# LENGUAJE DE PROGRAMACIÓN V  |  UMECIT  |  Unidad III
# Script 2: Simplex Manual — Tablas paso a paso
# =============================================================================
#
# PROBLEMA DEL CURSO:
#   Max Z = 50x₁ + 30x₂
#   s.a.   4x₁ + 2x₂ ≤ 240   (Carpintería)
#          2x₁ + 3x₂ ≤ 180   (Pintura)
#          x₁, x₂   ≥   0
#
# El script implementa el algoritmo desde cero e imprime cada tableau,
# mostrando exactamente el proceso del libro de texto:
#   - Variable entrante: columna con coef más negativo en fila Z
#   - Variable saliente: razón mínima b/aᵢⱼ (aᵢⱼ > 0)
#   - Pivoteo: operaciones de fila elementales
#   - Verificación final contra PuLP
#
# INSTALACIÓN:  pip install numpy pulp
# EJECUCIÓN:    python U3_Script2_Simplex_Tablas_Manuales.py
# =============================================================================

import numpy as np
from fractions import Fraction
from pulp import *

SEP = "=" * 70

print(SEP)
print("  UNIDAD III  |  Simplex Manual con Tablas (Tableau)")
print("  Fábrica de Mesas y Sillas  —  Datos reales del curso")
print(SEP)

# ─────────────────────────────────────────────────────────────────────────────
# DATOS DEL PROBLEMA
# Variables: x1=mesas, x2=sillas, s1=holgura carpintería, s2=holgura pintura
# Columnas del tableau: [x1, x2, s1, s2, b]
# ─────────────────────────────────────────────────────────────────────────────
C_OBJ    = [50.0, 30.0]
N_VARS   = 2                      # x1, x2
N_REST   = 2                      # carpintería, pintura
A        = np.array([[4.0, 2.0],  # carpintería
                     [2.0, 3.0]]) # pintura
B        = np.array([240.0, 180.0])

COL_NAMES = ["x₁", "x₂", "s₁", "s₂", "b"]
VAR_DESC  = {
    "x₁": "mesas/semana",
    "x₂": "sillas/semana",
    "s₁": "holgura carpintería (h)",
    "s₂": "holgura pintura (h)",
}
BASE_INI  = ["s₁", "s₂"]

# ─────────────────────────────────────────────────────────────────────────────
# FUNCIONES
# ─────────────────────────────────────────────────────────────────────────────

def fmt(v):
    """Formatea un número: enteros limpios, fracciones simples o decimal."""
    if abs(v) < 1e-9:
        return "0"
    if abs(v - round(v)) < 1e-6:
        return str(int(round(v)))
    f = Fraction(v).limit_denominator(12)
    if abs(float(f) - v) < 1e-5:
        return str(f)
    return f"{v:.3f}"

def build_tableau():
    """Construye el tableau inicial [A | I | b] con fila Z negada."""
    n = N_VARS + N_REST
    T = np.zeros((N_REST + 1, n + 1))
    T[:N_REST, :N_VARS]              = A
    T[:N_REST, N_VARS:N_VARS+N_REST] = np.eye(N_REST)
    T[:N_REST, -1]                   = B
    T[N_REST, :N_VARS]               = -np.array(C_OBJ)
    return T

def print_tableau(T, base, it):
    """Imprime el tableau con bordes, fracciones y columna de razones."""
    W = 10
    linea = "─" * (W * (len(COL_NAMES) + 1) + 4)
    titulo = f"Tableau — Iteración {it}"
    if it == 0:
        titulo += "  (INICIAL)"
    print(f"\n  ┌── {titulo}")
    print(f"  {linea}")

    # Cabecera
    h = f"  {'Base':>{W}}"
    for c in COL_NAMES:
        h += f"  {c:>{W}}"
    print(h)

    # Calcular razones para columna entrante
    col_e = entrante(T)
    print(f"  {linea}")

    for i in range(N_REST):
        row = f"  {base[i]:>{W}}"
        for val in T[i, :]:
            row += f"  {fmt(val):>{W}}"
        # Razón b/aᵢⱼ
        if col_e is not None:
            aij = T[i, col_e]
            if aij > 1e-9:
                ratio = T[i, -1] / aij
                row += f"  {fmt(ratio):>{W}}"
                if i == saliente(T, col_e):
                    row += "  ← mín"
            else:
                row += f"  {'—':>{W}}"
        print(row)

    print(f"  {linea}")
    zrow = f"  {'Z':>{W}}"
    for val in T[-1, :]:
        zrow += f"  {fmt(val):>{W}}"
    print(zrow)
    print(f"  {linea}")

def entrante(T):
    """Columna con coef más negativo en fila Z. None si todos ≥ 0 (óptimo)."""
    fz  = T[-1, :-1]
    idx = int(np.argmin(fz))
    return idx if fz[idx] < -1e-9 else None

def saliente(T, col):
    """Fila con razón mínima positiva b/aᵢⱼ. None si no acotado."""
    razones = [(T[i,-1]/T[i,col], i)
               for i in range(N_REST) if T[i, col] > 1e-9]
    return min(razones)[1] if razones else None

def pivotar(T, fil, col):
    """Operaciones de fila: normalizar fila pivote y eliminar de las demás."""
    T2 = T.astype(float).copy()
    T2[fil, :] /= T[fil, col]
    for i in range(T2.shape[0]):
        if i != fil:
            T2[i, :] -= T[i, col] * T2[fil, :]
    return T2

# ─────────────────────────────────────────────────────────────────────────────
# EJECUCIÓN DEL ALGORITMO
# ─────────────────────────────────────────────────────────────────────────────
print("\n  Problema: Max Z = 50x₁ + 30x₂")
print("  Forma estándar:")
print("    4x₁ + 2x₂ + s₁       = 240   (Carpintería)")
print("    2x₁ + 3x₂       + s₂ = 180   (Pintura)")
print("    Z − 50x₁ − 30x₂      =   0")
print("\n  Variables básicas iniciales: s₁=240, s₂=180  (x₁=x₂=0, Z=0)")

T    = build_tableau()
base = BASE_INI.copy()

print_tableau(T, base, 0)

for it in range(1, 20):
    col_e = entrante(T)

    if col_e is None:
        print(f"\n  {'─'*60}")
        print(f"  ✓ ÓPTIMO: todos los coefs. en Z ≥ 0  ({it-1} iteración(es))")
        break

    fil_s = saliente(T, col_e)
    if fil_s is None:
        print("  ✗ PROBLEMA NO ACOTADO.")
        break

    var_e = COL_NAMES[col_e]
    var_s = base[fil_s]
    piv   = T[fil_s, col_e]
    razon = T[fil_s, -1] / piv

    print(f"\n  ┌── Iteración {it} — Decisiones:")
    print(f"  │  ENTRA : {var_e}  (coef. en Z = {fmt(T[-1,col_e])}  ← más negativo)")
    print(f"  │  SALE  : {var_s}  (razón = {fmt(razon)}  ← mínima positiva)")
    print(f"  │  PIVOTE: T[{fil_s},{col_e}] = {fmt(piv)}")
    print(f"  │")
    print(f"  │  Operaciones:")
    print(f"  │    R{fil_s+1}_nueva  = R{fil_s+1} ÷ {fmt(piv)}")
    for k in range(N_REST + 1):
        if k != fil_s:
            coef = T[k, col_e]
            if abs(coef) > 1e-9:
                signo = "−" if coef > 0 else "+"
                print(f"  │    R{k+1}_nueva  = R{k+1} {signo} {fmt(abs(coef))}·R{fil_s+1}_nueva")

    T = pivotar(T, fil_s, col_e)
    base[fil_s] = var_e
    print_tableau(T, base, it)

# ─────────────────────────────────────────────────────────────────────────────
# SOLUCIÓN ÓPTIMA
# ─────────────────────────────────────────────────────────────────────────────
print("\n── SOLUCIÓN ÓPTIMA ──────────────────────────────────────")
sol = {v: 0.0 for v in COL_NAMES[:-1]}
for i, vb in enumerate(base):
    sol[vb] = T[i, -1]

print(f"  {'Variable':<8}  {'Valor':>10}  Descripción")
print(f"  {'─'*50}")
for v in COL_NAMES[:-1]:
    desc  = VAR_DESC.get(v, "")
    marca = " ★ básica" if sol[v] > 1e-6 else ""
    print(f"  {v:<8}  {sol[v]:>10.4f}  {desc}{marca}")

print(f"\n  Z* = ${T[-1,-1]:,.2f}  (utilidad máxima semanal)")

# Precios sombra desde la fila Z (columnas de holgura)
print(f"\n  Precios sombra (coefs. de holguras en Z óptimo):")
for i, vn in enumerate(COL_NAMES[N_VARS:-1]):
    ps = T[-1, N_VARS + i]
    nombre_r = ["Carpintería", "Pintura"][i]
    print(f"  {vn} → {nombre_r}: PS = ${ps:.4f}/hora")

# ─────────────────────────────────────────────────────────────────────────────
# VERIFICACIÓN CON PULP
# ─────────────────────────────────────────────────────────────────────────────
print("\n── VERIFICACIÓN CON PULP ────────────────────────────────")
p  = LpProblem("Verificacion", LpMaximize)
v1 = LpVariable("x1", lowBound=0)
v2 = LpVariable("x2", lowBound=0)
p += 50*v1 + 30*v2
p += 4*v1 + 2*v2 <= 240, "Carp"
p += 2*v1 + 3*v2 <= 180, "Pint"
p.solve(PULP_CBC_CMD(msg=0))

z_manual = T[-1, -1]
z_pulp   = value(p.objective)
gap      = abs(z_manual - z_pulp)

print(f"  Z* manual  = ${z_manual:,.4f}")
print(f"  Z* PuLP    = ${z_pulp:,.4f}")
print(f"  Diferencia = {gap:.2e}  {'✓  CORRECTO' if gap < 0.01 else '✗  Revisar'}")

print(f"\n{SEP}")
print("  Script 2 completado.")
print("  Siguiente: U3_Script3_Analisis_Sensibilidad.py")
print(SEP)
