# =============================================================================
# LENGUAJE DE PROGRAMACIÓN V  |  UMECIT  |  Unidad III
# Script 2: Simplex Manual — Tablas paso a paso (Agroindustrias del Istmo)
# =============================================================================
#
# PROBLEMA ADAPTADO:
#   Max Z = 12x₁ + 15x₂
#   s.a.   0.8x₁ + 1.0x₂  ≤ 120   (Capacidad de Procesamiento / horas)
#          0.5x₁ + 0.75x₂ ≤ 90    (Capacidad de Empaque / horas)
#          0.4x₁ + 0.6x₂  ≤ 80    (Materia Prima / toneladas)
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

# Línea divisoria dinámica un poco más ancha para las nuevas columnas
SEP = "=" * 80

print(SEP)
print("  UNIDAD III  |  Simplex Manual con Tablas (Tableau) — 3 Restricciones")
print("  Agroindustrias del Istmo  —  Datos reales del problema")
print(SEP)

# ─────────────────────────────────────────────────────────────────────────────
# DATOS DEL PROBLEMA
# Variables: x1=piña, x2=mango, s1=procesamiento, s2=empaque, s3=materia prima
# Columnas del tableau: [x₁, x₂, s₁, s₂, s₃, b]
# ─────────────────────────────────────────────────────────────────────────────
C_OBJ    = [12.0, 15.0]
N_VARS   = 2                      # x1, x2
N_REST   = 3                      # Procesamiento, Empaque, Materia Prima
A        = np.array([[0.8, 1.0],  # Procesamiento
                     [0.5, 0.75], # Empaque
                     [0.4, 0.6]]) # Materia Prima
B        = np.array([120.0, 90.0, 80.0])

COL_NAMES = ["x₁", "x₂", "s₁", "s₂", "s₃", "b"]
VAR_DESC  = {
    "x₁": "cajas de piña",
    "x₂": "cajas de mango",
    "s₁": "holgura procesamiento (h)",
    "s₂": "holgura empaque (h)",
    "s₃": "holgura materia prima (t)",
}
BASE_INI  = ["s₁", "s₂", "s₃"]

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
print("\n  Problema: Max Z = 12x₁ + 15x₂")
print("  Forma estándar con variables de holgura:")
print("    0.8x₁ + 1.0x₂ + s₁           = 120   (Procesamiento)")
print("    0.5x₁ + 0.75x₂     + s₂      = 90    (Empaque)")
print("    0.4x₁ + 0.6x₂           + s₃ = 80    (Materia Prima)")
print("    Z − 12x₁ − 15x₂              =  0")
print("\n  Variables básicas iniciales: s₁=120, s₂=90, s₃=80  (x₁=x₂=0, Z=0)")

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
print(f"  {'─'*55}")
for v in COL_NAMES[:-1]:
    desc  = VAR_DESC.get(v, "")
    marca = " ★ básica" if sol[v] > 1e-6 else ""
    print(f"  {v:<8}  {sol[v]:>10.4f}  {desc}{marca}")

print(f"\n  Z* = ${T[-1,-1]:,.2f} USD  (Margen de utilidad máximo)")

# Precios sombra desde la fila Z (columnas de holgura s1, s2, s3)
print(f"\n  Precios sombra (coefs. de holguras en Z óptimo):")
nombres_restr = ["Procesamiento", "Empaque", "Materia_Prima"]
unidades      = ["hora", "hora", "tonelada"]

for i, vn in enumerate(COL_NAMES[N_VARS:-1]):
    ps = T[-1, N_VARS + i]
    print(f"  {vn} → {nombres_restr[i]}: PS = ${ps:.4f} USD/{unidades[i]}")

# ─────────────────────────────────────────────────────────────────────────────
# VERIFICACIÓN CON PULP
# ─────────────────────────────────────────────────────────────────────────────
print("\n── VERIFICACIÓN CON PULP ────────────────────────────────")
p  = LpProblem("Verificacion", LpMaximize)
v1 = LpVariable("x1", lowBound=0)
v2 = LpVariable("x2", lowBound=0)
p += 12*v1 + 15*v2
p += 0.8*v1 + 1.0*v2  <= 120, "Proc"
p += 0.5*v1 + 0.75*v2 <= 90,  "Emp"
p += 0.4*v1 + 0.6*v2  <= 80,  "MatP"
p.solve(PULP_CBC_CMD(msg=0))

z_manual = T[-1, -1]
z_pulp   = value(p.objective)
gap      = abs(z_manual - z_pulp)

print(f"  Z* manual  = ${z_manual:,.4f}")
print(f"  Z* PuLP    = ${z_pulp:,.4f}")
print(f"  Diferencia = {gap:.2e}  {'✓  CORRECTO' if gap < 0.01 else '✗  Revisar'}")

print(f"\n{SEP}")
print("  Script 2 completado con datos de Agroindustrias del Istmo.")
print("  Siguiente: U3_Script3_Analisis_Sensibilidad.py")
print(SEP)