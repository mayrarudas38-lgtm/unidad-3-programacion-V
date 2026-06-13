# =============================================================================
# LENGUAJE DE PROGRAMACIÓN V  |  UMECIT  |  Unidad III
# Script 1: Método Simplex con PuLP — Agroindustrias del Istmo
# =============================================================================
#
# PROBLEMA ADAPTADO:
#   Max Z = 12x₁ + 15x₂
#   s.a.
#     0.8x₁ + 1.0x₂  ≤ 120   (Capacidad de Procesamiento / horas)
#     0.5x₁ + 0.75x₂ ≤ 90    (Capacidad de Empaque / horas)
#     0.4x₁ + 0.6x₂  ≤ 80    (Materia Prima / toneladas)
#     x₁, x₂   ≥  0
#
# SOLUCIÓN ESPERADA:
#   x₁* = 150.00 cajas de piña
#   x₂* = 0.00 cajas de mango
#   Z* = $1,800.00 USD
#   Precio sombra Procesamiento: $15.00/hora
#   Precio sombra Empaque:       $0.00/hora (recurso holgado)
#   Precio sombra Materia Prima: $0.00/tonelada (recurso holgado)
#
# INSTALACIÓN:  pip install pulp
# EJECUCIÓN:    python U3_Script1_Simplex_PuLP.py
# =============================================================================

from pulp import *

SEP  = "=" * 65
SEP2 = "─" * 65

print(SEP)
print("  UNIDAD III  |  Método Simplex con PuLP")
print("  Agroindustrias del Istmo  —  Análisis Primal-Dual")
print(SEP)

# ─────────────────────────────────────────────────────────────────────────────
# SECCIÓN A: DEFINICIÓN DEL MODELO
# ─────────────────────────────────────────────────────────────────────────────
print("\n── A. MODELO ────────────────────────────────────────────")

prob = LpProblem("Agroindustrias_del_Istmo", LpMaximize)

x1 = LpVariable("cajas_pina",  lowBound=0, cat="Continuous")
x2 = LpVariable("cajas_mango", lowBound=0, cat="Continuous")

# Función objetivo
prob += 12*x1 + 15*x2, "Z_Margen_Total_USD"

# Restricciones nombradas (necesario para leer precios sombra)
prob += 0.8*x1 + 1.0*x2  <= 120, "Procesamiento"
prob += 0.5*x1 + 0.75*x2 <= 90,  "Empaque"
prob += 0.4*x1 + 0.6*x2  <= 80,  "Materia_Prima"

print("  Max Z  = 12·cajas_pina + 15·cajas_mango")
print("  s.a.     0.8·cajas_pina + 1.0·cajas_mango  ≤ 120  (Procesamiento, h)")
print("           0.5·cajas_pina + 0.75·cajas_mango ≤ 90   (Empaque, h)")
print("           0.4·cajas_pina + 0.6·cajas_mango  ≤ 80   (Materia Prima, t)")
print("           cajas_pina, cajas_mango ≥ 0")

# ─────────────────────────────────────────────────────────────────────────────
# SECCIÓN B: RESOLUCIÓN
# ─────────────────────────────────────────────────────────────────────────────
print("\n── B. RESOLUCIÓN ────────────────────────────────────────")

prob.solve(PULP_CBC_CMD(msg=0))

print(f"  Estado   : {LpStatus[prob.status]}")
print(f"  Piña     : x₁* = {x1.value():.2f}  cajas")
print(f"  Mango    : x₂* = {x2.value():.2f}  cajas")
print(f"  Utilidad : Z* = $ {value(prob.objective):,.2f} USD")

# Verificar manualmente
z_check = 12 * x1.value() + 15 * x2.value()
print(f"\n  Verificación: 12×{x1.value():.0f} + 15×{x2.value():.0f} = ${z_check:,.2f}  "
      f"{'✓' if abs(z_check - value(prob.objective)) < 0.01 else '✗'}")

# ─────────────────────────────────────────────────────────────────────────────
# SECCIÓN C: PRECIOS SOMBRA (ANÁLISIS DUAL)
# ─────────────────────────────────────────────────────────────────────────────
print("\n── C. PRECIOS SOMBRA ────────────────────────────────────")
print(f"  {'Restricción':<16} {'Recurso usado':>14} {'Límite':>8} "
      f"{'Holgura':>10} {'PS (yᵢ*)':>12}  Estado")
print(f"  {SEP2}")

for nombre, restr in prob.constraints.items():
    limite  = -restr.constant
    uso     = limite + restr.value()
    holgura = limite - uso
    pi      = restr.pi if hasattr(restr, "pi") and restr.pi is not None else 0.0
    estado  = "★ ACTIVA" if abs(holgura) < 1e-6 else f"holgada  (slack={holgura:.1f})"
    print(f"  {nombre:<16} {uso:>14.2f} {limite:>8.0f} "
          f"{holgura:>10.2f} {pi:>12.4f}  {estado}")

# ─────────────────────────────────────────────────────────────────────────────
# SECCIÓN D: VERIFICACIÓN — HOLGURA COMPLEMENTARIA
# Condición de optimalidad: yᵢ* × (bᵢ − aᵢᵀx*) = 0  ∀ i
# ─────────────────────────────────────────────────────────────────────────────
print("\n── D. HOLGURA COMPLEMENTARIA ────────────────────────────")
print("  Condición: yᵢ* × slackᵢ = 0  (debe cumplirse en el óptimo)\n")

todos_ok = True
for nombre, restr in prob.constraints.items():
    limite  = -restr.constant
    uso     = limite + restr.value()
    holgura = limite - uso
    pi      = restr.pi if hasattr(restr, "pi") and restr.pi is not None else 0.0
    prod    = abs(pi * holgura)
    ok      = prod < 1e-5
    if not ok:
        todos_ok = False
    print(f"  {nombre}: y*={pi:.4f} × slack={holgura:.4f} = {prod:.2e}  "
          f"{'✓' if ok else '✗ ERROR'}")

print()
print(f"  {'✓  Holgura Complementaria VERIFICADA' if todos_ok else '✗  Revisar formulación'}")

# ─────────────────────────────────────────────────────────────────────────────
# SECCIÓN E: INTERPRETACIÓN ECONÓMICA
# ─────────────────────────────────────────────────────────────────────────────
print("\n── E. INTERPRETACIÓN ECONÓMICA ──────────────────────────")
print(f"  Plan de producción óptimo:")
print(f"    • Producir {x1.value():.0f} cajas de piña →  ${12*x1.value():,.0f}")
print(f"    • Producir {x2.value():.0f} cajas de mango →  ${15*x2.value():,.0f}")
print(f"    • Utilidad total: Z* = ${value(prob.objective):,.2f} USD")

print()
for nombre, restr in prob.constraints.items():
    limite  = -restr.constant
    uso     = limite + restr.value()
    holgura = limite - uso
    pi      = restr.pi if hasattr(restr, "pi") and restr.pi is not None else 0.0
    
    # Adaptación de unidades para despliegue
    unidad = "t" if nombre == "Materia_Prima" else "h"
    
    if abs(pi) > 1e-6:
        print(f"  ▶ {nombre} (capacidad actual: {limite:.0f} {unidad}):")
        print(f"    Precio sombra = ${pi:.4f} USD/{unidad}")
        print(f"    → Cada unidad extra aumenta Z* en ${pi:.4f}")
        print(f"    → Conviene pagar hasta ${pi:.4f} extra por cada {unidad} adicional")
    else:
        print(f"  ▷ {nombre}: precio sombra = $0  (holgura disponible: {holgura:.1f} {unidad})")
    print()

print(SEP)
print("  Script 1 completado con datos de Agroindustrias del Istmo.")
print("  Siguiente: U3_Script2_Simplex_Tablas_Manuales.py")
print(SEP)