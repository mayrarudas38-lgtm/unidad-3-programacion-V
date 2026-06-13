"""
Actividad 2: Análisis Primal-Dual – Caso Agroindustrias del Istmo
Estudiante: [Tus datos / Equipo]
Fecha: 2026
"""

# Importar librería PuLP
from pulp import *

# ==============================================
# PASO 1: Resolución del MODELO PRIMAL
# ==============================================
# Crear problema de maximización
primal = LpProblem("Primal_Produccion_Frutas", LpMaximize)

# Variables de decisión
x1 = LpVariable("Cajas_Pina", lowBound=0, cat='Continuous')
x2 = LpVariable("Cajas_Mango", lowBound=0, cat='Continuous')

# Función objetivo
primal += 12*x1 + 15*x2, "Margen_Total"

# Restricciones
primal += 0.8*x1 + 1.0*x2 <= 120, "Capacidad_Procesamiento"
primal += 0.5*x1 + 0.75*x2 <= 90,  "Capacidad_Empaque"
primal += 0.4*x1 + 0.6*x2 <= 80,   "Materia_Prima"

# Resolver
primal.solve(PULP_CBC_CMD(msg=False))

print("===== SOLUCIÓN MODELO PRIMAL =====")
print(f"Estado: {LpStatus[primal.status]}")
print(f"x1 = {x1.value():.2f} cajas de piña")
print(f"x2 = {x2.value():.2f} cajas de mango")
print(f"Valor óptimo Z* = {value(primal.objective):.2f} USD\n")

# Extraer precios sombra (valores duales)
precios_sombra = [
    primal.constraints["Capacidad_Procesamiento"].pi,
    primal.constraints["Capacidad_Empaque"].pi,
    primal.constraints["Materia_Prima"].pi
]
print("===== PRECIOS SOMBRA =====")
print(f"y1* (Procesamiento): {precios_sombra[0]:.2f} USD/h")
print(f"y2* (Empaque): {precios_sombra[1]:.2f} USD/h")
print(f"y3* (Materia prima): {precios_sombra[2]:.2f} USD/t\n")

# ==============================================
# PASO 2: Resolución del MODELO DUAL
# ==============================================
dual = LpProblem("Dual_Recursos", LpMinimize)

# Variables duales
y1 = LpVariable("Precio_Procesamiento", lowBound=0)
y2 = LpVariable("Precio_Empaque", lowBound=0)
y3 = LpVariable("Precio_MateriaPrima", lowBound=0)

# Función objetivo
dual += 120*y1 + 90*y2 + 80*y3, "Costo_Total_Recursos"

# Restricciones
dual += 0.8*y1 + 0.5*y2 + 0.4*y3 >= 12, "Demanda_Pina"
dual += 1.0*y1 + 0.75*y2 + 0.6*y3 >= 15, "Demanda_Mango"

# Resolver
dual.solve(PULP_CBC_CMD(msg=False))

print("\n===== SOLUCIÓN MODELO DUAL =====")
print(f"Estado: {LpStatus[dual.status]}")
print(f"y1 = {y1.value():.2f}")
print(f"y2 = {y2.value():.2f}")
print(f"y3 = {y3.value():.2f}")
print(f"Valor óptimo W* = {value(dual.objective):.2f} USD\n")

# ==============================================
# PASO 3: Verificación de Z* = W*
# ==============================================
print("===== VERIFICACIÓN DE DUALIDAD =====")
Z = value(primal.objective)
W = value(dual.objective)
print(f"Z* = {Z:.2f} | W* = {W:.2f}")
print(f"¿Son iguales? {abs(Z - W) < 1e-6} → Hueco de dualidad = {abs(Z - W):.6f}\n")

# ==============================================
# PASO 4: Verificación de Holgura Complementaria
# ==============================================
print("===== VERIFICACIÓN DE HOLGURA COMPLEMENTARIA =====")
# Valores lados derechos y uso de recursos
b = [120, 90, 80]
uso = [
    0.8*x1.value() + 1.0*x2.value(),
    0.5*x1.value() + 0.75*x2.value(),
    0.4*x1.value() + 0.6*x2.value()
]

for i in range(3):
    holgura = b[i] - uso[i]
    producto = precios_sombra[i] * holgura
    print(f"Restricción {i+1}: y{i+1}* * (b{i+1} - aᵀx*) = {precios_sombra[i]:.4f} * ({b[i]:.2f} - {uso[i]:.2f}) = {producto:.6f}")
    print(f"→ ¿≈ 0? {abs(producto) < 1e-6}\n")
