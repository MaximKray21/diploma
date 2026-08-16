import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from skopt import gp_minimize
from skopt.space import Real
import warnings

warnings.filterwarnings("ignore")

# --- КОНСТАНТЫ ---
MU_MOON = 4.9048695e12 # Чуть точнее гравитационный параметр
R_MOON = 1737400.0        
M_INITIAL = 1462.0
M_PROPELLANT_INITIAL = 958.0
DRY_MASS = M_INITIAL - M_PROPELLANT_INITIAL
H_ORBIT = 100000.0        
THRUST = 2930.0     
ISP = 310.0          
G0 = 9.80665            
MASS_FLOW = THRUST / (ISP * G0) 
G_MOON = 1.62

HS_TARGET = 2000.0   
TARGET_VX = 35.0
TARGET_VY = -35.0

# --- 1. АНАЛИТИКА ---
def calculate_doi(H_pdi):
    r_apo = R_MOON + H_ORBIT
    r_peri = R_MOON + H_pdi
    a_ellipse = (r_apo + r_peri) / 2.0
    V_circ = np.sqrt(MU_MOON / r_apo)
    V_apo_ellipse = np.sqrt(MU_MOON * (2.0 / r_apo - 1.0 / a_ellipse))
    Delta_V = V_circ - V_apo_ellipse
    M_start_descent = M_INITIAL * np.exp(-Delta_V / (ISP * G0))
    fuel_doi = M_INITIAL - M_start_descent
    Vx_peri = np.sqrt(MU_MOON * (2.0 / r_peri - 1.0 / a_ellipse))
    return Vx_peri, M_start_descent, fuel_doi

# --- 2. СИМУЛЯТОР ---
def lander_dynamics(t, state, Kp, Vy_start_target, Vy_end, N, H_pdi):
    x, y, vx, vy, m = state
    h_norm = np.clip(y / H_pdi, 0.0, 1.0)
    
    vy_desired = Vy_end + (Vy_start_target - Vy_end) * (h_norm ** N)
    
    ay_desired = Kp * (vy_desired - vy)
    Ty = m * (ay_desired + G_MOON)
    Ty = np.clip(Ty, 0, THRUST)
        
    Tx_max = np.sqrt(max(0, THRUST**2 - Ty**2))
    
    # === НОВАЯ ЛОГИКА ТОРМОЖЕНИЯ (БЕЗ ГРАВИТАЦИОННЫХ ПОТЕРЬ) ===
    if vx > TARGET_VX + 5.0:
        Tx = -Tx_max # Тормозим на максимум
    elif vx > TARGET_VX:
        # Плавное снижение тяги, чтобы не проскочить 35 м/с
        Tx = -Tx_max * ((vx - TARGET_VX) / 5.0) 
    else:
        Tx = 0 # Достигли 35 м/с — перестаем жечь топливо по X! Высвобождаем мощь для Ty.
        
    ax = Tx / m
    ay = (Ty / m) - G_MOON
    dm_dt = -MASS_FLOW
    
    if m <= DRY_MASS: 
        ax, ay, dm_dt = 0, -G_MOON, 0
    
    return [vx, vy, ax, ay, dm_dt]

def reach_hs(t, state, *args):
    return state[1] - HS_TARGET
reach_hs.terminal = True
reach_hs.direction = -1

# --- 3. ФУНКЦИЯ ПОТЕРЬ ---
def objective(params):
    Kp, Vy_start_target, Vy_end, N, H_pdi = params
    
    Vx0, M0, fuel_doi = calculate_doi(H_pdi)
    initial_state = [0.0, H_pdi, Vx0, 0.0, M0]
    
    sol = solve_ivp(
        lander_dynamics, (0, 1500), initial_state, 
        args=(Kp, Vy_start_target, Vy_end, N, H_pdi),
        events=reach_hs, max_step=2.0
    )
    
    if len(sol.t_events[0]) == 0:
        return 999999.0
        
    vx_final = sol.y[2][-1]
    vy_final = sol.y[3][-1]
    m_final = sol.y[4][-1]
    
    if m_final <= DRY_MASS + 5.0:
        return 999999.0
    
    total_fuel_consumed = fuel_doi + (M0 - m_final)
    
    # Штраф за Vx теперь минимален, так как автопилот сам ее держит.
    # Главная задача оптимизатора - идеально попасть в Vy = -35.
    error_vy = abs(vy_final - TARGET_VY)
    
    loss = total_fuel_consumed + 1000.0 * error_vy
    return loss

# --- 4. ОПТИМИЗАЦИЯ ---
space = [
    Real(0.05, 0.5, name='Kp'),                 
    Real(-30.0, 0.0, name='Vy_start_target'),  
    Real(-60.0, -20.0, name='Vy_end'),         
    Real(0.5, 2.5, name='N'),                   
    Real(12000.0, 25000.0, name='H_pdi') # Слабому двигателю может потребоваться тормозить с бОльшей высоты     
]

x0 = [0.1, -10.0, -40.0, 1.0, 18000.0]

print("Запуск УМНОЙ оптимизации... Ожидайте.")
res = gp_minimize(objective, space, x0=x0, n_calls=70, random_state=42, verbose=False)
best_Kp, best_Vy_start, best_Vy_end, best_N, best_H_pdi = res.x

print("\n" + "="*50)
print(" ОПТИМИЗАЦИЯ УСПЕШНО ЗАВЕРШЕНА")
print("="*50)
print(f"Идеальная высота схода с орбиты (H_pdi): {best_H_pdi:.1f} м")
print(f"Коэффициент П-регулятора (Kp): {best_Kp:.4f}")
print(f"Целевая начальная скорость (Vy_start): {best_Vy_start:.2f} м/с")
print(f"Целевая конечная скорость (Vy_end): {best_Vy_end:.2f} м/с")
print(f"Степень кривизны глиссады (N): {best_N:.3f}")

# --- 5. ФИНАЛЬНЫЙ РАСЧЕТ ---
Vx0, M0, fuel_doi = calculate_doi(best_H_pdi)
sol_best = solve_ivp(
    lander_dynamics, (0, 1500), [0.0, best_H_pdi, Vx0, 0.0, M0], 
    args=(best_Kp, best_Vy_start, best_Vy_end, best_N, best_H_pdi),
    events=reach_hs, max_step=1.0
)

t_best, y_best = sol_best.t, sol_best.y[1]
vx_best, vy_best, m_best = sol_best.y[2], sol_best.y[3], sol_best.y[4]
x_best = sol_best.y[0]
total_best_fuel = fuel_doi + (M0 - m_best[-1])

print(f"\n[Результаты на высоте передачи управления ИИ (2000 м)]")
print(f"Фактическая Vx: {vx_best[-1]:.2f} м/с (Требовалось {TARGET_VX})")
print(f"Фактическая Vy: {vy_best[-1]:.2f} м/с (Требовалось {TARGET_VY})")
print(f"СУММАРНОЕ ТОПЛИВО (Миссия): {total_best_fuel:.2f} кг")
print(f"Остаток топлива для ИИ (ECS-AI): {m_best[-1] - DRY_MASS:.2f} кг")

# Расчет угла на высоте 2 км
h_norm_hs = np.clip(y_best[-1] / best_H_pdi, 0.0, 1.0)
vy_desired_hs = best_Vy_end + (best_Vy_start - best_Vy_end) * (h_norm_hs ** best_N)
ay_desired_hs = best_Kp * (vy_desired_hs - vy_best[-1])

Ty_hs = m_best[-1] * (ay_desired_hs + G_MOON)
Ty_hs = np.clip(Ty_hs, 0, THRUST)
Tx_hs = np.sqrt(max(0, THRUST**2 - Ty_hs**2))

if vx_best[-1] > TARGET_VX + 5.0:
    Tx_hs_actual = Tx_hs
elif vx_best[-1] > TARGET_VX:
    Tx_hs_actual = Tx_hs * ((vx_best[-1] - TARGET_VX) / 5.0)
else:
    Tx_hs_actual = 0

pitch_deg = np.degrees(np.arctan2(Ty_hs, Tx_hs_actual))
print(f"Угол наклона тяги к горизонту на 2 км: {pitch_deg:.1f} градусов")
print(f"Отклонение КА от вертикали (Pitch-up maneuver): {90.0 - pitch_deg:.1f} градусов")

# --- 6. ВИЗУАЛИЗАЦИЯ ---
fig, axs = plt.subplots(2, 2, figsize=(14, 9))

# 1. Профиль
axs[0, 0].plot(x_best/1000, y_best/1000)
axs[0, 0].scatter(x_best[-1]/1000, y_best[-1]/1000, color='red', s=50, zorder=5, label='Старт TerrAn/ECS (2 км)')
axs[0, 0].set_title(f"Профиль траектории (Сход с {best_H_pdi/1000:.1f} км)")
axs[0, 0].set_xlabel("Дальность X, км")
axs[0, 0].set_ylabel("Высота Y, км")
axs[0, 0].grid()
axs[0, 0].legend()

# 2. Скорости
axs[0, 1].plot(t_best, vx_best, label='Vx (Горизонтальная)')
axs[0, 1].plot(t_best, vy_best, label='Vy (Вертикальная)')
axs[0, 1].axhline(TARGET_VX, color='g', linestyle=':', label='Target Vx (35 m/s)')
axs[0, 1].axvline(t_best[-1], color='red', linestyle='--')
axs[0, 1].set_title("Компоненты скорости от времени")
axs[0, 1].set_xlabel("Время, с")
axs[0, 1].set_ylabel("Скорость, м/с")
axs[0, 1].grid()
axs[0, 1].legend()

# 3. Фазовый портрет
axs[1, 0].plot(x_best/1000, vx_best)
axs[1, 0].axvline(x_best[-1]/1000, color='red', linestyle='--')
axs[1, 0].set_title("Фазовый портрет: Гориз. Скорость от Дальности")
axs[1, 0].set_xlabel("Дальность X, км")
axs[1, 0].set_ylabel("Скорость Vx, м/с")
axs[1, 0].grid()

# 4. Масса
axs[1, 1].plot(t_best, m_best)
axs[1, 1].axhline(DRY_MASS, color='black', linestyle='-', linewidth=2, label='Сухая масса (Бак пуст)')
axs[1, 1].axvline(t_best[-1], color='red', linestyle='--')
axs[1, 1].set_title("Масса КА от времени (Расход)")
axs[1, 1].set_xlabel("Время, с")
axs[1, 1].set_ylabel("Масса, кг")
axs[1, 1].grid()
axs[1, 1].legend()

plt.tight_layout()
plt.show()
