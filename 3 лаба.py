import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

def load_road_network_data(filename):
    """Загрузка данных 3D дорожной сети из txt файла"""
    try:
       
        data = np.loadtxt(filename, delimiter=',')
        print(f"Данные успешно загружены. Размер: {data.shape}")
        
       
        X = data[:, :2]  
        y = data[:, 2]   
        
       
        X = (X - X.mean(axis=0)) / X.std(axis=0)
        
        return X, y
        
    except Exception as e:
        print(f"Ошибка загрузки данных: {e}")
        return create_synthetic_road_data()

def create_synthetic_road_data():
    """Создание синтетических данных дорожной сети"""
    np.random.seed(42)
    n_samples = 1000
    
    t = np.linspace(0, 4*np.pi, n_samples)
    x = 10 * np.cos(t) + np.random.normal(0, 0.5, n_samples)
    y = 10 * np.sin(t) + np.random.normal(0, 0.5, n_samples)
    z = (np.sin(x/2) + np.cos(y/3) + 0.5 * np.sin(0.7*x) * np.cos(0.5*y) + np.random.normal(0, 0.1, n_samples))
    
    X = np.column_stack([x, y])
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    return X, z


file_path = r'C:\моё\Универ\5 семетср\МО\3D_spatial_network.txt'
X, y = load_road_network_data(file_path)

print(f"Размерность данных: X {X.shape}, y {y.shape}")
print(f"Диапазон координат X: [{X[:, 0].min():.2f}, {X[:, 0].max():.2f}] x [{X[:, 1].min():.2f}, {X[:, 1].max():.2f}]")
print(f"Диапазон высот y: [{y.min():.2f}, {y.max():.2f}]")
print(f"Среднее высот: {y.mean():.2f}, Стандартное отклонение: {y.std():.2f}")


plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='terrain', s=1, alpha=0.6)
plt.colorbar(scatter, label='Высота')
plt.xlabel('Нормализованная координата X')
plt.ylabel('Нормализованная координата Y')
plt.title('Распределение высот в дорожной сети')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.hist(y, bins=50, density=True, alpha=0.7, color='skyblue', edgecolor='black')
plt.xlabel('Высота')
plt.ylabel('Плотность')
plt.title('Распределение высот')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


def train_test_split_custom(X, y, test_size=0.2, random_state=None):
    """Кастомная функция разделения на обучающую и тестовую выборки"""
    if random_state:
        np.random.seed(random_state)
    
    n_samples = X.shape[0]
    n_test = int(n_samples * test_size)
    
    indices = np.random.permutation(n_samples)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    
    return X_train, X_test, y_train, y_test

# Разделение данных
X_train, X_test, y_train, y_test = train_test_split_custom(X, y, test_size=0.2, random_state=42)
print(f"Обучающая выборка: {X_train.shape[0]} samples")
print(f"Тестовая выборка: {X_test.shape[0]} samples")


print("\n")
print("ЛИНЕЙНАЯ РЕГРЕССИЯ")
print("\n")

# Обучение линейной регрессии
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

# Предсказания
y_train_pred = lin_reg.predict(X_train)
y_test_pred = lin_reg.predict(X_test)

# Оценка точности
train_mse_linear = mean_squared_error(y_train, y_train_pred)
test_mse_linear = mean_squared_error(y_test, y_test_pred)
train_r2_linear = r2_score(y_train, y_train_pred)
test_r2_linear = r2_score(y_test, y_test_pred)

print(f"Коэффициенты модели: {lin_reg.coef_}")
print(f"Свободный член: {lin_reg.intercept_:.4f}")
print(f"MSE обучающей выборки: {train_mse_linear:.6f}")
print(f"MSE тестовой выборки: {test_mse_linear:.6f}")
print(f"R² обучающей выборки: {train_r2_linear:.4f}")
print(f"R² тестовой выборки: {test_r2_linear:.4f}")


def polynomial_regression_analysis(X_train, X_test, y_train, y_test, max_degree=4):
    """Анализ полиномиальной регрессии для разных степеней"""
    train_errors = []
    test_errors = []
    train_r2_scores = []
    test_r2_scores = []
    degrees = range(1, max_degree + 1)
    
    for degree in degrees:
        try:
            # Создание полиномиальных признаков
            poly_features = PolynomialFeatures(degree=degree, include_bias=False)
            
            model = Pipeline([
                ('poly', poly_features),
                ('scaler', StandardScaler()),
                ('lin_reg', LinearRegression())
            ])
            
            # Обучение модели
            model.fit(X_train, y_train)
            
            # Предсказания
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            # Расчет ошибок
            train_errors.append(mean_squared_error(y_train, y_train_pred))
            test_errors.append(mean_squared_error(y_test, y_test_pred))
            train_r2_scores.append(r2_score(y_train, y_train_pred))
            test_r2_scores.append(r2_score(y_test, y_test_pred))
            
            print(f"Степень {degree}: Train MSE = {train_errors[-1]:.6f}, Test MSE = {test_errors[-1]:.6f}, Test R² = {test_r2_scores[-1]:.4f}")
            
        except Exception as e:
            print(f"Ошибка для степени {degree}: {e}")
            break
    
    return degrees, train_errors, test_errors, train_r2_scores, test_r2_scores

print("\n")
print("ПОЛИНОМИАЛЬНАЯ РЕГРЕССИЯ")
print("\n")
degrees, train_errors_poly, test_errors_poly, train_r2_poly, test_r2_poly = polynomial_regression_analysis(
    X_train, X_test, y_train, y_test, max_degree=4
)

# Построение графиков для полиномиальной регрессии
plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.plot(degrees, train_errors_poly, 'bo-', label='Обучающая выборка', linewidth=2, markersize=8)
plt.plot(degrees, test_errors_poly, 'ro-', label='Тестовая выборка', linewidth=2, markersize=8)
plt.xlabel('Степень полинома')
plt.ylabel('MSE')
plt.title('Зависимость MSE от степени полинома')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(degrees, train_r2_poly, 'bo-', label='Обучающая выборка', linewidth=2, markersize=8)
plt.plot(degrees, test_r2_poly, 'ro-', label='Тестовая выборка', linewidth=2, markersize=8)
plt.xlabel('Степень полинома')
plt.ylabel('R² score')
plt.title('Зависимость R² от степени полинома')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


def regularization_analysis(X_train, X_test, y_train, y_test, degree=3, alpha_values=np.logspace(-3, 3, 50)):
    """Анализ моделей с регуляризацией"""
    # Создаем полиномиальные признаки
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    
    # Масштабирование признаков
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_poly)
    X_test_scaled = scaler.transform(X_test_poly)
    
    ridge_train_errors = []
    ridge_test_errors = []
    lasso_train_errors = []
    lasso_test_errors = []
    ridge_r2_scores = []
    lasso_r2_scores = []
    
    print("Прогресс регуляризации: ", end="")
    
    for i, alpha in enumerate(alpha_values):
        if i % 10 == 0:
            print(f"{i*2}%", end=" ")
        
        # Ridge регрессия
        ridge = Ridge(alpha=alpha, random_state=42, max_iter=1000)
        ridge.fit(X_train_scaled, y_train)
        
        y_train_pred_ridge = ridge.predict(X_train_scaled)
        y_test_pred_ridge = ridge.predict(X_test_scaled)
        
        ridge_train_errors.append(mean_squared_error(y_train, y_train_pred_ridge))
        ridge_test_errors.append(mean_squared_error(y_test, y_test_pred_ridge))
        ridge_r2_scores.append(r2_score(y_test, y_test_pred_ridge))
        
        # Lasso регрессия
        lasso = Lasso(alpha=alpha, max_iter=5000, random_state=42, tol=0.001)
        lasso.fit(X_train_scaled, y_train)
        
        y_train_pred_lasso = lasso.predict(X_train_scaled)
        y_test_pred_lasso = lasso.predict(X_test_scaled)
        
        lasso_train_errors.append(mean_squared_error(y_train, y_train_pred_lasso))
        lasso_test_errors.append(mean_squared_error(y_test, y_test_pred_lasso))
        lasso_r2_scores.append(r2_score(y_test, y_test_pred_lasso))

    return (alpha_values, ridge_train_errors, ridge_test_errors, ridge_r2_scores,
            lasso_train_errors, lasso_test_errors, lasso_r2_scores)

print("\n")
print("РЕГУЛЯРИЗАЦИЯ (RIDGE И LASSO)")
print("\n")

alpha_values = np.logspace(-3, 3, 50)
results = regularization_analysis(X_train, X_test, y_train, y_test, degree=3, alpha_values=alpha_values)

# Построение графиков для регуляризации
plt.figure(figsize=(15, 10))

# Графики MSE для Ridge и Lasso
plt.subplot(2, 2, 1)
plt.semilogx(alpha_values, results[1], 'b-', label='Ridge Train', linewidth=2, alpha=0.7)
plt.semilogx(alpha_values, results[2], 'r-', label='Ridge Test', linewidth=2)
plt.xlabel('Коэффициент регуляризации (alpha)')
plt.ylabel('MSE')
plt.title('Ridge Regression: Зависимость MSE от alpha')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 2)
plt.semilogx(alpha_values, results[4], 'b-', label='Lasso Train', linewidth=2, alpha=0.7)
plt.semilogx(alpha_values, results[5], 'r-', label='Lasso Test', linewidth=2)
plt.xlabel('Коэффициент регуляризации (alpha)')
plt.ylabel('MSE')
plt.title('Lasso Regression: Зависимость MSE от alpha')
plt.legend()
plt.grid(True, alpha=0.3)

# Графики R² для Ridge и Lasso
plt.subplot(2, 2, 3)
plt.semilogx(alpha_values, results[3], 'g-', label='Ridge Test R²', linewidth=2)
plt.xlabel('Коэффициент регуляризации (alpha)')
plt.ylabel('R² score')
plt.title('Ridge Regression: Зависимость R² от alpha')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 4)
plt.semilogx(alpha_values, results[6], 'g-', label='Lasso Test R²', linewidth=2)
plt.xlabel('Коэффициент регуляризации (alpha)')
plt.ylabel('R² score')
plt.title('Lasso Regression: Зависимость R² от alpha')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n")
print("ДЕТАЛЬНЫЙ АНАЛИЗ РЕЗУЛЬТАТОВ")
print("\n")

# Нахождение оптимальных параметров
optimal_poly_degree = degrees[np.argmin(test_errors_poly)]
optimal_ridge_alpha = alpha_values[np.argmin(results[2])]
optimal_lasso_alpha = alpha_values[np.argmin(results[5])]

print("ОПТИМАЛЬНЫЕ ПАРАМЕТРЫ:")
print(f"• Степень полинома: {optimal_poly_degree}")
print(f"• Alpha для Ridge: {optimal_ridge_alpha:.6f}")
print(f"• Alpha для Lasso: {optimal_lasso_alpha:.6f}")

print("\nСРАВНЕНИЕ КАЧЕСТВА МОДЕЛЕЙ:")
print(f"• Линейная регрессия - Test R²: {test_r2_linear:.4f}")
print(f"• Полиномиальная регрессия (степень {optimal_poly_degree}) - Test R²: {test_r2_poly[optimal_poly_degree-1]:.4f}")

# Обучение лучших моделей с регуляризацией
poly = PolynomialFeatures(degree=3, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_poly)
X_test_scaled = scaler.transform(X_test_poly)

best_ridge = Ridge(alpha=optimal_ridge_alpha, random_state=42, max_iter=1000)
best_ridge.fit(X_train_scaled, y_train)
y_test_pred_ridge = best_ridge.predict(X_test_scaled)
ridge_final_r2 = r2_score(y_test, y_test_pred_ridge)

best_lasso = Lasso(alpha=optimal_lasso_alpha, max_iter=5000, random_state=42, tol=0.001)
best_lasso.fit(X_train_scaled, y_train)
y_test_pred_lasso = best_lasso.predict(X_test_scaled)
lasso_final_r2 = r2_score(y_test, y_test_pred_lasso)

print(f"• Ridge регрессия (alpha={optimal_ridge_alpha:.2e}) - Test R²: {ridge_final_r2:.4f}")
print(f"• Lasso регрессия (alpha={optimal_lasso_alpha:.2e}) - Test R²: {lasso_final_r2:.4f}")

# Анализ переобучения
print("\nАНАЛИЗ ПЕРЕОБУЧЕНИЯ:")
overfitting_ratio = train_errors_poly[optimal_poly_degree-1] / test_errors_poly[optimal_poly_degree-1]
print(f"• Соотношение MSE train/test: {overfitting_ratio:.4f}")

if overfitting_ratio < 0.9:
    print("• ВЫВОД: Возможное недообучение")
elif overfitting_ratio > 1.1:
    print("• ВЫВОД: Признаки переобучения")
else:
    print("• ВЫВОД: Модель обобщается хорошо")

# Финальная визуализация сравнения моделей
models = ['Linear', f'Poly deg{optimal_poly_degree}', 'Ridge', 'Lasso']
r2_scores = [test_r2_linear, test_r2_poly[optimal_poly_degree-1], ridge_final_r2, lasso_final_r2]

plt.figure(figsize=(10, 6))
colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728']
bars = plt.bar(models, r2_scores, color=colors, alpha=0.7, edgecolor='black')
plt.ylabel('R² Score (тестовая выборка)')
plt.title('Сравнение эффективности моделей регрессии')
plt.ylim(0, max(r2_scores) * 1.1)
plt.grid(True, alpha=0.3, axis='y')

# Добавляем значения на столбцы
for bar, score in zip(bars, r2_scores):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{score:.4f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show()

print("\n")
print("ЗАКЛЮЧЕНИЕ И РЕКОМЕНДАЦИИ")
print("\n")
best_model_idx = np.argmax(r2_scores)
best_model = models[best_model_idx]
best_score = r2_scores[best_model_idx]

print(f"НАИЛУЧШАЯ МОДЕЛЬ: {best_model}")
print(f"Качество предсказания (R²): {best_score:.4f}")

if best_score < 0.3:
    print("РЕКОМЕНДАЦИЯ: Качество моделей низкое. Возможно:")
    print("- Данные имеют сложную нелинейную структуру")
    print("- Требуются дополнительные признаки")
elif best_score < 0.6:
    print("РЕКОМЕНДАЦИЯ: Умеренное качество моделей.")
    print("- Полиномиальные модели показывают значительное улучшение")
else:
    print("РЕКОМЕНДАЦИЯ: Хорошее качество моделей.")
    print("- Полиномиальные модели хорошо подходят для данных")

print(f"\nДля прогнозирования высот рекомендуется использовать: {best_model}")
