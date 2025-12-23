import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def generate_csv():
    
    x1 = np.linspace(-2*np.pi, 2*np.pi, 500) 
    x2 = np.linspace(-5, 5, 500)              
    
    
    y = np.cos(x1) * (x2**3)
    
    
    df = pd.DataFrame({
        'x1': x1,
        'x2': x2,
        'y': y
    })
    
    
    df.to_csv('lab2_data.csv', index=False)
    print("Файл 'lab2_data.csv' успешно создан!")
    return df


def analyze_data():
    
    df = pd.read_csv('lab2_data.csv')
    
    
    stats = {
        'x1': {'mean': df['x1'].mean(), 'min': df['x1'].min(), 'max': df['x1'].max()},
        'x2': {'mean': df['x2'].mean(), 'min': df['x2'].min(), 'max': df['x2'].max()},
        'y': {'mean': df['y'].mean(), 'min': df['y'].min(), 'max': df['y'].max()}
    }
    
    
    print("\nСтатистические показатели:")
    for column in ['x1', 'x2', 'y']:
        print(f"{column}: среднее = {stats[column]['mean']:.4f}, "
              f"минимум = {stats[column]['min']:.4f}, "
              f"максимум = {stats[column]['max']:.4f}")
    
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    
    x2_const = stats['x2']['mean']
    ax1.plot(df['x1'], df['y'], 'b-', linewidth=1)
    ax1.set_xlabel('x1')
    ax1.set_ylabel('y')
    ax1.set_title(f'График y от x1 (x2 = {x2_const:.2f} - константа)')
    ax1.grid(True)
    

    x1_const = stats['x1']['mean']
    ax2.plot(df['x2'], df['y'], 'ro', markersize=2, alpha=0.6)
    ax2.set_xlabel('x2')
    ax2.set_ylabel('y')
    ax2.set_title(f'График y от x2 (x1 = {x1_const:.2f} - константа)\nТочечный график')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return df, stats


def filter_and_save(df, stats):
  
    condition = (df['x1'] < stats['x1']['mean']) | (df['x2'] < stats['x2']['mean'])
    filtered_df = df[condition]
    
  
    filtered_df.to_csv('lab2_filtered_data.csv', index=False)
    print(f"\nОтфильтровано {len(filtered_df)} строк из {len(df)}")
    print("Файл 'lab2_filtered_data.csv' успешно создан!")
    
    return filtered_df


def plot_3d_graph(df):
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
  
    x1_3d = np.linspace(df['x1'].min(), df['x1'].max(), 50)
    x2_3d = np.linspace(df['x2'].min(), df['x2'].max(), 50)
    X1, X2 = np.meshgrid(x1_3d, x2_3d)
    Y = np.cos(X1) * (X2**3)
    
  
    surf = ax.plot_surface(X1, X2, Y, cmap='viridis', alpha=0.8, 
                          linewidth=0, antialiased=True)
    

    ax.scatter(df['x1'], df['x2'], df['y'], c='red', alpha=0.3, s=1)
    
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('y')
    ax.set_title('3D график функции: y = cos(x1) * x2³')
    

    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    
    plt.show()


def main():
    print("Функция: y = cos(x1) * x2³")
    print("=" * 50)
    

    df_generated = generate_csv()
    
 
    df, stats = analyze_data()
    

    filtered_df = filter_and_save(df, stats)
    
  
    plot_3d_graph(df)
    


if __name__ == "__main__":
    main()
