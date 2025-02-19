def calcular_regressao_linear(x, y):
    # Calcular as médias de x e y
    n = len(x)
    media_x = sum(x) / n
    media_y = sum(y) / n
    
    # Calcular beta_1 (coeficiente angular)
    numerador = sum((x[i] - media_x) * (y[i] - media_y) for i in range(n))
    denominador = sum((x[i] - media_x) ** 2 for i in range(n))
    beta_1 = numerador / denominador
    
    # Calcular beta_0 (coeficiente linear)
    beta_0 = media_y - beta_1 * media_x
    
    return beta_0, beta_1

# Exemplo de entrada
dados_x = [1, 2, 3, 4, 5]
dados_y = [2, 4, 5, 4, 5]

# Calcular os coeficientes
beta_0, beta_1 = calcular_regressao_linear(dados_x, dados_y)

# Exibir os resultados
print(f"Coeficiente linear (beta_0): {beta_0}")
print(f"Coeficiente angular (beta_1): {beta_1}")
