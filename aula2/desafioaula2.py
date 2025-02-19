def regressao_linear(x, y):
    n = len(x)
    media_x = sum(x) / n
    media_y = sum(y) / n

    numerador = 0
    denominador = 0

    for i in range(n):
        #Somando as multiplicações das subtrações de x[i] e y[i] em relação às suas respectivas médias.
        numerador += (x[i] - media_x) * (y[i] - media_y)
        #Somando os quadrados das subtrações de x[i] em relação à média de x.
        denominador += (x[i] - media_x) ** 2
    

    #quanto Y varia para cada unidade de X
    beta_1 = numerador / denominador
    #Y quando X = 0
    beta_0 = media_y - beta_1 * media_x
    
    return beta_0, beta_1

varia_x = [1, 2, 3, 4, 5]
varia_y = [2, 4, 5, 4, 5]

beta_0, beta_1 = regressao_linear(varia_x, varia_y)

print(f"Coeficiente linear (beta_0): {beta_0}")
print(f"Coeficiente angular (beta_1): {beta_1}")
