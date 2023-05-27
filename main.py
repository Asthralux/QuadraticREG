import numpy as np
import matplotlib.pyplot as plt

class QuadraticRegression:
    #El __init__ es como un constructor pero este no crea el objeto... se utiliza para inicializar un objeto después de que ha sido creado
    def __init__(self):
        # Conjunto de datos hardcodeados
        self.data = np.array([[-3, 7.5], [-2, 3], [-1, 0.5], [0, 1], [1, 3], [2, 6], [3, 14]])
        # Calcula los coeficientes Beta0, Beta1 y Beta2 para la regresión cuadrática
        self.beta0, self.beta1, self.beta2 = self.quadratic_fit(self.data)

    def quadratic_fit(self, data):
        # Extraer las coordenadas X e Y del conjunto de datos
        X = data[:, 0]
        Y = data[:, 1]

        # Calcular las sumas necesarias para la matriz A y el vector B
        n = len(X)
        x_sum = np.sum(X)
        x_squared_sum = np.sum(X**2)
        x_cubed_sum = np.sum(X**3)
        x_quartic_sum = np.sum(X**4)
        y_sum = np.sum(Y)
        xy_sum = np.sum(X * Y)
        x_squared_y_sum = np.sum(X**2 * Y)

        # Crear la matriz A y el vector B
        A = np.array([[x_quartic_sum, x_cubed_sum, x_squared_sum],
                      [x_cubed_sum, x_squared_sum, x_sum],
                      [x_squared_sum, x_sum, n]])
        B = np.array([x_squared_y_sum, xy_sum, y_sum])

        # Resolver el sistema de ecuaciones lineales para encontrar los coeficientes Beta0, Beta1 y Beta2
        return np.linalg.solve(A, B)

    def predict(self, x):
        # Predecir el valor Y utilizando la ecuación de la regresión cuadrática
        return self.beta0 * x ** 2 + self.beta1 * x + self.beta2

    def calculate_r_squared(self):
        # Calcular el coeficiente de determinación R^2
        Y = self.data[:, 1]
        y_mean = np.mean(Y)
        y_pred = self.predict(self.data[:, 0])

        SSE = np.sum((Y - y_pred)**2)
        SST = np.sum((Y - y_mean)**2)

        return 1 - (SSE / SST)

    def plot(self):
        # Graficar la curva de la regresión cuadrática y los puntos del conjunto de datos
        x = np.linspace(np.min(self.data[:, 0]) - 1, np.max(self.data[:, 0]) + 1, 100)
        y = self.predict(x)

        # La siguiente línea de código crea una gráfica de dispersión de los puntos de datos del dataset.
        # self.data[:, 0] representa los valores de X y self.data[:, 1] representa los valores de Y.
        # Los puntos de datos se colorean de rojo y se etiquetan como 'Data points' en la leyenda.
        plt.scatter(self.data[:, 0], self.data[:, 1], color='red', label='Data points')
        plt.plot(x, y, color='black', label='Quadratic fit')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.title('Quadratic Regression')
        plt.show()

    def __str__(self):
        # Devolver la ecuación de la regresión cuadrática como una cadena de texto
        return f"y = {self.beta0:.4f}x^2 + {self.beta1:.4f}x + {self.beta2:.4f}"

if __name__ == "__main__":
    # Crear una instancia de la clase QuadraticRegression
    qr = QuadraticRegression()

    # Imprimir la ecuación de la regresión cuadrática
    print("Ecuación de regresión cuadrática:")
    print(qr)

    # Predecir un valor de Y a partir de un valor X de entrada
    x_input = float(input("Ingrese un valor de X para predecir Y: "))
    y_predicted = qr.predict(x_input)
    print(f"Valor Y predecido: {y_predicted}")

    # Calcular e imprimir el coeficiente de determinación R^2
    r_squared = qr.calculate_r_squared()
    print(f"Coeficiente de determinación R^2: {r_squared}")

    # Graficar la curva de la regresión cuadrática y los puntos del conjunto de datos
    qr.plot()
