import numpy as np

class CounterpropagationNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        # Инициализация весов
        self.W_ih = np.random.uniform(-1, 1, (hidden_size, input_size))
        self.W_ho = np.random.uniform(-1, 1, (output_size, hidden_size))
        self.W_oh = np.random.uniform(-1, 1, (hidden_size, output_size))
        
        # Инициализация функций активации
        self.activation_func = lambda x: 1/(1+np.exp(-x))
        self.activation_func_prime = lambda x: x*(1-x)
        
        # Векторы состояния
        self.h_prev = None
        self.o_prev = None
    
    def forward(self, inputs):
        self.h_prev = self.activation_func(np.dot(self.W_ih, inputs))
        self.o_prev = self.activation_func(np.dot(self.W_ho, self.h_prev))
        return self.o_prev
    
    def backward(self, targets):
        # Вычисление ошибки на выходном слое
        o_error = targets - self.o_prev
        o_delta = o_error * self.activation_func_prime(self.o_prev)
        
        # Обновление весов выходного слоя
        self.W_oh += self.learning_rate * np.dot(o_delta.reshape(-1, 1), self.h_prev.reshape(1, -1))
        
        # Вычисление скрытого состояния на основе ошибок
        h_reconstruct = self.activation_func(np.dot(self.W_ho.T, targets))
        h_error = h_reconstruct - self.h_prev
        h_delta = h_error * self.activation_func_prime(self.h_prev)
        
        # Обновление весов скрытого слоя
        self.W_ih += self.learning_rate * np.dot(h_delta.reshape(-1, 1), inputs.reshape(1, -1))
    
    def train(self, inputs, targets, iterations=1000):
        for i in range(iterations):
            if i % 100 == 0:
                print(f"Iteration: {i}")
            outputs = self.forward(inputs)
            self.backward(targets)
    
    def predict(self, inputs):
        return self.forward(inputs)
      
