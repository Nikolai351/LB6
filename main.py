from counterpropagationNetwork import CounterpropagationNetwork


# Пример использования сети встречного распространения
if __name__ == "__main__":
    # Тестовые входные данные
    inputs = np.array([[0.5], [0.6]])
    targets = np.array([[0.7], [0.8]])
    
    # Параметры сети
    input_size = 1
    hidden_size = 2
    output_size = 2
    learning_rate = 0.1
    
    # Создание и обучение сети
    network = CounterpropagationNetwork(input_size, hidden_size, output_size, learning_rate)
    network.train(inputs, targets, iterations=1000)
    
    # Прогнозирование
    predictions = network.predict(inputs)
    print(f"Predictions: {predictions}")
