import numpy as np

# Activation functions


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def tanh(x):
    return np.tanh(x)

# Derivative of activation functions


def sigmoid_derivative(x):
    return x * (1 - x)


def tanh_derivative(x):
    return 1 - x**2


class LSTM:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Initializing weights and biases
        self.Wf = np.random.randn(hidden_size, input_size + hidden_size)
        self.bf = np.zeros((hidden_size, 1))

        self.Wi = np.random.randn(hidden_size, input_size + hidden_size)
        self.bi = np.zeros((hidden_size, 1))

        self.Wc = np.random.randn(hidden_size, input_size + hidden_size)
        self.bc = np.zeros((hidden_size, 1))

        self.Wo = np.random.randn(hidden_size, input_size + hidden_size)
        self.bo = np.zeros((hidden_size, 1))

        self.Wy = np.random.randn(output_size, hidden_size)
        self.by = np.zeros((output_size, 1))

    def forward(self, X):
        m = X.shape[1]
        self.c = np.zeros((self.hidden_size, m))
        self.h = np.zeros((self.hidden_size, m))
        self.y = np.zeros((self.output_size, m))

        for t in range(X.shape[0]):
            x_t = X[t, :].reshape(-1, 1)
            concat = np.vstack((self.h, x_t))

            # Forget gate
            ft = sigmoid(self.Wf.dot(concat) + self.bf)
            # Input gate
            it = sigmoid(self.Wi.dot(concat) + self.bi)
            # Candidate value
            cct = tanh(self.Wc.dot(concat) + self.bc)
            # Update cell state
            self.c = ft * self.c + it * cct
            # Output gate
            ot = sigmoid(self.Wo.dot(concat) + self.bo)
            # Hidden state
            self.h = ot * tanh(self.c)
            # Output
            self.y = self.Wy.dot(self.h) + self.by
        return self.y

    def backward(self, X, y, learning_rate=0.01):
        m = X.shape[1]
        dy = self.y - y

        # Backpropagate through time
        dWy = dy.dot(self.h.T)
        dby = np.sum(dy, axis=1, keepdims=True)
        dh = self.Wy.T.dot(dy)
        dc_next = np.zeros_like(self.c)

        for t in reversed(range(X.shape[0])):
            x_t = X[t, :].reshape(-1, 1)
            concat = np.vstack((self.h, x_t))

            dot = dh * tanh(self.c)
            dc = dc_next + (dh * self.Wo.T).dot(tanh_derivative(tanh(self.c))
                                                ) + dot * sigmoid_derivative(self.h)
            dft = dc * self.c
            dit = dc * tanh(self.c)
            do = dh * tanh(self.c)
            dcct = dc * self.i

            dWf = dft * sigmoid_derivative(self.Wf.dot(concat) + self.bf)
            dWi = dit * sigmoid_derivative(self.Wi.dot(concat) + self.bi)
            dWc = dcct * tanh_derivative(self.Wc.dot(concat) + self.bc)
            dWo = do * sigmoid_derivative(self.Wo.dot(concat) + self.bo)

            dX = (self.Wf.T.dot(dft) + self.Wi.T.dot(dit) +
                  self.Wc.T.dot(dcct) + self.Wo.T.dot(do))[-self.input_size:]
            dWy_t = dy[:, t].reshape(-1, 1)
            dby_t = dy[:, t].reshape(-1, 1)

            # Update weights and biases
            self.Wf -= learning_rate * dWf
            self.bf -= learning_rate * np.sum(dft, axis=1, keepdims=True)
            self.Wi -= learning_rate * dWi
            self.bi -= learning_rate * np.sum(dit, axis=1, keepdims=True)
            self.Wc -= learning_rate * dWc
            self.bc -= learning_rate * np.sum(dcct, axis=1, keepdims=True)
            self.Wo -= learning_rate * dWo
            self.bo -= learning_rate * np.sum(do, axis=1, keepdims=True)

            self.Wy -= learning_rate * dWy_t
            self.by -= learning_rate * dby_t

            dh = dX
            dc_next = dc * self.f

    def train(self, X, y, epochs=100, learning_rate=0.01):
        for epoch in range(epochs):
            y_pred = self.forward(X)
            loss = np.mean(0.5 * (y_pred - y)**2)
            self.backward(X, y, learning_rate)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.6f}")


input_size = 5
hidden_size = 10
output_size = 1

X_train = np.random.randn(100, input_size)
y_train = np.random.randn(100, output_size)

lstm = LSTM(input_size, hidden_size, output_size)
lstm.train(X_train, y_train)
