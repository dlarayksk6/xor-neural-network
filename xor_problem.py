import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])

model = Sequential([
    Dense(4, input_dim=2, activation='tanh'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy')

model.fit(X, y, epochs=5000, verbose=0)

pred = model.predict(X)
classes = (pred > 0.5).astype(int)

print("Olasılıklar:")
print(pred)

print("Sınıflar:")
print(classes)
