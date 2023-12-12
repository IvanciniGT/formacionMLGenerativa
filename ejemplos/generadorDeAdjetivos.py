import torch
import torch.nn as nn
import torch.optim as optim

# Leer el corpus
fichero = open("corpus.txt", "r")
corpus = fichero.read()
print("Corpus:", corpus)


# Preprocesamiento del texto
tokens = set(corpus.split())
word_to_idx = {word: idx for idx, word in enumerate(tokens)}
idx_to_word = {idx: word for idx, word in enumerate(tokens)}
num_tokens = len(tokens)
print("Tokens:", tokens)

# Crear pares de entrada y objetivo
# Quedarme con las palabras impares
input_sequence = [word_to_idx[word] for word in corpus.split()[::2]]
# Y ahora con las pares
target_sequence = [word_to_idx[word] for word in corpus.split()[1::2]]

print("Input sequence:", [word for word in corpus.split()[::2]])
print("Target sequence:", [word for word in corpus.split()[1::2]])


# Convertir a tensores de PyTorch
input_sequence = torch.LongTensor(input_sequence)
target_sequence = torch.LongTensor(target_sequence)


class TextGenerator(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, dropout_rate):
        super(TextGenerator, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # Convierte las palabras (codigificadas como números) en una secuencia (vector) de un tamaño dado
        # Esto es lo que va a recordar secuencias de palabras que van bien juntas (que aparecen en nuestro corpus)
        #
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout_rate) # Esta es la capa que recuerda los datos que ya hemos procesado.
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)  # Descartar aleatoriamente algunas neuronas en cada etapa del proceso de aprendizaje para que demos oportunidad a otras neuronas a aprender más
        self.fc2 = nn.Linear(hidden_size, vocab_size) # Capa lineal de neuronas, que tiene tantas neuronas como token en nuestro corpus

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        x = self.relu(self.fc1(lstm_out))
        x = self.dropout(x)
        output = self.fc2(x)
        return output


generar_modelo = True

if(generar_modelo):
    # Configuración de hiperparámetros actualizada
    embedding_dim = 200
    hidden_size = 200
    learning_rate = 0.01
    num_epochs = 500
    num_layers = 2  # Son las cpas de memoria que conectamos : LSTM
    dropout_rate = 0.3  # Ajusta según sea necesario


    # Inicializar el modelo, la función de pérdida y el optimizador
    model = TextGenerator(num_tokens, embedding_dim, hidden_size, num_layers, dropout_rate)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Entrenamiento del modelo con los nuevos hiperparámetros
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        output = model(input_sequence.unsqueeze(0))
        loss = criterion(output.view(-1, num_tokens), target_sequence)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


    # Guardar el modelo
    torch.save(model, "model.pth")
else:
    # Cargar el modelo preentrenado
    model = torch.load("model.pth")



# seed_text Es el texto del que partimos a la hora de generar un nuevo texto
# length Longitud del texto que vamos a generar
# Temperature: Lo creativo o poco creativo que va a ser el texto que vamos a generar
#     Valores muy bajos-> 0, implican que seremos poco creativos
#     Valores más altos implican que seremos muy creativos
def generate_text(model, seed_text, length=50, temperature=1.0):
    model.eval()
    with torch.no_grad():
        seed_sequence = [word_to_idx[word] for word in seed_text.split()]
        generated_sequence = seed_sequence.copy()

        for _ in range(length):
            input_tensor = torch.LongTensor(seed_sequence).unsqueeze(0)
            output_probs = model(input_tensor) # Obteniendo las probabilidades de que cada palabra del corpus encaje detrás de ese texto del que partimos
            output_probs = output_probs[:, -1, :] / temperature  # 1
            # 1 lo deja como está (la probabilidad)
            # 0 aumenta la probabilidad
            # Número mayor que 1 me disminuye la probabilidad

            # De todas las palabras del corpus me quedo con 1, en base a esas probabilidades... que he trucado con la TEMPERATURA
            softmax_probs = nn.functional.softmax(output_probs, dim=-1)
            predicted_idx = torch.multinomial(softmax_probs, 1).item()

            # Añadimos la nueva palabra a la secuencia que llevo... y empezamos de nuevo... así como tantas palabras que me digan.
            generated_sequence.append(predicted_idx)
            seed_sequence = generated_sequence[-2:]


        # Hemos acabado generando una secuencia nueva de Números, que nos toca convertir ahora en palabras
        generated_text = ' '.join([idx_to_word[idx] for idx in generated_sequence])
        return generated_text

# Generar texto de prueba
seed_text = "Yogur"
for i in range(10):
    generated_text = generate_text(model, seed_text, length=1, temperature=1)
    print("Texto generado:", generated_text)

seed_text = "Coche"
for i in range(10):
    generated_text = generate_text(model, seed_text, length=1, temperature=1)
    print("Texto generado:", generated_text)
