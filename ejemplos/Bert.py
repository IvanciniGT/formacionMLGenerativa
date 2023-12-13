# Para descargar un modelo ya creado y con pre-entrenamiento con el que trabajar
from transformers import BertTokenizer, BertForQuestionAnswering
# BertForQuestionAnswering:
    # es un modelo pre-entrenado optimizado para respuesta a preguntas
# BertForTokenClassification:
    # es un modelo pre-entrenado optimizado para clasificación de tokens
# BertForSecuenceClassification:
    # es un modelo pre-entrenado optimizado para clasificación de tokens
# BertForNextSentencePrediction:
    # es un modelo pre-entrenado optimizado para predicción de la siguiente frase
# Vamos a usar torch para afinar el modelo que descargamos
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

import json


# Definir la clase del conjunto de datos
class QADataset(Dataset):
    def __init__(self, tokenizer, filepath):
        self.tokenizer = tokenizer
        self.data = []

        with open(filepath, 'r') as file:
            raw_data = json.load(file)
            for item in raw_data:
                self.data.append((
                    item['question'],
                    item['context'],
                    item['answer']
                ))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question, context, answer = self.data[idx]

        # Codificar la pregunta y el contexto juntos
        encoded = self.tokenizer.encode_plus(question, context, add_special_tokens=True, return_tensors="pt",
                                             max_length=512, truncation=True, padding="max_length")

        # Encontrar las posiciones de inicio y fin de la respuesta
        # Para ello debemos asegurarnos que la respuesta aparece literalmente en el contexto
        start_idx = encoded.input_ids[0].tolist().index(self.tokenizer.encode(answer, add_special_tokens=False)[0])
        end_idx = start_idx + len(self.tokenizer.encode(answer, add_special_tokens=False)) - 1

        return encoded.input_ids, encoded.attention_mask, torch.tensor([start_idx]), torch.tensor([end_idx])


# Cargar el tokenizer y el modelo
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model = BertForQuestionAnswering.from_pretrained('bert-base-multilingual-cased')

# Preparar el conjunto de datos y el DataLoader
dataset = QADataset(tokenizer, 'preguntas.json')
data_loader = DataLoader(dataset, batch_size=2, shuffle=True)

# Configurar el optimizador
optimizer = AdamW(model.parameters(), lr=5e-5)

# Entrenamiento
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if(torch.bachends.mps.is_available()) :
    device = torch.device('mps')
# Hay un monton de devices disponibles en ttorch, por ejemplo
# para mac: mps: Multi-Processing Service

# Una cosa es el motor que se utiliza para hacer cálculos: CPU, GPU, TPU, etc.
# Y otra cosa es la memoria donde voy a colocar los datos con los que alimentar a ese motor
# Habitualmente las GPUs y TPUs tienen su propia memoria integrada, pero la CPU lleva la RAM aparte
# Lo que vamos a hacer es ir colocando los datos con los que trabajo en la memoria del motor que vamos a usar

# Hay que tener cuidado de no estar moviendo constantemente los datos de un sitio a otro, porque eso es muy costoso... 
# y retrasa mucho el proceso de entrenamiento

model.to(device) # Nuestro modelo va a usar el motor de calculo del dispositivo indicado.
# Al final, o alquilo una máquina en un cloud, con una GPU (NVidia), en Google si alquilo una máquina puedo usar TPU
# O tengo una máquina monstruosa en la empresa que la dedico a estos menesteres.

for epoch in range(3):  # Número de épocas
    model.train()
    for input_ids, attention_mask, start_positions, end_positions in data_loader:
        input_ids = input_ids.to(device).squeeze(1) # Movemos los datos de la memoria RAM a la memoria del dispositivo concreto que vamos a usar
        attention_mask = attention_mask.to(device).squeeze(1)
        start_positions = start_positions.to(device)
        end_positions = end_positions.to(device)

        optimizer.zero_grad() ## Arranco iteracion

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, start_positions=start_positions,
                        end_positions=end_positions) # Genero resultados
        loss = outputs.loss # Calculo perdida

        loss.backward()     # Y retropropago (calculo los pesos nuevos)
        optimizer.step()    # Y actualizo los pesos

    print(f"Epoch {epoch} completed")

# Poner el modelo en modo de evaluación
model.eval()

def answer_question(question, context):
    # Mover el modelo al dispositivo adecuado (GPU si está disponible, de lo contrario CPU)
    model.to(device)

    # Codificar la entrada para el modelo
    inputs = tokenizer.encode_plus(question, context, add_special_tokens=True, return_tensors="pt", max_length=512, truncation=True)

    # Mover los tensores de entrada al mismo dispositivo que el modelo
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    # Hacer la predicción
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    # Encontrar la parte del texto de entrada que corresponde a la respuesta
    answer_start_scores, answer_end_scores = outputs.start_logits, outputs.end_logits
    answer_start = torch.argmax(answer_start_scores)
    answer_end = torch.argmax(answer_end_scores) + 1

    # Asegurarse de mover los tensores de vuelta a la CPU para la conversión de tokens a string
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[0].to('cpu')[answer_start:answer_end]))

    return answer



# Ejemplo de uso
context = "España es un país en la península ibérica. Su capital es Madrid."
question = "Capital de España"

print(answer_question(question, context))
