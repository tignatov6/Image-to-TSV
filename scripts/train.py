import torch
from PIL import Image
from transformers import VisionEncoderDecoderModel, DonutProcessor
from peft import LoraConfig, get_peft_model
import os

# --- 1. Конфигурация ---
# Укажите пути к вашим данным и куда сохранить модель
IMAGE_PATH = "data/processed/images/table.png"
LABEL_PATH = "data/processed/labels/table.txt"
OUTPUT_DIR = "models/image-to-tsv-v1"

# Название базовой модели Donut
PRETRAINED_MODEL_NAME = "naver-clova-ix/donut-base"

# Параметры обучения (для одного примера много не нужно)
NUM_TRAIN_EPOCHS = 5000
LEARNING_RATE = 3e-5


def train_on_single_example():
    """
    Функция для дообучения модели Donut на одном примере "изображение + текст".
    """
    # --- 2. Проверка наличия данных ---
    if not os.path.exists(IMAGE_PATH) or not os.path.exists(LABEL_PATH):
        print("Ошибка: Не найдены файлы данных!")
        print(f"Проверьте, что существует файл изображения: {IMAGE_PATH}")
        print(f"И файл с текстом: {LABEL_PATH}")
        return

    # --- 3. Настройка устройства (GPU или CPU) ---
    # --- Умный выбор устройства (GPU/CPU) ---
    if torch.cuda.is_available():
        device = "cuda"
    else:
      try:
        import torch_directml
        device = torch_directml.device()
      except:
        device = "cpu"

    print(f"Используемое устройство: {device}")

    # --- 4. Загрузка модели и процессора ---
    print(f"Загрузка базовой модели: {PRETRAINED_MODEL_NAME}...")
    model = VisionEncoderDecoderModel.from_pretrained(PRETRAINED_MODEL_NAME, use_safetensors=True)
    processor = DonutProcessor.from_pretrained(PRETRAINED_MODEL_NAME, use_safetensors=True)
    
    model.to(device)
    print("Модель и процессор успешно загружены.")

    model.config.decoder_start_token_id = processor.tokenizer.bos_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id    

    # --- 5. Применение LoRA для эффективного дообучения ---
    # Это позволяет обучать только небольшую часть весов, экономя память
    lora_config = LoraConfig(
        r=16, # Ранг (чем больше, тем больше обучаемых параметров)
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"], # Слой, к которому применяется LoRA
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    print("\nLoRA адаптер применен к модели. Обучаемые параметры:")
    model.print_trainable_parameters()

    print("\nЗамораживаем параметры визуального энкодера...")
    for param in model.encoder.parameters():
        param.requires_grad = False
    print("Энкодер заморожен.")

    # --- 6. Подготовка данных ---
    print("\nПодготовка одного обучающего примера...")
    # Загружаем изображение
    image = Image.open(IMAGE_PATH).convert("RGB")
    
    # Читаем целевой текст из файла
    with open(LABEL_PATH, 'r', encoding='utf-8') as f:
        target_text = f.read()

    # Используем процессор для преобразования текста в токены (labels)
    # Это то, что модель должна научиться генерировать
    pixel_values = processor(image, return_tensors="pt").pixel_values
    decoder_input_ids = processor.tokenizer(
        target_text, add_special_tokens=False, max_length=512, padding="max_length", truncation=True, return_tensors="pt"
    ).input_ids
    
    # Для Donut, labels - это то же самое, что и decoder_input_ids
    labels = decoder_input_ids.clone()
    # Во время обучения нам не нужно предсказывать padding-токены
    labels[labels == processor.tokenizer.pad_token_id] = -100

    # Перемещаем тензоры на нужное устройство
    pixel_values = pixel_values.to(device)
    labels = labels.to(device)
    print("Данные успешно подготовлены и перемещены на устройство.")

    # --- 7. Настройка оптимизатора ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # --- 8. Цикл обучения ---
    print("\n--- Начало обучения ---")
    model.train() # Переводим модель в режим обучения

    for epoch in range(NUM_TRAIN_EPOCHS):
        encoder_outputs = model.encoder(pixel_values=pixel_values)

        encoder_outputs.last_hidden_state = encoder_outputs.last_hidden_state.detach()

        outputs = model(encoder_outputs=encoder_outputs, labels=labels)

        # Получаем loss (ошибку)
        loss = outputs.loss

        # Выводим значение loss, пока он еще на GPU
        print(f"Эпоха: {epoch + 1}/{NUM_TRAIN_EPOCHS}, Loss: {loss.item():.4f}")

        # --- ИСПРАВЛЕНИЕ: Гибридный обратный проход ---
        # 1. Перемещаем только сам loss на CPU
        loss_cpu = loss.to("cpu")

        # 2. Выполняем обратный проход на CPU, который может обработать сложный граф
        loss_cpu.backward()

        # Оптимизатор обновляет веса, которые остались на GPU, используя градиенты,
        # посчитанные на CPU.
        optimizer.step()
        optimizer.zero_grad()

    print("--- Обучение завершено ---")

    # --- 9. Сохранение модели ---
    print(f"\nСохранение модели в папку: {OUTPUT_DIR}")
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    model.save_pretrained(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    
    print("Модель и процессор успешно сохранены!")


if __name__ == "__main__":
    train_on_single_example()
