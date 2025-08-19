import torch
from PIL import Image
from transformers import VisionEncoderDecoderModel, DonutProcessor
import os

# --- 1. Конфигурация ---
# Путь к обученной модели
MODEL_PATH = "models/image-to-tsv-v1"

# Путь к изображению, которое будем распознавать
# ВАЖНО: Используем то же самое изображение, на котором обучались
IMAGE_TO_TEST = "data/processed/images/table.png"

def run_inference():
    """
    Загружает обученную модель и распознает текст на изображении.
    """
    # --- 2. Проверка наличия данных ---
    if not os.path.exists(MODEL_PATH) or not os.path.exists(IMAGE_TO_TEST):
        print("Ошибка: Не найдена папка с моделью или тестовое изображение!")
        print(f"Проверьте путь к модели: {MODEL_PATH}")
        print(f"И путь к изображению: {IMAGE_TO_TEST}")
        return
        
    # --- 3. Настройка устройства ---
    # --- Умный выбор устройства (GPU/CPU) ---
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.dml.is_available(): # Проверяем наличие DirectML
        device = torch.device("dml")
    else:
        device = "cpu"

    print(f"Используемое устройство: {device}")
    
    # --- 4. Загрузка обученной модели и процессора ---
    print(f"Загрузка модели из: {MODEL_PATH}...")
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_PATH) 
    processor = DonutProcessor.from_pretrained(MODEL_PATH) 
    
    model.to(device)
    print("Модель и процессор успешно загружены.")
    
    model.config.decoder_start_token_id = processor.tokenizer.bos_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id

    # --- 5. Подготовка изображения и запуск распознавания ---
    image = Image.open(IMAGE_TO_TEST).convert("RGB")
    
    # Готовим изображение для модели
    pixel_values = processor(image, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)
    
    # Генерируем текстовое представление
    # Для Donut нужно передать decoder_input_ids с начальным токеном
    task_prompt = "<s_i2t>" # специальный токен для задачи "image to text"
    decoder_input_ids = torch.tensor([[model.config.decoder_start_token_id]], device=device)
    decoder_input_ids = decoder_input_ids.to(device)

    print("\nРаспознавание...")
    # Увеличиваем max_length, чтобы хватило места для всего текста
    outputs = model.generate(
        pixel_values,
        decoder_input_ids=decoder_input_ids,
        max_length=1024,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        use_cache=True,
        bad_words_ids=[[processor.tokenizer.unk_token_id]],
        return_dict_in_generate=True,
    )
    
    # --- 6. Декодирование и вывод результата ---
    sequence = processor.batch_decode(outputs.sequences)[0]
    # Убираем специальные токены из вывода
    sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "").replace(task_prompt, "")
    
    print("\n--- РЕЗУЛЬТАТ РАСПОЗНАВАНИЯ ---")
    print(sequence.strip())

if __name__ == "__main__":
    run_inference()