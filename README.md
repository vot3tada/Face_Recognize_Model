### Модель распознавания эмоций человека [RU]

Эта модель разработана для определения эмоций человека, обученная на наборе данных, доступным по ссылке: https://www.kaggle.com/datasets/astraszab/facial-expression-dataset-image-folders-fer2013). 

Кроме того, репозиторий содержит скрипт для запуска распознавания в реальном времени с использованием веб-камеры.

#### Установка

1. **Создание виртуальной среды для Python:**
   ```
   python3 -m venv venv
   source venv/bin/activate  # На Windows используйте `venv\Scripts\activate`
   ```

2. **Установка библиотек с помощью pip и requirements.txt:**
   ```
   pip install -r requirements.txt
   ```

3. **Запуск камеры:**
   ```
   python cam.py
   ```

#### Файлы в репозитории

- `cam.py`: Скрипт на Python для запуска камеры для распознавания в реальном времени.
- `face_model.ipynb`: Блокнот Jupyter, содержащий код для обучения модели распознавания лиц.

#### Набор данных

- **Размер:** 62.39 МБ
- **Разделение данных:** Набор данных разделен на тренировочный (80%), тестовый (10%) и валидационный (10%) наборы в формате ImageFolder.
- **Метки:**
  - 0: Злость
  - 1: Отвращение
  - 2: Страх
  - 3: Счастье
  - 4: Грусть
  - 5: Удивление
  - 6: Нейтральность

### The model for human emotion recognition [EN]

This model is designed for facial recognition, trained with the dataset available [here](https://www.kaggle.com/datasets/astraszab/facial-expression-dataset-image-folders-fer2013). Additionally, it includes real-time recognition capabilities using a webcam.

#### Installation

1. **Create Virtual Environment for Python:**
   ```
   python3 -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

2. **Install Libraries via Pip and Requirements.txt:**
   ```
   pip install -r requirements.txt
   ```

3. **Start Camera:**
   ```
   python cam.py
   ```

#### Files in Repository

- `cam.py`: Python script to start the camera for real-time recognition.
- `face_model.ipynb`: Jupyter notebook containing the code for training the face recognition model.

#### Dataset

- **Size:** 62.39 MB
- **Data Division:** The dataset is divided into train (80%), test (10%), and validation (10%) sets in the ImageFolder format.
- **Labels:**
  - 0: Angry
  - 1: Disgust
  - 2: Fear
  - 3: Happy
  - 4: Sad
  - 5: Surprise
  - 6: Neutral
