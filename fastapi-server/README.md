# Запуск

1. Установить зависимости:
   ```shell
   poetry install
   ```
2. Создать инференсы преобразователей данных и модели:
   ```shell
   poetry run train_and_save_inferences.py *путь к файлу с данными*
   ```
3. Запустить сервер:
   ```shell
   poetry run uvicorn server:app
   ```