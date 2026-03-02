# Проект: Прогнозирование стоимости страховых выплат
## Описание проекта
Цель проекта — разработать модель машинного обучения для предсказания
итоговой стоимости страхового возмещения (UltimateIncurredClaimCost)
на основе характеристик работника и параметров случая.
## Датасет
Используется датасет **Workers Compensation** (ID: 42876), содержащий
100,000 записей о страховых случаях.
## Требования
- Python 3.11+
- Зависимости из `requirements.txt`
## Установка и запуск
1. Клонируйте репозиторий:
   ```bash
   git clone https://github.com/user123qqqs/workers-compensation-streamlit
   cd workers-compensation-streamlit
2. Создать и активировать виртуальное окружение
   python -m venv .venv
   .venv\Scripts\activate
3) Установить зависимости
   pip install -r requirements.txt
4) Запустить приложение
   streamlit run app.py

