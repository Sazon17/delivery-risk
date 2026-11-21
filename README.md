# delivery-risk

Оценка риска того, что заказ маркетплейса будет доставлен позже обещанной даты.
Данные - Brazilian E-Commerce Public Dataset by Olist (9 таблиц, 99 441 заказ).
Прогноз делается в момент подтверждения оплаты.

Практические работы по дисциплине «Технологии и инструментарий машинного обучения»,
Сазонов В.С., ИМБО-02-24.

| Тег | Что сделано |
|-----|-------------|
| `pr2` | данные под DVC, разбиение на исходную часть и 5 порций, проверки Д1-Д13, разведочный анализ |

## Запуск

```bash
python -m venv .venv
.venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
pip install pip-tools
pip-sync requirements.txt
dvc pull                         # или положить 9 CSV Olist в data/raw/olist
dvc repro
pytest
```

Хранилище DVC в учебном варианте - каталог `../dvc-storage` рядом с репозиторием.

## Структура

```
configs/params.yaml        границы порций, пороги проверок
data/raw/olist.dvc         описатель исходных CSV (сами файлы - в DVC)
src/delivery_risk/data     сборка строки заказа, разбиение, проверки, история продавца
src/delivery_risk/eda      разведочный анализ
tests/                     тесты проверок данных
reports/                   validation.json, eda_summary.json
```

Этапы конвейера описаны в `dvc.yaml`: split -> validate -> eda.
