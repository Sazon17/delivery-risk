# delivery-risk

Оценка риска того, что заказ маркетплейса будет доставлен позже обещанной даты.
Данные - Brazilian E-Commerce Public Dataset by Olist (9 таблиц, 99 441 заказ).
Прогноз делается в момент подтверждения оплаты.

Практические работы по дисциплине «Технологии и инструментарий машинного обучения»,
Сазонов В.С., ИМБО-02-24.

| Тег | Что сделано |
|-----|-------------|
| `pr2` | данные под DVC, разбиение на исходную часть и 5 порций, проверки Д1-Д13, разведочный анализ |
| `pr3` | конвейер признаков, сравнение моделей по помесячным блокам, подбор, итоговая модель `models/model_v1` |

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

`requirements.txt` собран командой `pip-compile --generate-hashes` под Python 3.11 на Linux.
Если на Windows pip-sync сообщит о недостающем пакете, достаточно `pip install -r requirements.in`.

Хранилище DVC в учебном варианте - каталог `../dvc-storage` рядом с репозиторием.

## Структура

```
configs/params.yaml        границы порций, пороги проверок
data/raw/olist.dvc         описатель исходных CSV (сами файлы - в DVC)
src/delivery_risk/data     сборка строки заказа, разбиение, проверки, история продавца
src/delivery_risk/eda      разведочный анализ
src/delivery_risk/features prepare.py - любой файл заказов в таблицу для модели,
                           pipeline.py - признаки и Pipeline (признаки -> предобработка -> модель)
src/delivery_risk/models   split.py - выборки, блоки, метрики; experiments.py - сравнение;
                           train.py - выбор и обучение итоговой модели; plots.py - графики
tests/                     тесты проверок данных и признаков
reports/                   validation.json, eda_summary.json, experiments.csv, pr3/
models/model_v1.json       описание итоговой модели (сама модель - в DVC)
```

Этапы конвейера описаны в `dvc.yaml`:
split -> validate -> eda, features -> experiments -> train -> plots.

Подготовить одну порцию для модели:

```bash
python src/delivery_risk/features/prepare.py data/incoming/batch_01.csv
```

Журнал экспериментов MLflow ведётся в локальной базе `mlflow.db`:
`mlflow ui --backend-store-uri sqlite:///mlflow.db`.
