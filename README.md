# Diabetes Prediction — Kedro + FastAPI + Docker

Projeto de deployment production-ready para predição de diabetes.
Baseado nos slides da disciplina "Deployment: Production-Ready Data Science" (Insper, 2026).

---

## Arquitetura

```
Infrastructure
└── Docker Engine
    └── Docker Container
        └── uv Virtual Environment
            └── Kedro Project
                ├── Data Engineering Pipeline
                ├── Modelling Pipeline
                ├── Inference Pipeline
                └── FastAPI Layer  ←→  User (HTTP)
```

---

## Pipelines Kedro

### 1. Data Engineering
`raw_data → replace_zeros → impute (KNN) → cap_outliers → create_features → encode → fit_scaler → transform → master_table`

### 2. Modelling
`master_table → split → train → evaluate → production_model`

### 3. Inference
`raw_inference_data + production_scaler + production_model → prepare → predict → inference_predictions`

---

## Setup

```bash
# Instalar dependências
uv sync

# Ou com pip
pip install -e ".[api]"
```

---

## Executar Pipelines

```bash
# Engenharia de dados + treinamento (pipeline padrão)
uv run kedro run

# Pipeline individual
uv run kedro run --pipeline data_engineering
uv run kedro run --pipeline modelling
uv run kedro run --pipeline inference

# Visualizar o DAG
uv run kedro viz
```

---

## Servir como API (FastAPI)

```bash
uv run uvicorn diabetes.api:app --host 0.0.0.0 --port 8000
```

Acessar documentação automática: http://localhost:8000/docs

### Endpoints

| Método | Rota                  | Descrição                                      |
|--------|-----------------------|------------------------------------------------|
| GET    | `/health`             | Health check                                   |
| POST   | `/train`              | Dispara treino em background (async)           |
| GET    | `/train/{run_id}`     | Consulta status do treino                      |
| POST   | `/inference`          | Inferência online síncrona (JSON → predições)  |
| POST   | `/batch-inference`    | Inferência batch assíncrona (lê do catalog)    |

### Exemplo de inferência online

```bash
curl -X POST http://localhost:8000/inference \
  -H "Content-Type: application/json" \
  -d '{
    "instances": [
      {
        "Pregnancies": 6, "Glucose": 148, "BloodPressure": 72,
        "SkinThickness": 35, "Insulin": 0, "BMI": 33.6,
        "DiabetesPedigreeFunction": 0.627, "Age": 50
      }
    ]
  }'
```

---

## Docker

```bash
# Build e iniciar
docker compose up --build

# Iniciar em background
docker compose up -d --build

# Ver logs
docker compose logs -f

# Parar
docker compose down
```

A API estará disponível em http://localhost:8000

---

## Estrutura do Projeto

```
diabetes/
├── conf/
│   ├── base/
│   │   ├── catalog.yml        # Datasets declarativos (sem I/O no código)
│   │   └── parameters.yml     # Parâmetros separados da lógica
│   └── local/                 # Credenciais (não commitado)
├── data/
│   ├── 01_raw/                # Dados brutos
│   ├── 02_intermediate/       # Dados processados
│   ├── 03_primary/            # Master table + splits
│   ├── 06_models/             # Scaler + modelo treinado
│   └── 07_model_output/       # Predições + métricas
├── src/diabetes/
│   ├── pipelines/
│   │   ├── data_engineering/  # nodes.py + pipeline.py
│   │   ├── modelling/         # nodes.py + pipeline.py
│   │   └── inference/         # nodes.py + pipeline.py
│   ├── api.py                 # FastAPI layer
│   ├── pipeline_registry.py   # Registro dos pipelines
│   └── settings.py
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
└── pyproject.toml
```
