# BodyScan 3D — ML Inference Service

Microserviço FastAPI que recebe 2 fotos (frontal + lateral) de silhueta corporal e retorna composição corporal estimada via Machine Learning.

## Como funciona

O pipeline tem 2 estágios:

1. **Stage A (CNN):** EfficientNet-B0 com cross-attention fusion recebe as 2 silhuetas (224x224 grayscale) e prediz 14 medidas corporais (pescoço, ombro, tórax, cintura, quadril, bíceps, antebraço, coxa, panturrilha, punho).

2. **Stage B (Ensemble):** XGBoost + LightGBM + Ridge Regression recebe as medidas + metadados (idade, altura, peso, gênero) com 50+ features engineered e prediz o percentual de gordura corporal.

A partir do body fat %, o serviço calcula: massa gorda, massa magra, massa muscular, massa óssea, % de água, TMB (Mifflin-St Jeor) e GET.

## Estrutura

```
bodyscan-service/
├── app/
│   ├── __init__.py
│   ├── config.py          # Pydantic Settings (env vars)
│   ├── dependencies.py    # Injeção dos modelos carregados
│   ├── inference.py       # Pipeline ML (Stage A + Stage B)
│   ├── main.py            # FastAPI app, rotas, CORS, lifespan
│   └── models.py          # Pydantic schemas (request/response)
├── Dockerfile             # Build de produção
├── requirements.txt       # Dependências Python
├── .env.example           # Exemplo de variáveis de ambiente
└── .dockerignore
```

## Modelos ML

Os modelos são baixados automaticamente durante o build do Docker a partir do BunnyCDN:

| Modelo | Arquivo | Tamanho | Descrição |
|--------|---------|---------|-----------|
| Stage A | `stage_a_v2_best.pth` | ~184 MB | CNN EfficientNet-B0 (14 medidas) |
| Stage B (M) | `stage_b__male.pkl` | ~1 MB | Ensemble masculino (body fat %) |
| Stage B (F) | `stage_b__female.pkl` | ~1.4 MB | Ensemble feminino (body fat %) |

URL base: `https://ds-course.b-cdn.net/`

## Deploy com Docker

### 1. Build da imagem

```bash
git clone https://github.com/thiagoferreira123/bodyscan-3d.git
cd bodyscan-3d
docker build -t bodyscan-service .
```

O build leva ~5-10 minutos (instala PyTorch CPU + baixa modelos do CDN).

### 2. Rodar o container

```bash
docker run -d \
  --name bodyscan \
  -p 8000:8000 \
  -e CORS_ORIGINS="https://seudominio.com.br" \
  -e LOG_LEVEL=info \
  --memory=2g \
  --restart unless-stopped \
  bodyscan-service
```

### 3. Verificar se está rodando

```bash
curl http://localhost:8000/health
# Resposta esperada: {"status":"ok","models_loaded":true}
```

## Variáveis de ambiente

| Variável | Default | Descrição |
|----------|---------|-----------|
| `HOST` | `0.0.0.0` | Endereço de bind |
| `PORT` | `8000` | Porta do servidor |
| `CORS_ORIGINS` | `http://localhost:3000,http://localhost:3001` | Origins permitidas (separadas por vírgula) |
| `MODELS_DIR` | `./models` | Diretório dos modelos (já configurado na imagem) |
| `LOG_LEVEL` | `info` | Nível de log (`debug`, `info`, `warning`, `error`) |

## Requisitos do servidor

| Recurso | Mínimo | Recomendado |
|---------|--------|-------------|
| RAM | 2 GB | 4 GB |
| CPU | 1 core | 2 cores |
| Disco | 3 GB | 5 GB |
| GPU | Não precisa | Não precisa |
| Internet | Apenas no build (baixa modelos) | — |

O serviço roda em **CPU-only** (PyTorch CPU). Não precisa de GPU.

## API

### `GET /health`

Health check. Retorna status e se os modelos estão carregados.

```json
{"status": "ok", "models_loaded": true}
```

### `POST /api/v1/analyze`

Analisa composição corporal a partir de 2 imagens.

**Content-Type:** `multipart/form-data`

| Campo | Tipo | Obrigatório | Descrição |
|-------|------|-------------|-----------|
| `front_image` | file | sim | Foto frontal (JPEG, PNG ou WebP) |
| `side_image` | file | sim | Foto lateral (JPEG, PNG ou WebP) |
| `age` | int | sim | Idade (10-120) |
| `height_cm` | float | sim | Altura em cm (50-250) |
| `weight_kg` | float | sim | Peso em kg (30-300) |
| `gender` | string | sim | `M` ou `F` |

**Exemplo com curl:**

```bash
curl -X POST http://localhost:8000/api/v1/analyze \
  -F "front_image=@foto_frontal.jpg" \
  -F "side_image=@foto_lateral.jpg" \
  -F "age=30" \
  -F "height_cm=175" \
  -F "weight_kg=80" \
  -F "gender=M"
```

**Resposta:**

```json
{
  "body_fat_pct": 18.5,
  "lean_mass_kg": 65.2,
  "fat_mass_kg": 14.8,
  "muscle_mass_kg": 36.51,
  "bone_mass_kg": 7.82,
  "water_pct": 59.42,
  "bmr": 1822.36,
  "tdee": 2186.83,
  "waist_cm": 84.3,
  "measurements": {
    "neck": 38.2,
    "shoulder": 112.5,
    "chest": 98.1,
    "waist": 84.3,
    "hip": 96.7,
    "bicep_left": 32.1,
    "bicep_right": 32.4,
    "forearm_left": 27.3,
    "forearm_right": 27.5,
    "thigh_left": 55.2,
    "thigh_right": 55.8,
    "calf_left": 37.1,
    "calf_right": 37.3,
    "wrist": 17.2
  },
  "calculated_metrics": {
    "bmi": 26.12,
    "whr": 0.48,
    "bsi": 0.08,
    "bai": 6.94,
    "ci": 1.24,
    "ponderal_index": 14.92
  },
  "model_predictions": {
    "xgboost": 18.3,
    "lightgbm": 18.7,
    "ridge": 18.5
  },
  "ensemble_weights": {
    "xgboost": 0.4,
    "lightgbm": 0.35,
    "ridge": 0.25
  },
  "model_versions": {
    "stage_a": "v2.0",
    "stage_b": "v6.3_hybrid"
  }
}
```

## Deploy em plataformas

### Coolify

1. Criar novo recurso → Application → GitHub repo
2. Build pack: **Dockerfile**
3. Porta: **8000**
4. Health check: `GET /health` na porta 8000
5. Limites: memory **4096 MB**, CPUs **2**
6. Adicionar env vars: `CORS_ORIGINS`, `LOG_LEVEL`

### Railway / Render / Fly.io

Qualquer plataforma que suporte Dockerfile funciona. Configurar:
- Porta exposta: `8000`
- Health check: `/health`
- Memória mínima: 2 GB

### VPS manual (Ubuntu)

```bash
# Instalar Docker
curl -fsSL https://get.docker.com | sh

# Clone, build e run
git clone https://github.com/thiagoferreira123/bodyscan-3d.git
cd bodyscan-3d
docker build -t bodyscan-service .
docker run -d --name bodyscan -p 8000:8000 \
  -e CORS_ORIGINS="https://seudominio.com.br" \
  --memory=2g --restart unless-stopped \
  bodyscan-service

# Verificar
curl http://localhost:8000/health
```

Para HTTPS, coloque um **reverse proxy** (Nginx/Caddy/Traefik) na frente com certificado SSL.

## Integração com o backend DietSystem

O backend NestJS chama este serviço via env var:

```env
BODYSCAN_SERVICE_URL=https://bodyscan.seudominio.com.br
```

O fluxo é: **Frontend → Backend NestJS → BodyScan Service → Resposta salva no banco**.

O frontend nunca chama este serviço diretamente.

## Desenvolvimento local

```bash
# Criar venv
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

# Instalar dependências
pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# Baixar modelos manualmente
mkdir -p models
curl -o models/stage_a_v2_best.pth https://ds-course.b-cdn.net/stage_a_v2_best.pth
curl -o models/stage_b__male.pkl https://ds-course.b-cdn.net/stage_b__male.pkl
curl -o models/stage_b__female.pkl https://ds-course.b-cdn.net/stage_b__female.pkl

# Copiar env
cp .env.example .env

# Rodar
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
