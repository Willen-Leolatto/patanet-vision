PataNet-Vision — Localizador de Pets (MVP)

Sistema de busca visual para identificar espécie/raça e recuperar imagens parecidas (para apoiar casos de animais perdidos).
Combina retrieval por conteúdo (CLIP + PCA + FAISS) com re-ranqueamento K-reciprocal e sinais auxiliares (head features e cor). Inclui API (FastAPI), demo web (Streamlit) e pipeline de avaliação e calibração.

✨ Recursos

API FastAPI com /docs (OpenAPI) e endpoints de busca e diagnóstico.

Demo Web (Streamlit) com upload, Top-K, percentuais, atributos e galeria de vizinhos.

Avaliação & Calibração (scripts/prepare_eval_and_grid.py):

curate: monta conjunto de validação por classe.

eval: mede acc@1/acc@3, retrieval@K, mAP, confusões.

grid: varre pesos/hiperparâmetros (sweep) e retoma de onde parou.

Gera metrics.json, per_class_summary.csv, confusion_matrix*.{csv,png} etc.

Index PCA + FAISS (CPU) para busca rápida.

Explicabilidade básica (atributos/“features” simples e score em %).

🏗️ Arquitetura (resumo)

Embeddings: openai/clip-vit-base-patch32 (ViT-B/32).

PCA (256D): redução de ruído e aceleração de kNN.

FAISS (CPU) para vizinhança aproximada.

K-reciprocal re-ranking: reforça vizinhos mutuamente próximos (krec_k1, krec_k2, krec_lambda).

Sinais auxiliares:

head: descritores “de cabeça/orelha” pré-computados da galeria.

color: histogramas/momentos de cor.

Fusão de scores:

score_final = w_krec * score_krecip
            + w_head * score_head
            + w_color * score_color


Intuição dos pesos

w_krec domina pois capta estrutura topológica dos vizinhos.

w_head ajuda em raças de formato facial/orelha característicos.

w_color estabiliza sob variação de pose/iluminação (peso menor para evitar viés).

📦 Requisitos

Windows 10/11 ou Linux

Python 3.10/3.11 (recomendado)

Git

(Windows) pode exigir Microsoft C++ Build Tools (para libs nativas)

Instalação
Windows (PowerShell)
git clone https://seu-repo.git patanet-vision
cd patanet-vision
python -m venv .venv
.\.venv\Scripts\Activate
pip install --upgrade pip
pip install -r requirements.txt
# Se necessário:
# pip install faiss-cpu==1.7.4
# pip install requests

Linux/macOS (Bash)
git clone https://seu-repo.git patanet-vision
cd patanet-vision
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

🗂️ Estrutura esperada
patanet-vision/
├─ app/
│  ├─ main.py                # API FastAPI
│  └─ ...
├─ demo/
│  └─ app.py                 # Demo Streamlit
├─ data/
│  ├─ stanford_dogs/Images/  # galeria/datasets
│  └─ oxford_pets/images/    # opcional
├─ index/
│  ├─ index_pca.faiss
│  ├─ gallery_pca.npy
│  └─ head_index.npy / head_lookup.json
├─ eval/                     # conjunto de validação
├─ outputs/
│  ├─ eval_results.csv
│  ├─ grid_*.csv
│  ├─ metrics.json
│  ├─ per_class_summary.csv
│  ├─ confusion_matrix.csv / confusion_matrix_top25.png
│  └─ refine_*.csv / sweep_*.csv
└─ scripts/
   └─ prepare_eval_and_grid.py

⚙️ Configuração de modelo (exemplo de /version)
{
  "model": "openai/clip-vit-base-patch32",
  "use_fast": true,
  "device": "cpu",
  "ntotal": 26111,
  "dim": 256,
  "gallery_pca": true,
  "head_index": true,
  "head_lookup": 26111
}

🚀 Subir a API
Desenvolvimento (hot reload)
uvicorn app.main:app --reload --port 8000

Produção/Paralelo (recomendado para grid/eval)
uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 4


Docs Swagger: http://127.0.0.1:8000/docs

OpenAPI JSON: http://127.0.0.1:8000/openapi.json

Versão/Estado: http://127.0.0.1:8000/version

Rotas
Método	Rota	Descrição
GET	/version	Metadados: modelo, index, classes, flags
POST	/search	Busca Top-K + diagnóstico/atributos
Parâmetros de /search
Nome	Tipo	Default	Descrição
file	file	—	Imagem (multipart/form-data)
k	int	5	Top-K vizinhos
use_head	int	1	Usa sinal head (0/1)
use_color	int	1	Usa sinal de cor (0/1)
w_krec	float	0.60	Peso K-reciprocal
w_head	float	0.25	Peso head
w_color	float	0.15	Peso color
krec_k1	int	20	K1 re-ranking
krec_k2	int	6	K2 re-ranking
krec_lambda	float	0.30	Mistura com distância original
return_diagnostics	int	1	Retorna scores parciais/atributos/espécie

Exemplo:

curl -X POST "http://127.0.0.1:8000/search?k=5&use_head=1&use_color=1&w_krec=0.6&w_head=0.25&w_color=0.15&krec_k1=20&krec_k2=6&krec_lambda=0.3&return_diagnostics=1" \
  -F "file=@teste.jpg"

🖥️ Demo Web (Streamlit)
streamlit run demo/app.py


Mostra Top-K com percentuais, diagnóstico e atributos simples.

Ajustado para use_container_width=True.

🧪 Avaliação / Grid / Retomada

Script: scripts/prepare_eval_and_grid.py (suporta resume e merge).

1) Curar conjunto de validação
python scripts/prepare_eval_and_grid.py curate ^
  --per-class 20 ^
  --dst eval

2) Avaliar (retomável)
# API deve estar rodando
python scripts/prepare_eval_and_grid.py eval ^
  --api http://127.0.0.1:8000 ^
  --root eval ^
  --max-images 1500 ^
  --workers 4 ^
  --timeout 60 ^
  --sleep-between 0.0 ^
  --out-csv outputs/eval_results.csv ^
  --resume


Gera: metrics.json, per_class_summary.csv, confusion_matrix*.{csv,png}, top_confusions.csv, hard_classes.csv.

3) Grid (calibração rápida de pesos/param)
python scripts/prepare_eval_and_grid.py grid ^
  --api http://127.0.0.1:8000 ^
  --root eval ^
  --max-images 600 ^
  --workers 4 ^
  --timeout 60 ^
  --sleep-between 0.0 ^
  --objective mixed ^
  --sweep "w_krec=[0.4,0.6,0.8];w_head=[0.1,0.2];w_color=[0.05,0.1];krec_k1=[12,20,30];krec_k2=[4,6,8];krec_lambda=[0.3,0.5]" ^
  --out outputs/grid_round1.csv


Pode interromper e retomar. Linhas já concluídas são ignoradas quando --out é reutilizado.

4) Consolidação & limpeza

Consolidar eval_results.csv + grid_*.csv.

Arquivar ou remover parciais (refine_*, sweep_*) para reduzir espaço.

Manter apenas índices essenciais em index/ (evitar versionar datasets).

👥 Boas práticas (casos reais)

Envie várias fotos (ângulos, luzes diferentes).

Evite filtros/zoom exagerado.

Centralize o animal e recorte distrações.

Use Top-K 8–12 em buscas difíceis.

Cruce com metadados (local, data, porte, cor).

🌐 Acesso rápido

API (Swagger): http://127.0.0.1:8000/docs

Versão/estado: http://127.0.0.1:8000/version

Demo Web: streamlit run demo/app.py

🧹 O que versionar no Git?

✅ Código, scripts e configs.
✅ Artefatos de índice pequenos/estáveis (se couber).
❌ Datasets/crops/imagens de usuário → usar DVC/LFS ou S3/Drive.
❌ Parciais de varredura (opcional manter só consolidado).

.gitignore sugerido:

data/
eval/
outputs/*.png
outputs/*_partial*.csv
outputs/refine_*.csv
outputs/sweep_*.csv
index/*.bin
*.pt
*.ckpt
*.onnx
.DS_Store
.venv/
__pycache__/

🛠️ Troubleshooting

WinError 10061 / conexão recusada → API não está rodando (suba o Uvicorn).

ModuleNotFoundError: requests → pip install requests.

FAISS/Torch em Windows → use versões estáveis (faiss-cpu==1.7.4, Torch LTS).

Streamlit use_column_width → já migrado para use_container_width=True.

Sem métricas → confirme outputs/eval_results.csv e outputs/metrics.json.

📊 Metas de qualidade (MVP)

Imagens ideais: acc@1 ≥ 80% nas raças mais comuns.

Cenários difíceis: apoiar em Top-K, atributos e cor.

Use per_class_summary.csv e top_confusions.csv para priorizar melhorias.

📅 Roadmap

 Augmentations leves no índice (robustez a iluminação).

 Atributos explicativos extras (focinho, patas, cauda).

 Cache de respostas no demo.

 Dockerfile + docker-compose (API + demo).

 Deploy econômico (Railway/Render) da API + índice estático.

👨‍💻 Comandos úteis
# Ativar venv
.\.venv\Scripts\Activate

# Subir API
uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 4

# Docs
# http://127.0.0.1:8000/docs

# Demo Web
streamlit run demo/app.py

# Curar conjunto de validação
python scripts/prepare_eval_and_grid.py curate --per-class 20 --dst eval

# Avaliar (retomável)
python scripts/prepare_eval_and_grid.py eval --api http://127.0.0.1:8000 --root eval --max-images 1500 --workers 4 --timeout 60 --sleep-between 0.0 --out-csv outputs/eval_results.csv --resume

# Grid (rápido)
python scripts/prepare_eval_and_grid.py grid --api http://127.0.0.1:8000 --root eval --max-images 600 --workers 4 --timeout 60 --sleep-between 0.0 --objective mixed --sweep "w_krec=[0.4,0.6,0.8];w_head=[0.1,0.2];w_color=[0.05,0.1];krec_k1=[12,20,30];krec_k2=[4,6,8];krec_lambda=[0.3,0.5]" --out outputs/grid_round1.csv


Licenças: respeite as licenças dos datasets (Stanford Dogs, Oxford-IIIT Pets) e LGPD/GDPR para imagens de usuários.