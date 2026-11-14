# Machamba IA

Assistente agrícola com IA para pequenos produtores de Moçambique.
O README completo será escrito na Fase 7.

## Arranque rápido (desenvolvimento)

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
pytest
uvicorn app.main:app --reload
```
