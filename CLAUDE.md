# Machamba IA — Instruções para o Claude Code

## O que é este projecto
Assistente agrícola com IA para pequenos produtores de Moçambique, acessível por um bot de Telegram (WhatsApp numa fase futura).

**Problema:** os pequenos agricultores perdem colheitas porque não identificam pragas e doenças a tempo, há poucos técnicos agrícolas (extensionistas) para muitas famílias, e a informação meteorológica não lhes chega de forma simples.

**Solução:** o agricultor envia uma foto ou uma pergunta e recebe em segundos um diagnóstico e uma recomendação baseados em manuais técnicos. Os casos incertos vão para um técnico humano através de um painel web. O sistema envia também alertas meteorológicos diários.

**Objectivo:** construir um sistema útil de verdade para os agricultores. O autor quer perceber e dominar cada parte do sistema. **Por isso, explica sempre as decisões técnicas que tomares, em linguagem simples.**

## Funcionalidades do MVP (por esta ordem)
1. Diagnóstico por foto: classificador de imagens com nível de confiança
2. Perguntas e respostas com RAG sobre manuais da FAO e do IIAM, sempre com a fonte indicada
3. Encaminhamento dos casos incertos para um painel do técnico (human-in-the-loop)
4. Alertas meteorológicos diários por distrito

Fora do MVP (não implementar sem pedido): áudio/Whisper, mapa de pragas, WhatsApp, pagamentos M-Pesa.

## Stack
- Python 3.11+, FastAPI, Pydantic, Uvicorn
- Visão: PyTorch + timm (EfficientNet-B0), treinado no Google Colab
- LLM: API do Claude (SDK `anthropic`), atrás de uma interface para permitir trocar de fornecedor
- Embeddings: sentence-transformers `intfloat/multilingual-e5-small`
- Base vectorial: ChromaDB (persistente em disco)
- PDFs: pymupdf
- Base de dados: SQLite + SQLAlchemy
- Bot: python-telegram-bot
- Meteorologia: Open-Meteo (sem chave)
- Agendamento: APScheduler
- Painel: Streamlit
- Testes: pytest
- Deploy: Docker + docker-compose

## Estrutura de pastas
```
machamba-ia/
├── CLAUDE.md
├── README.md
├── .env.example
├── requirements.txt
├── docker-compose.yml
├── notebooks/        # treino do modelo (corre no Colab)
├── models/           # modelo treinado (.pt) — NÃO vai para o Git
├── data/
│   ├── manuais/      # PDFs para o RAG
│   └── chroma/       # base vectorial — NÃO vai para o Git
├── app/
│   ├── main.py       # app FastAPI
│   ├── config.py     # lê variáveis do .env
│   ├── api/          # endpoints
│   ├── vision/       # carregar modelo e prever
│   ├── rag/          # ingestão e pesquisa
│   ├── llm/          # cliente do LLM (interface + implementação Claude)
│   ├── bot/          # bot do Telegram
│   ├── alerts/       # meteorologia e agendamento
│   └── db/           # modelos e sessão SQLAlchemy
├── dashboard/        # painel Streamlit do técnico
└── tests/
```

## Regras de design importantes
- **Limites de confiança da visão:** ≥ 0.80 responde normalmente; 0.50–0.80 responde com cautela e pede outra foto; < 0.50 não diagnostica e encaminha para o técnico. Os valores ficam em `config.py`.
- **Filtro de entrada:** antes de classificar, verificar se a imagem é de uma das culturas suportadas (milho, mandioca, caju, tomate). Se não for, dizer que ainda não é suportada.
- **RAG sem alucinações:** o LLM só responde com base nos trechos encontrados. Se a semelhança máxima ficar abaixo do limite configurado, responder que não há informação confiável e encaminhar o caso. Indicar sempre o documento de origem.
- **Human-in-the-loop:** cada caso encaminhado fica na tabela `casos` com o estado `pendente`. A resposta do técnico é enviada ao agricultor, e a etiqueta corrigida fica guardada para voltar a treinar o modelo.
- **Separação de camadas:** a lógica (vision, rag, llm) não depende do Telegram. O bot só chama serviços, para que adicionar o WhatsApp seja apenas uma nova camada.
- **Linguagem para o agricultor:** português simples, frases curtas, sem jargão técnico.

## Regras de trabalho
1. **Antes de escrever código, apresenta um plano curto** e espera pela aprovação.
2. Trabalha **uma fase de cada vez** (ver PROMPTS.md). Não avances para a fase seguinte sem pedido.
3. Código com type hints e docstrings curtas. Nomes de variáveis e funções em inglês; comentários, mensagens ao utilizador e documentação em português.
4. Escreve testes pytest para cada endpoint e serviço. Corre os testes antes de dar a fase por concluída.
5. **Nunca coloques segredos no código.** Tudo vem do `.env`, e mantém o `.env.example` actualizado. Garante que `.env`, `models/`, `data/chroma/` e os datasets estão no `.gitignore`.
6. No fim de cada fase:
   - resume o que foi feito e porquê
   - explica os 2–3 conceitos mais importantes da fase
   - sugere uma mensagem de commit
7. Quando algo tiver de ser feito manualmente pelo autor (criar o bot no BotFather, correr o notebook no Colab, descarregar o dataset), diz exactamente o que fazer, passo a passo.
8. Prefere soluções simples e legíveis a soluções "espertas".

## Dataset
Crop Pest and Disease Detection (Gana): https://data.mendeley.com/datasets/bwh3zbpkpv
4 culturas (caju, mandioca, milho, tomate), 22 classes. O autor descarrega-o manualmente para o Google Drive e o treino é feito no Colab.

## Estado actual
(Actualiza esta secção no fim de cada fase.)
- [x] Fase 0 — Estrutura do projecto
- [ ] Fase 1 — Notebook de treino do modelo
- [ ] Fase 2 — Serviço de diagnóstico
- [ ] Fase 3 — RAG
- [ ] Fase 4 — Bot do Telegram
- [ ] Fase 5 — Painel do técnico
- [ ] Fase 6 — Alertas meteorológicos
- [ ] Fase 7 — Docker, deploy e README
