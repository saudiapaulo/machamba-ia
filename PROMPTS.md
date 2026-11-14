# Prompts para o Claude Code — Machamba IA

Copia um prompt de cada vez para o Claude Code. Só passa ao seguinte quando a fase estiver a funcionar e tiveres feito commit.

---

## Fase 0 — Estrutura do projecto
```
Lê o CLAUDE.md. Vamos fazer a Fase 0.
Cria a estrutura de pastas, o requirements.txt, o .gitignore, o .env.example,
o app/config.py e um app/main.py FastAPI com um endpoint GET /health.
Escreve um teste para o /health.
Mostra-me primeiro o plano. Depois de implementares, diz-me como criar o
ambiente virtual, instalar as dependências e correr o servidor e os testes.
```

## Fase 1 — Notebook de treino (corre no Colab)
```
Fase 1. Cria notebooks/treino_modelo.ipynb para correr no Google Colab, que:
1. monta o Google Drive e lê o dataset do Gana (pergunta-me o caminho);
2. divide em treino/validação/teste (70/15/15), estratificado por classe;
3. aplica data augmentation no treino;
4. faz fine-tuning do EfficientNet-B0 com timm;
5. mostra a accuracy, um relatório por classe e a matriz de confusão;
6. guarda o modelo e um ficheiro labels.json.
Explica cada secção com células de texto em português.
No fim, diz-me passo a passo como o correr no Colab e onde pôr os ficheiros
gerados (models/).
```

## Fase 2 — Serviço de diagnóstico
```
Fase 2. O modelo treinado está em models/. Implementa:
- app/vision/: carregar o modelo uma vez no arranque e uma função predict(image)
  que devolve a cultura, a doença e a confiança;
- o filtro de entrada para imagens que não são das culturas suportadas;
- POST /diagnose, que recebe uma imagem e aplica os limites de confiança do CLAUDE.md;
- testes (usa um modelo falso/mock nos testes para não depender do ficheiro real).
Plano primeiro.
```

## Fase 3 — RAG
```
Fase 3. Vou colocar PDFs em data/manuais/. Implementa:
- app/rag/ingest.py: script que lê os PDFs, divide em chunks com sobreposição,
  gera embeddings com multilingual-e5-small e guarda no ChromaDB com
  metadados (ficheiro, página);
- app/llm/: interface LLMClient + implementação com a API do Claude;
- app/rag/service.py: pesquisa, limite de semelhança, prompt que proíbe
  inventar, resposta com a fonte;
- POST /ask;
- testes com mock do LLM.
Explica-me o que são chunks, embeddings e o limite de semelhança.
Diz-me também onde posso encontrar bons PDFs da FAO/IIAM sobre estas pragas.
```

## Fase 4 — Base de dados e bot do Telegram
```
Fase 4. Implementa:
- app/db/: modelos SQLAlchemy Agricultor (telegram_id, nome, distrito) e
  Caso (tipo, conteúdo, caminho da foto, palpite da IA, confiança, estado,
  resposta do técnico, etiqueta corrigida, datas);
- app/bot/: bot com /start, /registar (pede o distrito), receber fotos → serviço
  de visão, receber texto → serviço RAG; os casos incertos são guardados como
  'pendente' e o agricultor é avisado de que um técnico vai responder;
- um comando para correr o bot.
O bot chama os serviços directamente, não a API HTTP.
Diz-me como criar o bot no @BotFather e onde pôr o token.
```

## Fase 5 — Painel do técnico
```
Fase 5. Cria dashboard/app.py em Streamlit:
- lista dos casos pendentes (foto, pergunta, palpite da IA, confiança, distrito);
- formulário para o técnico responder e, nos casos de foto, escolher a
  etiqueta correcta;
- ao submeter: guardar, marcar como 'respondido' e enviar a resposta ao
  agricultor pelo Telegram;
- página simples com estatísticas (casos por estado, por doença, por distrito).
Protege o painel com uma palavra-passe definida no .env.
```

## Fase 6 — Alertas meteorológicos
```
Fase 6. Implementa app/alerts/:
- tabela/dicionário de distritos com coordenadas (começa por 10 distritos de
  províncias diferentes);
- consulta diária à Open-Meteo (chuva, temperatura máxima, vento) para os
  distritos com agricultores registados;
- regras de alerta (chuva forte, calor extremo, vento forte) configuráveis;
- o LLM escreve um aviso curto e simples; envio pelo bot;
- agendamento com APScheduler (ex.: 6h00, hora de Maputo) e um comando para
  testar o envio manualmente;
- testes com mock da API.
```

## Fase 7 — Docker, deploy e README
```
Fase 7. Prepara o sistema para ser usado e partilhado:
- Dockerfile e docker-compose (api, bot, dashboard);
- README.md claro em português: problema (com dados de Moçambique),
  solução, diagrama de arquitectura em Mermaid, stack, resultados do modelo
  (accuracy e matriz de confusão), como correr localmente, decisões técnicas
  (limites de confiança, RAG sem alucinações, human-in-the-loop), limitações
  e próximos passos (WhatsApp, áudio, mapa de pragas);
- sugere onde fazer o deploy gratuito e guia-me passo a passo.
```

---

## Prompts úteis a qualquer momento
- `Explica-me este ficheiro linha a linha.`
- `O que pode falhar nesta fase quando agricultores reais usarem o sistema? Como nos protegemos?`
- `Corre os testes e corrige o que falhar.`
- `Revê o código desta fase: há problemas de segurança, bugs ou partes difíceis de entender?`
- `Actualiza a secção "Estado actual" do CLAUDE.md.`
