"""Textos (prompts) que o CLIP compara com a foto.

O CLIP mede a semelhança entre a foto e cada frase. Somamos as probabilidades
das frases "aceitar" para obter a pontuação do filtro (entre 0 e 1).

Regras usadas para escolher as frases:
- Em inglês: o CLIP foi treinado sobretudo com legendas em inglês.
- Plantas DOENTES também são aceites: sem frases como "a diseased ... leaf with
  spots", uma folha seca ou manchada podia parecer "solo" ou "outra planta".
- Pragas só aparecem EM CIMA das culturas ("a caterpillar on a maize leaf").
  O dataset tem fotos de gafanhotos e lagartas: uma frase genérica como
  "an insect" no grupo "rejeitar" mandaria embora fotos legítimas.
- O grupo "rejeitar" cobre os erros de envio mais comuns num bot (pessoas,
  animais, documentos, céu...) e, sobretudo, OUTRAS CULTURAS comuns em
  Moçambique — o caso mais perigoso, porque se parecem com as nossas.

Mudar estas frases muda as pontuações: depois de qualquer alteração, corre
de novo scripts/avaliar_filtro.py e revê o CLIP_MIN_SCORE.
"""

ACCEPT_PROMPTS: list[str] = [
    # Milho
    "a photo of a maize leaf",
    "a close-up photo of a diseased maize leaf with spots",
    "a photo of a maize plant in a field",
    "a photo of a caterpillar on a maize leaf",
    "a photo of a grasshopper on a maize leaf",
    "a photo of a beetle on a maize leaf",
    # Mandioca
    "a photo of a cassava leaf",
    "a close-up photo of a cassava leaf with yellow mosaic",
    "a photo of a diseased cassava leaf with brown spots",
    "a photo of a cassava plant in a field",
    # Caju
    "a photo of a cashew tree leaf",
    "a close-up photo of a diseased cashew leaf",
    "a photo of a cashew tree branch",
    "a photo of a cashew tree trunk oozing gum",
    # Tomate
    "a photo of a tomato leaf",
    "a close-up photo of a diseased tomato leaf with spots",
    "a photo of a tomato plant",
    "a photo of a tomato plant with curled leaves",
    "a photo of a wilting tomato plant",
]

REJECT_PROMPTS: list[str] = [
    # Pessoas e animais
    "a photo of a person",
    "a selfie of a person's face",
    "a photo of a group of people",
    "a photo of a chicken",
    "a photo of a goat",
    "a photo of a dog",
    "a photo of a cow",
    # Coisas e lugares
    "a photo of bare soil",
    "a photo of a dirt road",
    "a photo of a document with text",
    "a screenshot of a phone screen",
    "a photo of handwriting on paper",
    "a photo of the sky",
    "a photo of a house",
    "a photo of food on a plate",
    "a photo of a car",
    # Outras plantas e culturas (as mais perigosas)
    "a photo of a bean plant leaf",
    "a photo of a groundnut plant",
    "a photo of a rice plant",
    "a photo of a sorghum plant",
    "a photo of a sweet potato leaf",
    "a photo of a potato plant",
    "a photo of a pepper plant",
    "a photo of a banana leaf",
    "a photo of a mango tree leaf",
    "a photo of a cabbage",
    "a photo of a pumpkin leaf",
    "a photo of a cotton plant",
    "a photo of grass and weeds",
]
