"""Nomes em português e mensagens para o agricultor.

⚠️ TODOS os nomes das doenças têm de ser validados por um extensionista:
os nomes usados no campo variam de região para região.

Estado de cada nome (no comentário ao lado):
- "confirmado (fonte)": confirmado numa fonte indicada (ex.: manual do IIAM/FAO);
- "confirmado (Embrapa), nome local por confirmar": nome técnico confirmado numa
  fonte brasileira; falta confirmar o nome usado em Moçambique;
- "técnico, por confirmar localmente": nome técnico correcto, mas falta saber
  se é o nome que os agricultores e extensionistas usam em Moçambique;
- "ambíguo": a classe do dataset é genérica e o nome pode corresponder a
  várias pragas/doenças diferentes.
Quando um nome for confirmado, actualiza o comentário com a fonte.
"""

# Chave: o campo "folder" do labels.json (não muda com as correcções de ortografia).
DISEASE_NAMES_PT: dict[str, str] = {
    "Cashew/anthracnose": "antracnose",                                   # confirmado (estudos do IIAM, Nampula, sobre o caju)
    "Cashew/gumosis": "gomose",                                           # técnico, por confirmar localmente
    "Cashew/healthy": "planta saudável",                                  # não é doença
    "Cashew/leaf miner": "minador da folha",                              # técnico, por confirmar localmente
    "Cashew/red rust": "ferrugem vermelha (mancha de alga)",              # confirmado (literatura sobre Cephaleuros virescens)
    "Cassava/bacterial blight": "bacteriose da mandioca",                 # confirmado (Embrapa), nome local por confirmar
    "Cassava/brown spot": "mancha castanha (cercosporiose)",              # técnico, por confirmar localmente
    "Cassava/green mite": "ácaro verde",                                  # confirmado (estudo IIA Angola/Zâmbia/Moçambique)
    "Cassava/healthy": "planta saudável",                                 # não é doença
    "Cassava/mosaic": "mosaico da mandioca",                              # confirmado (guia NRI/Univ. Greenwich, em português para África)
    "Maize/fall armyworm": "lagarta do funil",                            # confirmado (Ministério da Agricultura de Moçambique e IIAM)
    "Maize/grasshoper": "gafanhoto",                                      # ambíguo: o dataset não indica a espécie
    "Maize/healthy": "planta saudável",                                   # não é doença
    "Maize/leaf beetle": "besouro da folha",                              # ambíguo: o dataset não indica a espécie
    "Maize/leaf blight": "queima das folhas (helmintosporiose)",          # confirmado (Embrapa), nome local por confirmar
    "Maize/leaf spot": "mancha da folha",                                 # ambíguo: pode ser várias manchas foliares
    "Maize/streak virus": "vírus do estriado",                            # técnico, por confirmar localmente
    "Tomato/healthy": "planta saudável",                                  # não é doença
    "Tomato/leaf blight": "queima da folha",                              # ambíguo: requeima ou pinta-preta (alternariose)?
    "Tomato/leaf curl": "enrolamento da folha (vírus)",                   # técnico, por confirmar localmente
    "Tomato/septoria leaf spot": "septoriose (manchas nas folhas)",       # confirmado (Embrapa), nome local por confirmar
    "Tomato/verticulium wilt": "murcha-de-verticílio",                    # confirmado (Embrapa), nome local por confirmar
}

PHOTO_TIP = "Tire outra foto da folha doente, de perto e com boa luz."
TECHNICIAN_NOTE = "Um técnico vai ver a sua foto e responder."


def disease_pt(folder: str, fallback: str) -> str:
    """Nome da doença em português; se não existir, usa o nome original."""
    return DISEASE_NAMES_PT.get(folder, fallback)


def msg_high(crop_pt: str, disease: str) -> str:
    return f"A sua planta de {crop_pt} parece ter {disease}."


def msg_high_healthy(crop_pt: str) -> str:
    return f"A sua planta de {crop_pt} parece saudável. Continue a vigiar as folhas."


def msg_medium(crop_pt: str, disease: str) -> str:
    return f"Pode ser {disease} na planta de {crop_pt}, mas não tenho a certeza. {PHOTO_TIP}"


def msg_medium_healthy(crop_pt: str) -> str:
    return f"A planta de {crop_pt} parece saudável, mas não tenho a certeza. {PHOTO_TIP}"


def msg_crop_only(crop_pt: str) -> str:
    return f"Vejo que é {crop_pt}, mas não consigo ver bem o problema. {PHOTO_TIP} {TECHNICIAN_NOTE}"


def msg_unverified() -> str:
    return "Recebemos a sua foto. Um técnico vai vê-la e responder."


def msg_low() -> str:
    return f"Não consigo dar um diagnóstico com segurança. {TECHNICIAN_NOTE}"


def msg_not_supported(crops_pt: list[str]) -> str:
    # Não dizemos "planta": a foto pode nem ser de uma planta.
    crops_text = ", ".join(crops_pt[:-1]) + " ou " + crops_pt[-1] if len(crops_pt) > 1 else crops_pt[0]
    return (f"Não consigo reconhecer esta foto. Envie uma foto de perto de uma folha de {crops_text}. "
            "Um técnico vai ver a sua foto.")
