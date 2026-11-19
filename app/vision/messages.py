"""Nomes em português e mensagens para o agricultor.

⚠️ Os nomes das doenças são uma proposta e têm de ser validados por um
extensionista: os nomes usados no campo variam de região para região.
"""

# Chave: o campo "folder" do labels.json (não muda com as correcções de ortografia).
DISEASE_NAMES_PT: dict[str, str] = {
    "Cashew/anthracnose": "antracnose",
    "Cashew/gumosis": "gomose",
    "Cashew/healthy": "planta saudável",
    "Cashew/leaf miner": "mineira da folha",
    "Cashew/red rust": "ferrugem vermelha",
    "Cassava/bacterial blight": "bacteriose da mandioca",
    "Cassava/brown spot": "mancha castanha",
    "Cassava/green mite": "ácaro verde",
    "Cassava/healthy": "planta saudável",
    "Cassava/mosaic": "mosaico da mandioca",
    "Maize/fall armyworm": "lagarta-do-funil",
    "Maize/grasshoper": "gafanhoto",
    "Maize/healthy": "planta saudável",
    "Maize/leaf beetle": "besouro da folha",
    "Maize/leaf blight": "queima da folha",
    "Maize/leaf spot": "mancha da folha",
    "Maize/streak virus": "vírus do estriado",
    "Tomato/healthy": "planta saudável",
    "Tomato/leaf blight": "queima da folha",
    "Tomato/leaf curl": "enrolamento da folha",
    "Tomato/septoria leaf spot": "septoriose (manchas nas folhas)",
    "Tomato/verticulium wilt": "murchidão (verticilose)",
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


def msg_low() -> str:
    return f"Não consigo dar um diagnóstico com segurança. {TECHNICIAN_NOTE}"


def msg_not_supported(crops_pt: list[str]) -> str:
    crops_text = ", ".join(crops_pt[:-1]) + " e " + crops_pt[-1] if len(crops_pt) > 1 else crops_pt[0]
    return f"Não reconheço esta planta. Por agora só conheço {crops_text}. {TECHNICIAN_NOTE}"
