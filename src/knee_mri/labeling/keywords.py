"""Từ điển thuật ngữ đa ngôn ngữ cho việc gán nhãn yếu từ report.

Khảo sát ``dataset/train.csv`` cho thấy report trải trên ít nhất 5 ngôn ngữ
(Anh ~1.6k, Tây Ban Nha ~0.7k, Hà Lan ~0.3k, Đức ~0.24k, Pháp ~0.08k). Bản cũ
chỉ xử lý phủ định tiếng Anh (``"no "``, ``"without"``, ``"negativ"``), nên mọi
câu kiểu *"Sin derrame articular"* hay *"kein Knochenödem"* đều bị gán nhãn
dương tính sai (lỗi P2-3).

Hai lỗi từ vựng khác của bản cũ cũng được sửa ở đây:

* ``Medial Meniscus`` và ``Lateral Meniscus`` cùng chứa từ khóa trần ``"menisc"``
  nên **luôn nhận giá trị giống hệt nhau** — 2/12 nhãn vô giá trị (lỗi P2-1).
  Nay hai nhãn này được suy ra từ *thuật ngữ giải phẫu* kết hợp với *chỉ định
  bên*, nên phân biệt được thật.
* ``"lcl"`` (dây chằng bên ngoài) bị xếp vào nhãn ``MCL`` (dây chằng bên trong) —
  hai cấu trúc khác nhau (lỗi P2-2). Nay đã tách.
"""

from __future__ import annotations

import re
from typing import Final

#: Ngưỡng độ dài: từ ngắn hơn mức này cần ranh giới từ để tránh khớp nhầm
#: (``"oa"`` trong ``"coarse"``, ``"no"`` trong ``"nodule"``). Từ dài hơn được
#: khớp dạng chuỗi con để bắt được từ ghép tiếng Đức (``"Innenmeniskus"``).
_BOUNDARY_MAX_LEN: Final[int] = 5

#: Ký tự được coi là thuộc về một từ, gồm cả nguyên âm có dấu của các ngôn ngữ Âu.
_WORD_CHARS: Final[str] = r"0-9a-zà-öø-ÿ"


def build_pattern(terms: list[str]) -> re.Pattern[str]:
    """Dựng regex khớp bất kỳ thuật ngữ nào trong danh sách.

    Từ ngắn được bọc ranh giới từ; từ dài khớp chuỗi con để bắt từ ghép.
    """
    parts: list[str] = []
    for term in sorted(terms, key=len, reverse=True):
        escaped = re.escape(term.lower())
        if len(term) <= _BOUNDARY_MAX_LEN:
            parts.append(rf"(?<![{_WORD_CHARS}]){escaped}(?![{_WORD_CHARS}])")
        else:
            parts.append(escaped)
    return re.compile("|".join(parts), re.IGNORECASE)


# --------------------------------------------------------------------- phủ định
#: Từ/cụm phủ định theo ngôn ngữ. Bao gồm cả các cách nói "bình thường",
#: "nguyên vẹn" — trong văn phong X-quang chúng phủ định bệnh lý một cách gián tiếp.
NEGATION_TERMS: Final[dict[str, list[str]]] = {
    "en": [
        "no",
        "not",
        "without",
        "absent",
        "absence",
        "negative for",
        "free of",
        "no evidence",
        "no sign",
        "unremarkable",
        "intact",
        "normal",
        "denies",
        "rules out",
        "ruled out",
        "preserved",
        "within normal limits",
    ],
    "es": [
        "sin",
        "no",
        "ausencia",
        "ausente",
        "negativo",
        "integro",
        "íntegro",
        "normal",
        "normales",
        "conservado",
        "conservada",
        "no se observa",
        "no se identifica",
        "no hay",
        "dentro de limites normales",
        "dentro de límites normales",
    ],
    "nl": [
        "geen",
        "niet",
        "zonder",
        "afwezig",
        "intact",
        "normaal",
        "normale",
        "ongestoord",
        "gaaf",
        "geen aanwijzing",
    ],
    "de": [
        "kein",
        "keine",
        "ohne",
        "nicht",
        "unauffällig",
        "unauffaellig",
        "intakt",
        "regelrecht",
        "normal",
        "frei von",
        "kein hinweis",
    ],
    "fr": [
        "sans",
        "pas de",
        "aucun",
        "aucune",
        "absence",
        "normal",
        "normale",
        "intact",
        "intacte",
        "respecté",
    ],
    "it": ["senza", "nessun", "nessuna", "assenza", "normale", "integro", "indenne"],
    "pt": ["sem", "não", "nao", "ausência", "ausencia", "normal", "íntegro", "integro"],
}

NEGATION_PATTERN: Final[re.Pattern[str]] = build_pattern(
    [term for terms in NEGATION_TERMS.values() for term in terms]
)

# ------------------------------------------------------------ chỉ định bên (laterality)
#: Thuật ngữ chỉ bên trong / bên ngoài. Dùng để tách ``Medial *`` khỏi ``Lateral *``.
#: Danh sách gồm cả từ ghép Đức/Hà Lan (``Innenmeniskus``, ``buitenmeniscus``) vì
#: chúng mã hóa **cả** giải phẫu lẫn bên trong một từ duy nhất — nếu chỉ liệt kê
#: ``innen`` thì ranh giới từ sẽ không khớp được bên trong từ ghép.
SIDE_TERMS: Final[dict[str, list[str]]] = {
    "medial": [
        "medial",
        "mediale",
        "mediaal",
        "medialen",
        "mediali",
        "interno",
        "interna",
        "internal",
        "innen",
        "innere",
        "binnen",
        "médial",
        "médiale",
        "mediaal",
        "intern",
        "innenmeniskus",
        "innenmeniscus",
        "binnenmeniscus",
        "innenband",
    ],
    "lateral": [
        "lateral",
        "laterale",
        "lateraal",
        "lateralen",
        "externo",
        "externa",
        "external",
        "außen",
        "aussen",
        "buiten",
        "latéral",
        "latérale",
        "extern",
        "aussenmeniskus",
        "außenmeniskus",
        "buitenmeniscus",
        "aussenband",
        "außenband",
    ],
}

SIDE_PATTERNS: Final[dict[str, re.Pattern[str]]] = {
    side: build_pattern(terms) for side, terms in SIDE_TERMS.items()
}

# ------------------------------------------------------------------ giải phẫu
#: Thuật ngữ cho các cấu trúc cần kết hợp với chỉ định bên.
ANATOMY_TERMS: Final[dict[str, list[str]]] = {
    "meniscus": [
        "meniscus",
        "menisci",
        "meniscal",
        "menisco",
        "meniscos",
        "meniskus",
        "menisken",
        "meniscaal",
        "ménisque",
        "menisc",
        "binnenmeniscus",
        "buitenmeniscus",
        "innenmeniskus",
        "aussenmeniskus",
        "außenmeniskus",
    ],
    "osteoarthritis": [
        "osteoarthritis",
        "osteoarthrosis",
        "osteoarthr",
        "arthrosis",
        "artrosis",
        "arthrose",
        "artrose",
        "gonarthrosis",
        "gonarthrose",
        "gonartrosis",
        "chondropathy",
        "chondropathie",
        "chondropatia",
        "chondromalacia",
        "chondromalacie",
        "cartilage loss",
        "kraakbeenverlies",
        "knorpelschaden",
        "degenerative change",
        "degeneratieve",
        "degenerativ",
    ],
    "compartment": [
        "femorotibial",
        "femorotibiaal",
        "femorotibiale",
        "compartment",
        "compartiment",
        "kompartiment",
        "condyle",
        "cóndilo",
        "condylus",
        "condyl",
        "tibial plateau",
        "tibiaplateau",
    ],
}

ANATOMY_PATTERNS: Final[dict[str, re.Pattern[str]]] = {
    key: build_pattern(terms) for key, terms in ANATOMY_TERMS.items()
}

# ------------------------------------------------------------ nhãn khớp trực tiếp
#: Nhãn được nhận diện bằng thuật ngữ riêng, không cần suy luận bên.
DIRECT_LABEL_TERMS: Final[dict[str, list[str]]] = {
    "ACL": [
        "acl",
        "anterior cruciate",
        "cruciate anterior",
        "ligamentum cruciatum anterius",
        "ligamento cruzado anterior",
        "voorste kruisband",
        "vkb",
        "vorderes kreuzband",
        "ligament croisé antérieur",
        "lca",
    ],
    "MCL": [
        "mcl",
        "medial collateral",
        "collateral medial",
        "ligamento colateral medial",
        "ligamento colateral interno",
        "mediale collaterale band",
        "mediale band",
        "innenband",
        "mediales kollateralband",
        "ligament collatéral médial",
        "ligamentum collaterale mediale",
        "lcm",
    ],
    "PF OA": [
        "patellofemoral",
        "patelofemoral",
        "patellofemoraal",
        "patellofemorale",
        "femoropatellar",
        "femoropatelar",
        "femoropatellaire",
        "retropatellar",
        "retropatelar",
        "retropatellaire",
        "patellar cartilage",
        "chondropathia patellae",
    ],
    "Effusion": [
        "effusion",
        "joint effusion",
        "derrame",
        "derrame articular",
        "erguss",
        "gelenkerguss",
        "kniegelenkserguss",
        "hydrops",
        "epanchement",
        "épanchement",
        "vocht",
        "gewrichtsvocht",
        "versamento",
    ],
    "Synovitis": [
        "synovitis",
        "synovite",
        "synovitits",
        "sinovitis",
        "synovialitis",
        "synoviale",
        "synovial thickening",
        "synoviale verdikking",
        "pannus",
    ],
    "Baker's": [
        "baker",
        "bakercyste",
        "baker's cyst",
        "popliteal cyst",
        "quiste de baker",
        "quiste poplíteo",
        "quiste popliteo",
        "popliteumcyste",
        "poplitealzyste",
        "kyste poplité",
        "cisti di baker",
    ],
    "Contusion": [
        "contusion",
        "contusión",
        "contusio",
        "bone bruise",
        "bone marrow edema",
        "bone marrow oedema",
        "marrow edema",
        "marrow oedema",
        "botoedeem",
        "beenmergoedeem",
        "knochenmarksödem",
        "knochenmarksoedem",
        "knochenödem",
        "edema óseo",
        "edema oseo",
        "edema de médula",
        "oedeem",
        "kneuzing",
        "contusione",
        "œdème osseux",
    ],
    "Fracture": [
        "fracture",
        "fractura",
        "fraktur",
        "fractuur",
        "frattura",
        "fissure",
        "fissura",
        "stress fracture",
        "avulsion",
        "avulsión",
        "breuk",
    ],
}

DIRECT_LABEL_PATTERNS: Final[dict[str, re.Pattern[str]]] = {
    label: build_pattern(terms) for label, terms in DIRECT_LABEL_TERMS.items()
}

#: Nhãn suy ra từ (giải phẫu × bên). Giá trị là ``(khóa_giải_phẫu, bên)``.
LATERAL_LABEL_RULES: Final[dict[str, tuple[str, str]]] = {
    "Medial Meniscus": ("meniscus", "medial"),
    "Lateral Meniscus": ("meniscus", "lateral"),
    "Medial OA": ("osteoarthritis", "medial"),
    "Lateral OA": ("osteoarthritis", "lateral"),
}

__all__ = [
    "ANATOMY_PATTERNS",
    "ANATOMY_TERMS",
    "DIRECT_LABEL_PATTERNS",
    "DIRECT_LABEL_TERMS",
    "LATERAL_LABEL_RULES",
    "NEGATION_PATTERN",
    "NEGATION_TERMS",
    "SIDE_PATTERNS",
    "SIDE_TERMS",
    "build_pattern",
]
