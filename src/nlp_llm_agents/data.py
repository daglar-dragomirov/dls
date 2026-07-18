from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class Intent:
    name: str
    queries: tuple[str, ...]
    required: tuple[str, ...]
    optional: tuple[str, ...]
    negative: tuple[str, ...] = ()


INTENTS: tuple[Intent, ...] = (
    Intent(
        name="terrace_restaurant",
        queries=("ресторан с верандой", "ужин на летней террасе", "кафе с открытой верандой"),
        required=("restaurant", "terrace"),
        optional=("good_food", "reservation", "family_friendly"),
        negative=("takeaway_only",),
    ),
    Intent(
        name="romantic_jazz_bar",
        queries=("романтичный джаз-бар", "бар с живым джазом для свидания", "уютный джазовый бар"),
        required=("bar", "jazz", "romantic"),
        optional=("cocktails", "live_music", "late_hours"),
        negative=("noisy", "kids_party"),
    ),
    Intent(
        name="kids_cafe",
        queries=("кафе для детей", "семейное кафе с игровой комнатой", "куда сходить с ребенком поесть"),
        required=("cafe", "kids_room"),
        optional=("family_friendly", "desserts", "high_chairs"),
        negative=("adults_only",),
    ),
    Intent(
        name="quiet_coworking",
        queries=("тихое место поработать с ноутбуком", "кофейня с розетками и wifi", "спокойное кафе для работы"),
        required=("cafe", "wifi", "quiet"),
        optional=("power_outlets", "good_coffee", "long_stay"),
        negative=("loud_music",),
    ),
    Intent(
        name="pet_friendly",
        queries=("ресторан куда можно с собакой", "pet friendly кафе", "заведение с животными"),
        required=("pet_friendly",),
        optional=("terrace", "water_for_pets", "friendly_staff"),
        negative=("no_pets",),
    ),
    Intent(
        name="vegan_food",
        queries=("веганское кафе", "ресторан с vegan меню", "где поесть без мяса и молока"),
        required=("vegan",),
        optional=("healthy_food", "gluten_free", "good_food"),
        negative=("meat_only",),
    ),
    Intent(
        name="late_pharmacy",
        queries=("аптека ночью", "круглосуточная аптека рядом", "где купить лекарства поздно"),
        required=("pharmacy", "24h"),
        optional=("near_metro", "delivery"),
        negative=("closed_evening",),
    ),
    Intent(
        name="sports_bar",
        queries=("спортбар с трансляциями", "бар посмотреть футбол", "паб с большим экраном"),
        required=("bar", "sports_broadcasts"),
        optional=("beer", "big_screen", "late_hours"),
        negative=("quiet",),
    ),
)


TAG_TO_SURFACE = {
    "restaurant": ("ресторан", "кухня", "ужин", "блюда"),
    "cafe": ("кафе", "кофейня", "десерты", "кофе"),
    "bar": ("бар", "паб", "коктейли", "напитки"),
    "pharmacy": ("аптека", "лекарства", "фармацевт"),
    "terrace": ("веранда", "терраса", "летняя площадка"),
    "jazz": ("джаз", "саксофон", "живая музыка"),
    "romantic": ("романтичный", "свидание", "уютный свет"),
    "kids_room": ("детская комната", "аниматор", "детское меню"),
    "wifi": ("wifi", "вайфай", "интернет"),
    "quiet": ("тихо", "спокойно", "не шумно"),
    "pet_friendly": ("можно с собакой", "pet friendly", "пускают с питомцами"),
    "vegan": ("веган", "растительное меню", "без молока и мяса"),
    "24h": ("круглосуточно", "24 часа", "ночью открыто"),
    "sports_broadcasts": ("трансляции", "футбол", "матчи"),
    "good_food": ("вкусно", "сильная кухня", "хорошие блюда"),
    "reservation": ("бронь", "резерв", "заказ столика"),
    "family_friendly": ("семейно", "удобно с детьми", "дружелюбно к семье"),
    "cocktails": ("коктейли", "барная карта", "авторские напитки"),
    "live_music": ("живая музыка", "концерты", "выступления"),
    "late_hours": ("работает допоздна", "до ночи", "поздние часы"),
    "power_outlets": ("розетки", "зарядка", "место для ноутбука"),
    "good_coffee": ("хороший кофе", "зерно", "капучино"),
    "long_stay": ("можно сидеть долго", "не торопят", "удобные столы"),
    "water_for_pets": ("миска для собаки", "вода питомцам"),
    "friendly_staff": ("вежливый персонал", "заботливые официанты"),
    "healthy_food": ("здоровая еда", "полезное меню"),
    "gluten_free": ("без глютена", "gluten free"),
    "near_metro": ("рядом с метро", "у метро"),
    "delivery": ("доставка", "заказ онлайн"),
    "beer": ("пиво", "крафт", "эль"),
    "big_screen": ("большой экран", "проектор", "телевизоры"),
    "takeaway_only": ("только навынос",),
    "noisy": ("очень шумно",),
    "kids_party": ("детские праздники",),
    "adults_only": ("только для взрослых",),
    "loud_music": ("громкая музыка",),
    "no_pets": ("с животными нельзя",),
    "meat_only": ("только мясо",),
    "closed_evening": ("закрывается рано",),
}


ORG_PREFIXES = ("Север", "Мята", "Лампа", "Город", "Оливка", "Соль", "Атлас", "Место", "Фонарь")
ORG_SUFFIXES = ("на углу", "на Тверской", "Парк", "Лофт", "Плюс", "Room", "Daily", "Station")


def _surface(tag: str, rng: random.Random) -> str:
    return rng.choice(TAG_TO_SURFACE.get(tag, (tag,)))


def _make_review(tags: Iterable[str], rng: random.Random) -> str:
    selected = rng.sample(list(tags), k=min(max(2, rng.randint(2, 5)), len(list(tags))))
    phrases = [_surface(tag, rng) for tag in selected]
    return "В отзывах часто пишут: " + ", ".join(phrases) + "."


def _label_for(intent: Intent, public_tags: set[str], hidden_tags: set[str]) -> tuple[int, float]:
    all_tags = public_tags | hidden_tags
    required_hit = sum(tag in all_tags for tag in intent.required)
    optional_hit = sum(tag in all_tags for tag in intent.optional)
    negative_hit = sum(tag in all_tags for tag in intent.negative)
    required_score = required_hit / max(len(intent.required), 1)
    optional_score = optional_hit / max(len(intent.optional), 1)
    score = 0.78 * required_score + 0.22 * optional_score - 0.35 * negative_hit
    score = max(0.0, min(1.0, score))
    return int(score >= 0.62), score


def generate_dataset(size: int, seed: int = 42) -> pd.DataFrame:
    rng = random.Random(seed)
    rows = []
    for row_id in range(size):
        intent = rng.choice(INTENTS)
        query = rng.choice(intent.queries)
        should_match = rng.random() < 0.52

        public_tags: set[str] = set()
        hidden_tags: set[str] = set()

        if should_match:
            for tag in intent.required:
                (public_tags if rng.random() < 0.48 else hidden_tags).add(tag)
            for tag in intent.optional:
                if rng.random() < 0.56:
                    (public_tags if rng.random() < 0.42 else hidden_tags).add(tag)
        else:
            distractor = rng.choice([x for x in INTENTS if x.name != intent.name])
            public_tags.update(rng.sample(list(distractor.required), k=min(len(distractor.required), rng.randint(1, 2))))
            if rng.random() < 0.45 and intent.negative:
                hidden_tags.add(rng.choice(intent.negative))
            if rng.random() < 0.28:
                public_tags.add(rng.choice(intent.required))

        noise_tags = [tag for tag in TAG_TO_SURFACE if tag not in public_tags and tag not in hidden_tags]
        public_tags.update(rng.sample(noise_tags, k=rng.randint(1, 4)))
        hidden_tags.update(rng.sample(noise_tags, k=rng.randint(1, 4)))
        label, relevance_score = _label_for(intent, public_tags, hidden_tags)

        org_name = f"{rng.choice(ORG_PREFIXES)} {rng.choice(ORG_SUFFIXES)}"
        category = next((tag for tag in public_tags if tag in {"restaurant", "cafe", "bar", "pharmacy"}), rng.choice(["restaurant", "cafe", "bar"]))
        description = " ".join(_surface(tag, rng) for tag in sorted(public_tags)[:5])
        review = _make_review(hidden_tags | public_tags, rng)
        rows.append(
            {
                "pair_id": row_id,
                "query_id": intent.name,
                "query": query,
                "organization_name": org_name,
                "category": category,
                "public_description": description,
                "public_tags": " ".join(sorted(public_tags)),
                "review_snippets": review,
                "hidden_tags": " ".join(sorted(hidden_tags)),
                "label": label,
                "relevance_score": round(relevance_score, 4),
            }
        )
    return pd.DataFrame(rows)

