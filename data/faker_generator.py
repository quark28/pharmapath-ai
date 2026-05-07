#!/usr/bin/env python3
"""
PharmaPath AI — Mock Data Generator
====================================
Генерирует синтетический датасет промышленного качества:
  • 500 врачей в пределах МКАД (координаты, расписание, метрики)
  • 10  медицинских представителей с территориями
  • ~8 000 визитов за последние 12 месяцев (Пуассон)

Использование:
    python faker_generator.py                          # defaults
    python faker_generator.py --seed 42 --num_doctors 1000 --output_dir ./my_data

Зависимости:
    pip install numpy
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import math
import os
import random
import uuid
from dataclasses import asdict, dataclass, fields
from datetime import date, timedelta
from enum import Enum
from typing import Dict, List, Tuple

import numpy as np

# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

MOSCOW_CENTER_LAT = 55.7558
MOSCOW_CENTER_LON = 37.6173
MKAD_RADIUS_KM = 16.0
EARTH_RADIUS_KM = 6_371.0

NUM_DOCTORS_DEFAULT = 500
NUM_REPS_DEFAULT = 10
HISTORY_DAYS_DEFAULT = 365


# ══════════════════════════════════════════════════════════════════════════════
#  ENUMS
# ══════════════════════════════════════════════════════════════════════════════

class Specialty(str, Enum):
    THERAPIST = "Therapist"
    CARDIOLOGIST = "Cardiologist"
    NEUROLOGIST = "Neurologist"
    ENDOCRINOLOGIST = "Endocrinologist"
    GASTROENTEROLOGIST = "Gastroenterologist"
    PULMONOLOGIST = "Pulmonologist"
    RHEUMATOLOGIST = "Rheumatologist"
    UROLOGIST = "Urologist"


class Category(str, Enum):
    A = "A"
    B = "B"
    C = "C"


class VisitStatus(str, Enum):
    SUCCESS = "Success"
    CANCELLED = "Cancelled"
    MOVED = "Moved"


# ══════════════════════════════════════════════════════════════════════════════
#  DISTRIBUTION CONFIGS
# ══════════════════════════════════════════════════════════════════════════════

SPECIALTY_WEIGHTS: Dict[Specialty, float] = {
    Specialty.THERAPIST:         0.25,
    Specialty.CARDIOLOGIST:      0.20,
    Specialty.NEUROLOGIST:       0.15,
    Specialty.ENDOCRINOLOGIST:   0.12,
    Specialty.GASTROENTEROLOGIST:0.10,
    Specialty.PULMONOLOGIST:     0.08,
    Specialty.RHEUMATOLOGIST:    0.05,
    Specialty.UROLOGIST:         0.05,
}

CATEGORY_WEIGHTS: Dict[Category, float] = {
    Category.A: 0.20,
    Category.B: 0.50,
    Category.C: 0.30,
}

# Параметры продаж по брику (нормальное распределение)
SALES_PARAMS = {
    Category.A: {"mean": 85.0, "std": 15.0},
    Category.B: {"mean": 50.0, "std": 20.0},
    Category.C: {"mean": 20.0, "std": 10.0},
}

# Среднее число визитов / месяц (λ для Пуассона)
VISITS_LAMBDA_PER_MONTH = {
    Category.A: 2.5,
    Category.B: 1.5,
    Category.C: 0.7,
}

# Лояльность (нормальное распределение, μ ± σ)
LOYALTY_PARAMS = {
    Category.A: {"mean": 7.5, "std": 1.5},
    Category.B: {"mean": 5.0, "std": 2.0},
    Category.C: {"mean": 3.0, "std": 1.5},
}


# ══════════════════════════════════════════════════════════════════════════════
#  СПРАВОЧНИКИ (LOOKUP DATA)
# ══════════════════════════════════════════════════════════════════════════════

MOSCOW_STREETS = [
    "ул. Тверская", "ул. Арбат", "Ленинградский просп.",
    "Кутузовский просп.", "ул. Профсоюзная", "Каширское ш.",
    "ул. Бауманская", "Варшавское ш.", "Ломоносовский просп.",
    "ул. Люблинская", "ул. Марксистская", "Рязанский просп.",
    "Волгоградский просп.", "ул. Мясницкая", "Комсомольский просп.",
    "ул. Новослободская", "Дмитровское ш.", "ул. Большая Ордынка",
    "Нахимовский просп.", "ул. Академика Янгеля",
    "Алтуфьевское ш.", "ул. Щербаковская", "ул. Косыгина",
    "Ленинский просп.", "просп. Мира", "ул. Садовая-Кудринская",
    "ул. Маросейка", "Можайское ш.", "ул. Генерала Дорохова",
    "Авиамоторная ул.", "ул. Академика Королёва", "Балаклавский просп.",
    "ул. Земляной Вал", "ул. Сретенка", "Севастопольский просп.",
    "ул. Большая Полянка", "Ярославское ш.", "ул. 1905 года",
    "ул. Покровка", "Хорошёвское ш.",
]

FACILITY_TEMPLATES = [
    "Городская поликлиника №{n}",
    "ГКБ №{n}",
    "Медицинский центр «Здоровье»",
    "Клиника «Медси»",
    "Клиника «Семейный доктор»",
    "МЦ «Он Клиник»",
    "Поликлиника при ГКБ №{n}",
    "КДЦ №{n}",
    "МЦ «АльфаМед»",
    "Многопрофильная клиника «Столица»",
    "Клиника «Чайка»",
    "МЦ «СМ-Клиника»",
    "Городская поликлиника №{n}, филиал {f}",
]

# ── Имена ─────────────────────────────────────────────────────────────────────

LAST_M = [
    "Иванов","Петров","Сидоров","Козлов","Новиков","Морозов","Волков",
    "Соловьёв","Васильев","Зайцев","Павлов","Семёнов","Голубев",
    "Виноградов","Богданов","Воробьёв","Фёдоров","Михайлов","Беляев",
    "Тарасов","Белов","Комаров","Орлов","Киселёв","Макаров",
    "Андреев","Ковалёв","Ильин","Гусев","Титов",
    "Кузьмин","Кудрявцев","Баранов","Куликов","Алексеев",
]
LAST_F = [s[:-2] + "а" if s.endswith("ов") else
          s[:-2] + "а" if s.endswith("ев") else
          s + "а" for s in LAST_M]

FIRST_M = [
    "Александр","Дмитрий","Максим","Сергей","Андрей","Алексей","Артём",
    "Илья","Кирилл","Михаил","Никита","Матвей","Роман","Егор","Арсений",
    "Иван","Денис","Евгений","Даниил","Тимофей",
]
FIRST_F = [
    "Анна","Мария","Елена","Ольга","Наталья","Татьяна","Ирина",
    "Светлана","Екатерина","Юлия","Марина","Валентина","Людмила",
    "Галина","Дарья","Алина","Виктория","Кристина","Вероника","Полина",
]
PATRON_M = [
    "Александрович","Дмитриевич","Сергеевич","Андреевич","Алексеевич",
    "Михайлович","Иванович","Николаевич","Владимирович","Петрович",
    "Евгеньевич","Олегович","Викторович","Юрьевич","Борисович",
]
PATRON_F = [p.replace("ич", "на") for p in PATRON_M]

# ── Транслитерация (для e-mail) ──────────────────────────────────────────────

_TRANSLIT = {
    "а":"a","б":"b","в":"v","г":"g","д":"d","е":"e","ё":"yo",
    "ж":"zh","з":"z","и":"i","й":"y","к":"k","л":"l","м":"m",
    "н":"n","о":"o","п":"p","р":"r","с":"s","т":"t","у":"u",
    "ф":"f","х":"kh","ц":"ts","ч":"ch","ш":"sh","щ":"shch",
    "ъ":"","ы":"y","ь":"","э":"e","ю":"yu","я":"ya",
}

def transliterate(text: str) -> str:
    """Кириллица → латиница (упрощённая)."""
    result = []
    for ch in text.lower():
        result.append(_TRANSLIT.get(ch, ch))
    return "".join(result)


# ── Фарм-препараты (наши / конкуренты) ───────────────────────────────────────

OUR_DRUGS: Dict[Specialty, List[str]] = {
    Specialty.CARDIOLOGIST:      ["Лориста", "Аторис", "Тарка"],
    Specialty.THERAPIST:         ["Лориста", "Аторис", "Супракс"],
    Specialty.NEUROLOGIST:       ["Нейромакс", "ГабаПент", "Церебролизин"],
    Specialty.ENDOCRINOLOGIST:   ["Метфогамма", "Глимепирид", "Диабетон"],
    Specialty.GASTROENTEROLOGIST:["Омез", "Ранитидин", "Де-Нол"],
    Specialty.PULMONOLOGIST:     ["Беродуал", "Сингуляр", "Форадил"],
    Specialty.RHEUMATOLOGIST:    ["Мовалис", "Аркоксиа", "Метотрексат"],
    Specialty.UROLOGIST:         ["Омник", "Простамол", "Витапрост"],
}

COMPETITOR_DRUGS: Dict[Specialty, List[str]] = {
    Specialty.CARDIOLOGIST:      ["Валз", "Липримар", "Конкор", "Престариум"],
    Specialty.THERAPIST:         ["Амоксиклав", "Нолипрел", "Эналаприл"],
    Specialty.NEUROLOGIST:       ["Лирика", "Тебантин", "Кортексин"],
    Specialty.ENDOCRINOLOGIST:   ["Янувия", "Форсига", "Глюкофаж"],
    Specialty.GASTROENTEROLOGIST:["Нексиум", "Париет", "Ганатон"],
    Specialty.PULMONOLOGIST:     ["Серетид", "Спирива", "Пульмикорт"],
    Specialty.RHEUMATOLOGIST:    ["Целебрекс", "Вольтарен", "Ксефокам"],
    Specialty.UROLOGIST:         ["Фокусин", "Аводарт", "Сетегис"],
}

CONDITIONS: Dict[Specialty, List[str]] = {
    Specialty.CARDIOLOGIST:      ["артериальной гипертензии","ИБС","ХСН","аритмии"],
    Specialty.THERAPIST:         ["ОРВИ","гипертензии","бронхита","ангины"],
    Specialty.NEUROLOGIST:       ["нейропатической боли","мигрени","эпилепсии"],
    Specialty.ENDOCRINOLOGIST:   ["СД 2 типа","гипотиреоза","ожирения"],
    Specialty.GASTROENTEROLOGIST:["ГЭРБ","язвенной болезни","СРК"],
    Specialty.PULMONOLOGIST:     ["ХОБЛ","бронхиальной астмы","пневмонии"],
    Specialty.RHEUMATOLOGIST:    ["ревматоидного артрита","остеоартроза","подагры"],
    Specialty.UROLOGIST:         ["ДГПЖ","хронического простатита","МКБ"],
}

# ── Шаблоны отчётов ──────────────────────────────────────────────────────────

REPORT_TEMPLATES = [
    "Визит к {doc}. Обсудили применение {drug} при {cond}. Врач {react}. {extra}",
    "Встреча с {doc}. Презентовал {drug}. {react}. Врач упомянул, что использует {comp}. {extra}",
    "Плановый визит. {doc} заинтересован в {drug}. {react}. {extra}",
    "{doc} выслушал презентацию по {drug}. {react}. Основное возражение: {obj}. {extra}",
    "Визит к {doc}. Детейлинг {drug}. {react}. Договорились: {agree}.",
    "Короткий визит. {doc} был занят, но удалось обсудить {drug}. {react}. {extra}",
]

REACTIONS = [
    "положительно отнёсся к препарату",
    "выразил заинтересованность",
    "скептически отнёсся",
    "попросил доп. исследования",
    "готов попробовать назначить",
    "уже назначает аналог конкурента",
    "нейтрально воспринял информацию",
    "задал много вопросов по побочным эффектам",
]

OBJECTIONS = [
    "высокая цена для пациентов",
    "недостаточно клинических данных",
    "побочные эффекты (ЖКТ)",
    "пациенты привыкли к другому препарату",
    "нет в формуляре больницы",
    "предпочитает оригинальный препарат",
]

AGREEMENTS = [
    "назначить 3 пациентам в течение месяца",
    "повторный визит через 2 недели",
    "оставить образцы и буклеты",
    "организовать круглый стол для коллег",
    "предоставить результаты последних КИ",
    "включить в рекомендации отделения",
]

EXTRAS = [
    "Оставил 2 упаковки образцов.",
    "Передал буклет с результатами исследования.",
    "Врач попросил визитку.",
    "В следующий раз принести сравнительные таблицы.",
    "",
    "",
    "Врач спешил на обход, визит был коротким.",
    "Присутствовала медсестра, тоже заинтересовалась.",
]

TERRITORIES = [
    "Север", "Юг", "Запад", "Восток", "Центр",
    "Северо-Запад", "Северо-Восток", "Юго-Запад", "Юго-Восток", "ЦАО",
]


# ══════════════════════════════════════════════════════════════════════════════
#  DATA CLASSES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Doctor:
    id: str
    full_name: str
    gender: str
    specialty: str
    category: str
    work_address: str
    latitude: float
    longitude: float
    schedule_json: str
    loyalty_score: float
    avg_sales_brick: float
    phone: str
    email: str


@dataclass
class Rep:
    id: str
    full_name: str
    territory: str
    home_lat: float
    home_lon: float


@dataclass
class Visit:
    id: str
    doctor_id: str
    rep_id: str
    visit_date: str        # ISO-формат: 2025-03-15
    visit_time: str        # HH:MM
    day_of_week: int       # 0=Mon … 6=Sun
    status: str
    duration_minutes: int
    report_text: str


# ══════════════════════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def _random_point_in_circle(
    center_lat: float,
    center_lon: float,
    radius_km: float,
) -> Tuple[float, float]:
    """
    Равномерно-случайная точка внутри круга на поверхности Земли.
    Используем sqrt(U) для uniform-area distribution.
    """
    angle = random.uniform(0, 2 * math.pi)
    dist = radius_km * math.sqrt(random.random())

    delta_lat = (dist / EARTH_RADIUS_KM) * (180.0 / math.pi)
    delta_lon = (
        dist / (EARTH_RADIUS_KM * math.cos(math.radians(center_lat)))
    ) * (180.0 / math.pi)

    lat = center_lat + delta_lat * math.cos(angle)
    lon = center_lon + delta_lon * math.sin(angle)
    return round(lat, 6), round(lon, 6)


def _make_full_name() -> Tuple[str, str]:
    """Возвращает (ФИО, пол)."""
    gender = random.choice(("M", "F"))
    if gender == "M":
        name = (
            f"{random.choice(LAST_M)} "
            f"{random.choice(FIRST_M)} "
            f"{random.choice(PATRON_M)}"
        )
    else:
        name = (
            f"{random.choice(LAST_F)} "
            f"{random.choice(FIRST_F)} "
            f"{random.choice(PATRON_F)}"
        )
    return name, gender


def _make_address() -> str:
    """Генерирует адрес медучреждения в Москве."""
    tpl = random.choice(FACILITY_TEMPLATES)
    facility = tpl.format(n=random.randint(1, 220), f=random.randint(1, 5))
    street = random.choice(MOSCOW_STREETS)
    bld = random.randint(1, 120)
    return f"{facility}, {street}, д. {bld}"


def _make_schedule() -> dict:
    """Генерирует JSON расписания врача (часы приёма)."""
    weekdays = ["monday", "tuesday", "wednesday", "thursday", "friday"]
    if random.random() < 0.3:
        weekdays.append("saturday")

    pattern = random.choice(("standard", "morning", "afternoon", "split"))
    schedule: dict = {}

    for day in weekdays:
        if random.random() < 0.08:       # ~8 % — выходной
            continue

        if pattern == "standard":
            h_start, h_end = random.choice([8, 9]), random.choice([16, 17, 18])
        elif pattern == "morning":
            h_start, h_end = random.choice([7, 8]), random.choice([13, 14, 15])
        elif pattern == "afternoon":
            h_start, h_end = random.choice([12, 13, 14]), random.choice([18, 19, 20])
        else:
            h_start, h_end = 9, 18

        schedule[day] = {"start": f"{h_start:02d}:00", "end": f"{h_end:02d}:00"}

    return schedule


def _make_phone(rng: np.random.Generator) -> str:
    return (
        f"+7(9{rng.integers(10, 100):02d})"
        f"{rng.integers(100, 1000):03d}-"
        f"{rng.integers(10, 100):02d}-"
        f"{rng.integers(10, 100):02d}"
    )


def _make_email(full_name: str, uid: str) -> str:
    parts = full_name.split()
    last = transliterate(parts[0])
    first_init = transliterate(parts[1][0])
    domain = random.choice(["mail.ru", "yandex.ru", "gmail.com", "rambler.ru"])
    return f"{last}.{first_init}{uid[:4]}@{domain}"


def _shorten_name(full_name: str) -> str:
    """Иванов Александр Сергеевич → Иванов А.С."""
    parts = full_name.split()
    return f"{parts[0]} {parts[1][0]}.{parts[2][0]}."


def _make_report(doctor_name: str, specialty: Specialty) -> str:
    """Генерирует текст отчёта медпреда (рус.)."""
    tpl = random.choice(REPORT_TEMPLATES)
    return tpl.format(
        doc=_shorten_name(doctor_name),
        drug=random.choice(OUR_DRUGS.get(specialty, ["препарат"])),
        comp=random.choice(COMPETITOR_DRUGS.get(specialty, ["аналог"])),
        cond=random.choice(CONDITIONS.get(specialty, ["заболевания"])),
        react=random.choice(REACTIONS),
        obj=random.choice(OBJECTIONS),
        agree=random.choice(AGREEMENTS),
        extra=random.choice(EXTRAS),
    ).strip()


# ══════════════════════════════════════════════════════════════════════════════
#  GENERATORS
# ══════════════════════════════════════════════════════════════════════════════

def generate_doctors(n: int, rng: np.random.Generator) -> List[Doctor]:
    """Сгенерировать n врачей."""
    spec_list = list(SPECIALTY_WEIGHTS.keys())
    spec_probs = np.array(list(SPECIALTY_WEIGHTS.values()))
    spec_probs /= spec_probs.sum()                         # на случай float drift

    cat_list = list(CATEGORY_WEIGHTS.keys())
    cat_probs = np.array(list(CATEGORY_WEIGHTS.values()))
    cat_probs /= cat_probs.sum()

    doctors: List[Doctor] = []

    for _ in range(n):
        uid = str(uuid.uuid4())
        full_name, gender = _make_full_name()

        spec: Specialty = spec_list[int(rng.choice(len(spec_list), p=spec_probs))]
        cat:  Category  = cat_list[int(rng.choice(len(cat_list), p=cat_probs))]

        lat, lon = _random_point_in_circle(
            MOSCOW_CENTER_LAT, MOSCOW_CENTER_LON, MKAD_RADIUS_KM
        )

        lp = LOYALTY_PARAMS[cat]
        loyalty = float(np.clip(round(rng.normal(lp["mean"], lp["std"]), 1), 0, 10))

        sp = SALES_PARAMS[cat]
        sales = float(max(0, round(rng.normal(sp["mean"], sp["std"]), 2)))

        doctors.append(Doctor(
            id=uid,
            full_name=full_name,
            gender=gender,
            specialty=spec.value,
            category=cat.value,
            work_address=_make_address(),
            latitude=lat,
            longitude=lon,
            schedule_json=json.dumps(_make_schedule(), ensure_ascii=False),
            loyalty_score=loyalty,
            avg_sales_brick=sales,
            phone=_make_phone(rng),
            email=_make_email(full_name, uid),
        ))

    return doctors


def generate_reps(n: int, rng: np.random.Generator) -> List[Rep]:
    """Сгенерировать n медпредов."""
    reps: List[Rep] = []
    for i in range(n):
        full_name, _ = _make_full_name()
        lat, lon = _random_point_in_circle(
            MOSCOW_CENTER_LAT, MOSCOW_CENTER_LON, MKAD_RADIUS_KM * 0.7
        )
        reps.append(Rep(
            id=f"REP-{i + 1:03d}",
            full_name=full_name,
            territory=TERRITORIES[i % len(TERRITORIES)],
            home_lat=lat,
            home_lon=lon,
        ))
    return reps


def generate_visits(
    doctors: List[Doctor],
    reps: List[Rep],
    history_days: int,
    rng: np.random.Generator,
) -> List[Visit]:
    """
    Сгенерировать историю визитов.
    Каждый медпред обслуживает свой пул врачей.
    Частота визитов ∼ Poisson(λ), λ зависит от категории.
    """
    visits: List[Visit] = []
    today = date.today()
    start_date = today - timedelta(days=history_days)

    docs_per_rep = len(doctors) // len(reps)

    # часы начала визитов (дискретное распределение)
    hours = np.array([9, 10, 11, 12, 13, 14, 15, 16])
    hour_probs = np.array([0.10, 0.15, 0.20, 0.10, 0.10, 0.15, 0.15, 0.05])
    hour_probs /= hour_probs.sum()

    for rep_idx, rep in enumerate(reps):
        lo = rep_idx * docs_per_rep
        hi = lo + docs_per_rep if rep_idx < len(reps) - 1 else len(doctors)
        pool = doctors[lo:hi]

        for doc in pool:
            cat = Category(doc.category)
            lam = VISITS_LAMBDA_PER_MONTH[cat] * (history_days / 30.0)
            n_visits = int(rng.poisson(lam))
            n_visits = max(0, min(n_visits, history_days // 5))

            if n_visits == 0:
                continue

            visit_day_offsets = sorted(
                rng.choice(history_days, size=n_visits, replace=False)
            )

            for offset in visit_day_offsets:
                vdate = start_date + timedelta(days=int(offset))

                # Пропускаем выходные с вероятностью 90 %
                if vdate.weekday() >= 5 and random.random() < 0.90:
                    continue

                hour = int(rng.choice(hours, p=hour_probs))
                minute = int(rng.choice([0, 15, 30, 45]))

                # Статус
                roll = random.random()
                if roll < 0.75:
                    status = VisitStatus.SUCCESS
                elif roll < 0.90:
                    status = VisitStatus.CANCELLED
                else:
                    status = VisitStatus.MOVED

                # Длительность
                if status == VisitStatus.SUCCESS:
                    dur = int(np.clip(rng.normal(20, 8), 5, 60))
                elif status == VisitStatus.CANCELLED:
                    dur = 0
                else:
                    dur = int(np.clip(rng.normal(10, 5), 0, 30))

                # Текст отчёта
                spec = Specialty(doc.specialty)
                report = (
                    _make_report(doc.full_name, spec)
                    if status == VisitStatus.SUCCESS
                    else ""
                )

                visits.append(Visit(
                    id=str(uuid.uuid4()),
                    doctor_id=doc.id,
                    rep_id=rep.id,
                    visit_date=vdate.isoformat(),
                    visit_time=f"{hour:02d}:{minute:02d}",
                    day_of_week=vdate.weekday(),
                    status=status.value,
                    duration_minutes=dur,
                    report_text=report,
                ))

    return visits


# ══════════════════════════════════════════════════════════════════════════════
#  I/O
# ══════════════════════════════════════════════════════════════════════════════

def _field_names(cls) -> List[str]:
    return [f.name for f in fields(cls)]


def save_csv(rows: list, path: str, cls) -> None:
    """Сохранить список dataclass-объектов в CSV."""
    names = _field_names(cls)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=names)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    print(f"  ✅  {len(rows):>6,} записей → {path}")


def save_meta(
    out_dir: str,
    doctors: List[Doctor],
    reps: List[Rep],
    visits: List[Visit],
    seed: int,
) -> None:
    """Сохранить метаинформацию о генерации."""
    cat_dist = {}
    spec_dist = {}
    for d in doctors:
        cat_dist[d.category] = cat_dist.get(d.category, 0) + 1
        spec_dist[d.specialty] = spec_dist.get(d.specialty, 0) + 1

    status_dist = {}
    for v in visits:
        status_dist[v.status] = status_dist.get(v.status, 0) + 1

    meta = {
        "seed": seed,
        "generated_at": date.today().isoformat(),
        "counts": {
            "doctors": len(doctors),
            "reps": len(reps),
            "visits": len(visits),
        },
        "distributions": {
            "category": dict(sorted(cat_dist.items())),
            "specialty": dict(sorted(spec_dist.items(), key=lambda x: -x[1])),
            "visit_status": dict(sorted(status_dist.items())),
        },
    }

    path = os.path.join(out_dir, "generation_meta.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)
    print(f"  📄  Метаданные → {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    ap = argparse.ArgumentParser(description="PharmaPath AI — Mock Data Generator")
    ap.add_argument("--seed",         type=int, default=42)
    ap.add_argument("--num_doctors",  type=int, default=NUM_DOCTORS_DEFAULT)
    ap.add_argument("--num_reps",     type=int, default=NUM_REPS_DEFAULT)
    ap.add_argument("--history_days", type=int, default=HISTORY_DAYS_DEFAULT)
    ap.add_argument("--output_dir",   type=str, default="./output")
    args = ap.parse_args()

    # Сиды
    random.seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 62)
    print("  PharmaPath AI — Mock Data Generator")
    print("=" * 62)
    print(f"  Seed ........... {args.seed}")
    print(f"  Doctors ........ {args.num_doctors}")
    print(f"  Reps ........... {args.num_reps}")
    print(f"  History ........ {args.history_days} days")
    print(f"  Output ......... {args.output_dir}/")
    print("-" * 62)

    # ── Generate ──────────────────────────────────────────────────────────────
    print("\n🏥  Генерация врачей...")
    doctors = generate_doctors(args.num_doctors, rng)

    print("👔  Генерация медпредов...")
    reps = generate_reps(args.num_reps, rng)

    print("📋  Генерация истории визитов...")
    visits = generate_visits(doctors, reps, args.history_days, rng)

    # ── Save ──────────────────────────────────────────────────────────────────
    print("\n💾  Сохранение...")
    save_csv(doctors, os.path.join(args.output_dir, "doctors_base.csv"),  Doctor)
    save_csv(reps,    os.path.join(args.output_dir, "reps.csv"),          Rep)
    save_csv(visits,  os.path.join(args.output_dir, "visits_log.csv"),    Visit)
    save_meta(args.output_dir, doctors, reps, visits, args.seed)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n📊  Статистика:")

    cat_counts = {}
    for d in doctors:
        cat_counts[d.category] = cat_counts.get(d.category, 0) + 1
    print(f"  Категории врачей:   {dict(sorted(cat_counts.items()))}")

    spec_counts = {}
    for d in doctors:
        spec_counts[d.specialty] = spec_counts.get(d.specialty, 0) + 1
    top3 = sorted(spec_counts.items(), key=lambda x: -x[1])[:3]
    print(f"  Топ-3 специальности: {top3}")

    st_counts = {}
    for v in visits:
        st_counts[v.status] = st_counts.get(v.status, 0) + 1
    print(f"  Статусы визитов:    {dict(sorted(st_counts.items()))}")

    # Preview
    print("\n🔍  Пример врача:")
    sample_doc = doctors[0]
    print(f"     {sample_doc.full_name} | {sample_doc.specialty} | "
          f"Cat {sample_doc.category} | Loyalty {sample_doc.loyalty_score} | "
          f"Sales {sample_doc.avg_sales_brick}")
    print(f"     📍 ({sample_doc.latitude}, {sample_doc.longitude})")
    print(f"     🏥 {sample_doc.work_address}")

    print("\n🔍  Пример визита:")
    success_visits = [v for v in visits if v.status == "Success"]
    if success_visits:
        sv = success_visits[0]
        print(f"     {sv.visit_date} {sv.visit_time} | {sv.duration_minutes} мин")
        print(f"     📝 {sv.report_text[:120]}...")

    print("\n" + "=" * 62)
    print("  ✨  Данные готовы для PharmaPath AI pipeline!")
    print("=" * 62)


if __name__ == "__main__":
    main()