"""
PharmaPath AI — Doctors Router
"""

from __future__ import annotations

import io
from typing import List, Optional

import pandas as pd
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile

from src.schemas import DoctorInfo

router = APIRouter(prefix="/doctors", tags=["Doctors"])


@router.get("/", response_model=List[DoctorInfo])
async def list_doctors(
    specialty: Optional[str] = Query(None),
    category: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    """Список врачей с фильтрацией."""
    from src.main import get_data_provider
    data = get_data_provider()

    df = data.get_all_doctors()
    visit_stats = data.get_bulk_visit_stats()

    if specialty:
        df = df[df["specialty"] == specialty]
    if category:
        df = df[df["category"] == category]

    df = df.iloc[offset: offset + limit]

    result = []
    for _, row in df.iterrows():
        stats = visit_stats.get(row["id"], {})
        result.append(DoctorInfo(
            id=row["id"],
            full_name=row["full_name"],
            specialty=str(row["specialty"]),
            category=str(row["category"]),
            work_address=str(row["work_address"]),
            latitude=float(row["latitude"]),
            longitude=float(row["longitude"]),
            loyalty_score=float(row["loyalty_score"]),
            avg_sales_brick=float(row["avg_sales_brick"]),
            total_visits=stats.get("total_visits", 0),
            success_rate=stats.get("success_rate", 0.0),
            days_since_last_visit=stats.get("days_since_last_visit", 999),
        ))

    return result


@router.get("/{doctor_id}", response_model=DoctorInfo)
async def get_doctor(doctor_id: str):
    """Карточка одного врача."""
    from src.main import get_data_provider
    data = get_data_provider()

    doc = data.get_doctor(doctor_id)
    if doc is None:
        raise HTTPException(404, f"Doctor {doctor_id} not found")

    stats = data.get_visit_stats(doctor_id)

    return DoctorInfo(
        id=doc["id"],
        full_name=doc["full_name"],
        specialty=str(doc["specialty"]),
        category=str(doc["category"]),
        work_address=str(doc["work_address"]),
        latitude=float(doc["latitude"]),
        longitude=float(doc["longitude"]),
        loyalty_score=float(doc["loyalty_score"]),
        avg_sales_brick=float(doc["avg_sales_brick"]),
        total_visits=stats.get("total_visits", 0),
        success_rate=stats.get("success_rate", 0.0),
        days_since_last_visit=stats.get("days_since_last_visit", 999),
    )


@router.post("/upload")
async def upload_doctors_csv(file: UploadFile = File(...)):
    """Загрузить/обновить базу врачей из CSV."""
    from src.main import get_data_provider
    data = get_data_provider()

    if not file.filename.endswith(".csv"):
        raise HTTPException(400, "Only CSV files accepted")

    content = await file.read()
    df = pd.read_csv(io.BytesIO(content), dtype={"id": str})

    required = {"id", "full_name", "specialty", "category", "latitude", "longitude"}
    missing = required - set(df.columns)
    if missing:
        raise HTTPException(400, f"Missing columns: {missing}")

    count = data.replace_doctors(df)
    return {"status": "ok", "doctors_loaded": count}