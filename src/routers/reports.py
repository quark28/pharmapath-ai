"""
PharmaPath AI — Reports Router
"""

from __future__ import annotations

import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException

from src.config import settings
from src.schemas import ReportResult, SubmitReportRequest
from src.services.llm_service import create_llm_service

router = APIRouter(prefix="/reports", tags=["Reports"])


@router.post("/submit", response_model=ReportResult)
async def submit_report(req: SubmitReportRequest):
    """
    Принять отчёт медпреда, обработать LLM, сохранить.
    """
    from src.main import get_data_provider
    data = get_data_provider()

    # Проверяем, что врач существует
    doc = data.get_doctor(req.doctor_id)
    if doc is None:
        raise HTTPException(404, f"Doctor {req.doctor_id} not found")

    # LLM-парсинг
    llm = create_llm_service(
        backend=settings.llm_backend,
        ollama_url=settings.ollama_url,
        ollama_model=settings.ollama_model,
    )
    parsed = llm.parse_report(req.report_text)

    # Сохраняем визит
    visit_id = str(uuid.uuid4())
    data.add_visit(
        visit_id=visit_id,
        doctor_id=req.doctor_id,
        rep_id=req.rep_id,
        visit_date=req.visit_date,
        visit_time=req.visit_time,
        status=req.status.value,
        duration_minutes=req.duration_minutes,
        report_text=req.report_text,
    )

    return ReportResult(
        visit_id=visit_id,
        sentiment=parsed.sentiment,
        competitors=parsed.competitors,
        objections=parsed.objections,
        agreements=parsed.agreements,
        key_topics=parsed.key_topics,
        raw_text=req.report_text,
        llm_backend=parsed.backend,
    )