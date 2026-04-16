from datetime import datetime
from typing import Optional

from sqlalchemy import (
    String, Integer, Float, DateTime, Text, ForeignKey, create_engine,
)
from sqlalchemy.orm import (
    DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker,
)

from .settings import settings

engine = create_engine(f"sqlite:///{settings.DB_PATH}", future=True)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


class Job(Base):
    __tablename__ = "jobs"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    source: Mapped[str] = mapped_column(String)
    input_name: Mapped[str] = mapped_column(String)
    status: Mapped[str] = mapped_column(String, default="queued")
    progress: Mapped[str] = mapped_column(String, default="")
    error: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    audio_path: Mapped[str] = mapped_column(String, default="")
    num_speakers: Mapped[int] = mapped_column(Integer, default=0)
    language: Mapped[str] = mapped_column(String, default="")
    speakers: Mapped[list["Speaker"]] = relationship(
        cascade="all, delete-orphan", back_populates="job"
    )
    segments: Mapped[list["Segment"]] = relationship(
        cascade="all, delete-orphan", back_populates="job"
    )


class Speaker(Base):
    __tablename__ = "speakers"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id"))
    speaker_id: Mapped[str] = mapped_column(String)
    display_name: Mapped[str] = mapped_column(String)
    total_duration: Mapped[float] = mapped_column(Float, default=0.0)
    job: Mapped[Job] = relationship(back_populates="speakers")


class Segment(Base):
    __tablename__ = "segments"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id"))
    start: Mapped[float] = mapped_column(Float)
    end: Mapped[float] = mapped_column(Float)
    speaker_id: Mapped[str] = mapped_column(String)
    text: Mapped[str] = mapped_column(Text, default="")
    job: Mapped[Job] = relationship(back_populates="segments")


def init_db() -> None:
    Base.metadata.create_all(engine)
