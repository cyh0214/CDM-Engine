"""
database.py — SQLite persistence layer for the Cognitive Decay Matrix.

Stores StudentState rows keyed by (student_id, node_id).
The CDM engine stays entirely in-memory; this module is the I/O bridge.
"""

from sqlalchemy import create_engine, Column, String, Float
from sqlalchemy.orm import DeclarativeBase, sessionmaker, Session
from typing import Generator

SQLALCHEMY_DATABASE_URL = "sqlite:///./cdm_state.db"

# check_same_thread=False is required for SQLite when used with FastAPI
engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


class DBStudentState(Base):
    """
    Persistent mirror of models.StudentState.
    Composite primary key: (student_id, node_id).
    """
    __tablename__ = "student_states"

    student_id   = Column(String, primary_key=True, index=True)
    node_id      = Column(String, primary_key=True, index=True)
    current_weight = Column(Float, nullable=False)
    last_updated   = Column(Float, nullable=False)   # UNIX timestamp

    def __repr__(self):
        return (
            f"DBStudentState(student={self.student_id!r}, node={self.node_id!r}, "
            f"weight={self.current_weight:.4f}, ts={self.last_updated})"
        )


def init_db() -> None:
    """Create all tables if they don't exist yet."""
    Base.metadata.create_all(bind=engine)


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency: yields a DB session and closes it after the request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
