from .db import Base
from sqlalchemy import String, ForeignKeyConstraint
from sqlalchemy.orm import mapped_column, relationship, Mapped


class FileIndex(Base):
    __tablename__ = "file_index"

    file_id: Mapped[str] = mapped_column("file_id", String, primary_key=True)
    dataset: Mapped[str] = mapped_column("dataset", String, primary_key=True)

    documents: Mapped[list["DocumentIndex"]] = relationship(
        "DocumentIndex",
        backref="file",
        cascade="all, delete-orphan",
    )


class DocumentIndex(Base):
    __tablename__ = "document_index"

    document_id: Mapped[str] = mapped_column("document_id", String, primary_key=True)
    dataset: Mapped[str] = mapped_column(
        "dataset",
        String,
        primary_key=True,
    )
    file_id: Mapped[str] = mapped_column(
        "file_id",
        String,
        primary_key=True,
    )

    __table_args__ = (
        ForeignKeyConstraint(
            columns=["file_id", "dataset"],
            refcolumns=["file_index.file_id", "file_index.dataset"],
        ),
        {},
    )
