"""initial-migration

Revision ID: 091c4ebd4f6d
Revises: 
Create Date: 2024-03-06 19:07:43.607651

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "091c4ebd4f6d"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table(
        "file_index",
        sa.Column("file_id", sa.String(), nullable=False),
        sa.Column("dataset", sa.String(), nullable=False),
        sa.PrimaryKeyConstraint("file_id", "dataset"),
    )
    op.create_table(
        "document_index",
        sa.Column("document_id", sa.String(), nullable=False),
        sa.Column("dataset", sa.String(), nullable=False),
        sa.Column("file_id", sa.String(), nullable=False),
        sa.ForeignKeyConstraint(
            name="document_index_dataset_fkey",
            columns=["file_id", "dataset"],
            refcolumns=["file_index.file_id", "file_index.dataset"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("document_id", "dataset", "file_id"),
    )
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_table("document_index")
    op.drop_table("file_index")
    # ### end Alembic commands ###