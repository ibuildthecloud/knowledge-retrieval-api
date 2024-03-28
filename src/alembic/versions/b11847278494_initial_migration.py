"""initial migration

Revision ID: b11847278494
Revises: 
Create Date: 2024-03-27 21:33:00.227641

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'b11847278494'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('datasets',
    sa.Column('name', sa.String(), nullable=False),
    sa.PrimaryKeyConstraint('name')
    )
    op.create_table('file_index',
    sa.Column('file_id', sa.String(), nullable=False),
    sa.Column('dataset', sa.String(), nullable=False),
    sa.ForeignKeyConstraint(['dataset'], ['datasets.name'], ),
    sa.PrimaryKeyConstraint('file_id', 'dataset')
    )
    op.create_table('document_index',
    sa.Column('document_id', sa.String(), nullable=False),
    sa.Column('dataset', sa.String(), nullable=False),
    sa.Column('file_id', sa.String(), nullable=False),
    sa.ForeignKeyConstraint(['file_id', 'dataset'], ['file_index.file_id', 'file_index.dataset'], ),
    sa.PrimaryKeyConstraint('document_id', 'dataset', 'file_id')
    )
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_table('document_index')
    op.drop_table('file_index')
    op.drop_table('datasets')
    # ### end Alembic commands ###