"""empty message

Revision ID: 8253d5ec3233
Revises: 9e0a0dfdeb0e
Create Date: 2019-03-01 16:41:25.599337

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '8253d5ec3233'
down_revision = '9e0a0dfdeb0e'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column('lot', sa.Column('title', sa.String(length=64), nullable=True))
    op.create_index(op.f('ix_lot_title'), 'lot', ['title'], unique=False)
    op.drop_index('ix_lot_location', table_name='lot')
    op.drop_column('lot', 'location')
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column('lot', sa.Column('location', sa.VARCHAR(length=64), nullable=True))
    op.create_index('ix_lot_location', 'lot', ['location'], unique=1)
    op.drop_index(op.f('ix_lot_title'), table_name='lot')
    op.drop_column('lot', 'title')
    # ### end Alembic commands ###