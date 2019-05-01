"""spots table

Revision ID: 9e0a0dfdeb0e
Revises: 1739dbf4b1aa
Create Date: 2019-03-01 15:35:23.936700

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '9e0a0dfdeb0e'
down_revision = '1739dbf4b1aa'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('spot',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('availability', sa.String(length=140), nullable=True),
    sa.Column('lot_id', sa.Integer(), nullable=True),
    sa.ForeignKeyConstraint(['lot_id'], ['lot.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_table('spot')
    # ### end Alembic commands ###