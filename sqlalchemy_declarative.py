import os
import sys
from sqlalchemy import Column, ForeignKey, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy import create_engine
 
Base = declarative_base()
 
 
class Spot(Base):
    __tablename__ = 'spot'
    # Here we define columns for the table address.
    # Notice that each column is also a normal Python instance attribute.
    id = Column(Integer, primary_key=True)
	status = Column(String(250))
    floor_number = Column(String(250))
    number = Column(String(250))
    location = Column(String(250))
 
# Create an engine that stores data in the local directory's
# sqlalchemy_example.db file.

engine = create_engine('sqlite:///sqlalchemy_example.db')
 
# Create all tables in the engine. This is equivalent to "Create Table"
# statements in raw SQL.
Base.metadata.create_all(engine)