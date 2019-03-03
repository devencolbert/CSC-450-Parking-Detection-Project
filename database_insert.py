from app import db
from app.models import Lot, Spot

#Insert data about lot into table
def insert_lot_data(title_data):
    u = Lot(title = title_data)
    db.session.add(u) #Add record to table
    db.session.commit() #Commit record to table
    u = Lot.query.get(1) #[TEST] Display records of Lot in CMD

#Insert data about parking spot into table
def insert_spot_data(availability_data):
    x = Spot(availability = availability_data)
    db.session.add(x) #Add record to table
    db.session.commit() #Commit Record to table
    x = Spot.query.get(1) #[TEST] Display records of Spot in CMD

t_data = 'INSERT_LOT_TITLE_HERE' #Location data passed by image processing system
ava_data = 'INSERT_SPOT_AVAILABILITY_HERE' #Availability data passed by image processing system
insert_lot_data(t_data)
insert_spot_data(ava_data)
