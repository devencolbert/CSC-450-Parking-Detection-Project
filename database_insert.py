from app import db
from app.models import Lot, Spot

#Insert data about lot into table
def insert_lot_data(title_data):
    u = Lot(title = title_data, percentage = 'test_percentage', total_spaces = 'test_total', available_spaces = 'test_available')
    db.session.add(u) #Add record to table
    db.session.commit() #Commit record to table
    u = Lot.query.get(1) #[TEST] Display records of Lot in CMD

#Insert data about parking spot into table
def insert_spot_data(availability_data):
    x = Spot(availability = availability_data, lot_id = 2)
    db.session.add(x) #Add record to table
    db.session.commit() #Commit Record to table
    x = Spot.query.get(1) #[TEST] Display records of Spot in CMD

t_data = 'test_title' #Location data passed by image processing system
ava_data = 0 #Availability data passed by image processing system
insert_lot_data(t_data)
insert_spot_data(ava_data)
