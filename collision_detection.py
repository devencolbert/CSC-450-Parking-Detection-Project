import yaml



with open("car_coor.yml", 'r') as car_yaml:
    car_coor = yaml.safe_load(car_yaml)

#for index in range(len(car_coor)):
#    for key in car_coor[index]:
#        print(car_coor[index][key])
carList = car_coor[0]['coors']
carList_x1 = car_coor[0]['coors'][0][0]
carList_y1 = car_coor[0]['coors'][0][1]
carList_x2 = car_coor[0]['coors'][1][0]
carList_y2 = car_coor[0]['coors'][1][1]
print(carList_x1)


with open("parking_spots.yml", 'r') as parking_yaml:
    parking_coor = yaml.safe_load(parking_yaml)

#for i in range(len(parking_coor)):
#    for k in parking_coor[i]:
        #print(parking_coor[i][k])\
spotList = [parking_coor[0]['points'][2], parking_coor[0]['points'][0]]
spotList_x1 = parking_coor[0]['points'][2][0]
spotList_y1 = parking_coor[0]['points'][2][1]
spotList_x2 = parking_coor[0]['points'][0][0]
spotList_y2 = parking_coor[0]['points'][0][1]
print(spotList_x2)


