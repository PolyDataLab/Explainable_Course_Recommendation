import pandas as pd
#calculating offered courses in each semster and storing the list of courses in a dictionary for post-processing 
def calculate_offered_dict(offered_course_dict, timestamp, course):
   
    if timestamp not in offered_course_dict:
        offered_course_dict[timestamp] = [course]
    else:
        if course not in offered_course_dict[timestamp]:
            offered_course_dict[timestamp].append(course)
    return offered_course_dict

def offered_course_cal(input_path):
    #data = pd.read_csv(input_path)
    data= pd.read_json(input_path, orient='records', lines=True)
    #data = input_path
    #print(data)
    offered_course_dict = {}
    #index1= 1
    for x in range(len(data)):
        timestamp = data['Semester'][x]
        course = data['itemID'][x]
        #if timestamp not in offered_course_dict:
        offered_course_dict = calculate_offered_dict(offered_course_dict, timestamp, course)

    return offered_course_dict

if __name__ == '__main__':
    offered_course_dict = offered_course_cal('./all_data.json')
    #print(offered_course_dict[1221])
