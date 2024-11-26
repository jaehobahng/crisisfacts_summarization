import requests

def get_eventsMeta(eventNoList='001', days=100):


    def getDaysForEventNo(eventNo):

        # We will download a file containing the day list for an event
        url = "http://trecis.org/CrisisFACTs/CrisisFACTS-"+eventNo+".requests.json"

        # Download the list and parse as JSON
        dayList = requests.get(url).json()

        return dayList

    eventNoList = eventNoList.split(',')

    eventsMeta = {}

    for eventNo in eventNoList: # for each event
        
        dailyInfo = getDaysForEventNo(eventNo) # get the list of days
        eventsMeta[eventNo]= dailyInfo[:days]

    return eventsMeta