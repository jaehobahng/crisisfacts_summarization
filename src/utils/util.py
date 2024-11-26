import requests

def get_eventsMeta(eventNoList, days):

    def getDaysForEventNo(eventNo):

        # We will download a file containing the day list for an event
        url = "http://trecis.org/CrisisFACTs/CrisisFACTS-"+eventNo+".requests.json"

        # Download the list and parse as JSON
        dayList = requests.get(url).json()

        return dayList

    eventsMeta = {}

    for eventNo in eventNoList: # for each event
        
        dailyInfo = getDaysForEventNo(eventNo) # get the list of days
        eventsMeta[eventNo]= dailyInfo[:days]

    return eventsMeta