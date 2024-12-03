import requests

def get_eventsMeta(eventNoList='001', days=100):
    """
    Fetch metadata for crisis events, including daily information for specified events.

    Args:
        eventNoList (str): A comma-separated string of event numbers to fetch metadata for. 
                           Defaults to '001'.
        days (int): The maximum number of daily records to retrieve for each event. Defaults to 100.

    Returns:
        dict: A dictionary where keys are event numbers and values are lists of daily information for each event.

    Notes:
        - The function fetches event data from the CrisisFACTs API, using event-specific JSON files.
        - Only the specified number of days (or fewer if the event has fewer records) will be retrieved per event.

    Example:
        >>> get_eventsMeta(eventNoList='001,002', days=10)
        {
            '001': [{'dateString': '2022-01-01', ...}, ...],
            '002': [{'dateString': '2022-02-01', ...}, ...]
        }
    """

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