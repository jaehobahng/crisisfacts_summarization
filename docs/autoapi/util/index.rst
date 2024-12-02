util
====

.. py:module:: util


Functions
---------

.. autoapisummary::

   util.get_eventsMeta


Module Contents
---------------

.. py:function:: get_eventsMeta(eventNoList='001', days=100)

   Fetch metadata for crisis events, including daily information for specified events.

   :param eventNoList: A comma-separated string of event numbers to fetch metadata for.
                       Defaults to '001'.
   :type eventNoList: str
   :param days: The maximum number of daily records to retrieve for each event. Defaults to 100.
   :type days: int

   :returns: A dictionary where keys are event numbers and values are lists of daily information for each event.
   :rtype: dict

   .. rubric:: Notes

   - The function fetches event data from the CrisisFACTs API, using event-specific JSON files.
   - Only the specified number of days (or fewer if the event has fewer records) will be retrieved per event.

   .. rubric:: Example

   >>> get_eventsMeta(eventNoList='001,002', days=10)
   {
       '001': [{'dateString': '2022-01-01', ...}, ...],
       '002': [{'dateString': '2022-02-01', ...}, ...]
   }


