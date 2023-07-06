#
# weather_tool.py
# This module contains all ingredients to build a langchain tool
# that incapsule any custom function.
#
import json
from langchain.agents import Tool


#
# weather_tool.py
# builds a langchain tool that incapsules a custom function
# that retrieve weather forecasts data
#
import json
from typing import List
from langchain.agents import Tool


#
# weather_data_retriever
# is an example of a custom python function
# that takes a list of custom arguments and returns a text (or in general any data structure)
#
def weather_data_retriever(where: str = None, when: str = None, required_data: List[str] = []) -> str:
    print("여기 호출" + "weather_data_retriever")
    '''
    given a location and a time period, this custom function
    returns weather forecast as a data structure (in JSON format).

    This is a mockup function, returning a fixed text tempalte.
    The function could wrap an external API returning realtime weather forecast.

    parameters:
        where: location as text, e.g. 'Genova, Italy'
        when: time period, e.g. 'today, now'

    returns:
        weather foreast description as a JSON. E.g.
        {"forecast": "sunny all the day", "temperature": "20 degrees Celsius"}

    '''
    if where and when:
        # this function is a mockup, returns fake/hardcoded weather forecast data
        data = {
            'forecast': 'sunny',
            'temperature': '20 degrees Celsius'
        }

    if not where:
        data['where'] = 'location is not specified'

    if not when:
        data['when'] = 'date is not specified'

    # if required variable names are not included in the data section,
    # the attribute is added to the dictionary with value I don't know.
    for variable_name in required_data:
        if variable_name not in data.keys():
            data[variable_name] = 'unknown'

    return json.dumps(data)


def weather(json_request: str) -> str:

    print("여기 호출" + "weather")
    '''
    Takes a JSON dictionary as input in the form:
        { "when":"<time>", "where":"<location>" }

    Example:
        { "when":"today", "where":"Genova, Italy" }

    Args:
        request (str): The JSON dictionary input string.

    Returns:
        The weather data for the specified location and time.
    '''
    print("여기 호출:" + json_request)
    arguments = json.loads(json_request)
    where = arguments["where"]
    when = arguments["when"]
    required_data = arguments["required_data"]
    return weather_data_retriever(where=where, when=when, required_data=required_data)


#
# instantiate the langchain tool.
# The tool description instructs the LLM to pass data using a JSON.
# Note the "{{" and "}}": this double quotation is needed
# to avoid a runt-time error triggered by the agent instatiation.
#
name = "weather"
request_format = '{{"when":"<time>","where":"<location>","required_data":["variable_name"]}}'
description = f'''
Helps to retrieve weather forecast.
Input should be JSON in the following format: {request_format}
'''

# create an instance of the custom langchain tool
Weather = Tool(name=name, func=weather, description=description)


if __name__ == '__main__':
    print(weather_data_retriever(where='Genova, Italy', when='today'))
    # => in Genova, Italy, today is sunny! Temperature is 20 degrees Celsius.

    print(weather('{ "when":"today", "where":"Genova, Italy" }'))
    # => in Genova, Italy, today is sunny! Temperature is 20 degrees Celsius.

    # print the Weather tool
    print(Weather)