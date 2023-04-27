import os

#Frappe API_KEY needs to be enter as environment variable
API_KEY = os.getenv("API_KEY") if os.getenv("API_KEY") else "a781365e955a738"

#Frappe API_SECRET needs to be enter as environment variable
API_SECRET = os.getenv("API_SECRET") if os.getenv("API_SECRET") else "d823567b933a5e1"

#Frappe url needs to be enter as environment variable
URL_API_PROMPTLAYER = os.getenv("URL_API_PROMPTLAYER") if os.getenv("URL_API_PROMPTLAYER") else "http://prompt.localhost:8000/api/method/promptlytics.api"

