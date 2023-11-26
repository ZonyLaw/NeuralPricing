import streamlit as st
from app_pages.multipage import MultiPage

# load pages scripts
from app_pages.page_prospect import page_prospect_body

app = MultiPage(app_name= "Pricing Analysis") # Create an instance of the app 

app.add_page("Trading Prospect", page_prospect_body)


app.run() # Run the  app
