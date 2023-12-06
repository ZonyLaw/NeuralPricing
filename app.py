import streamlit as st
from app_pages.multipage import MultiPage

# load pages scripts
# from app_pages.page_prospect import page_prospect_body
from app_pages.page_prospect_v2 import page_prospect_body2
from app_pages.page_prospect_v3 import page_prospect_body3

app = MultiPage(app_name= "Pricing Analysis") # Create an instance of the app 

app.add_page("Trading Prospect v2", page_prospect_body2)
app.add_page("Trading Prospect v3", page_prospect_body3)


app.run() # Run the  app
