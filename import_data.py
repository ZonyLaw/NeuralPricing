from django.conf import settings
import requests
import pandas as pd
from email.message import EmailMessage
import ssl
import smtplib
from .models import Ticker, PriceAlert
from config import ALPHA_API, GMAIL_PASSOWORD


def market_price(from_currency, to_currency):
    url = "https://www.alphavantage.co/query"

    API_KEY = ALPHA_API
    url_params = {
        "function": "CURRENCY_EXCHANGE_RATE",
        "from_currency": from_currency,
        "to_currency": to_currency,
        "apikey": API_KEY,
    }

    try:
        extract = requests.get(url, params=url_params)
        data = extract.json()
        normal_data = pd.json_normalize(data)
        print(normal_data)
        return normal_data.iloc[0, 4]
    except:
        print('fail to retrieve')


def send_email(msg):

    # Define email sender and receiver
    email_sender = 'sunnylaw18@gmail.com'
    email_password = GMAIL_PASSOWORD
    email_receiver = 'sunny_law@hotmail.com'

    # Set the subject and body of the email
    subject = 'USDJPY Alert!!!!'
    body = "LESSON 1: Don't make trade in the middle BB; it's risky\n" \
           + "LESSON 2: Don't enter to follow small flux of movement which are not trend but extreme reversal.\n" \
           + "LESSON 3: Don't panic of run away movement without big news!.\n\n" \
           + msg

    em = EmailMessage()
    em['From'] = email_sender
    em['To'] = email_receiver
    em['Subject'] = subject
    em.set_content(body)

    # Add SSL (layer of security)
    context = ssl.create_default_context()

    # Log in and send email
    with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context) as smtp:
        smtp.login(email_sender, email_password)
        smtp.sendmail(email_sender, email_receiver, em.as_string())


def checking():
    tickers = Ticker.objects.all()
    for ticker in tickers:
        alerts = ticker.pricealert_set.all()
        fromCur = ticker.name[:3]
        toCur = ticker.name[3:6]
        currentPrice = market_price(fromCur, toCur)
        currentPrice = float(currentPrice)
        print(currentPrice)

        for alert in alerts:
            if alert.order_type == 'Buy Stop':
                if (alert.trigger_price < currentPrice and alert.toggle):
                    send_email(
                        f'Buy signal triggered. Current price is above you trigger: {alert.trigger_price}')
                    alert.toggle = False
                    alert.save()
                    print("Buy signal triggered")
            elif alert.order_type == 'Sell Stop':
                if (alert.trigger_price > currentPrice and alert.toggle):
                    send_email(
                        f'Sell signal triggered. Current price is below you trigger: {alert.trigger_price}')
                    alert.toggle = False
                    alert.save()
                    print("Sell signal triggered")


def testing():
    tickers = Ticker.objects.filter(name="USDJPY")
    for ticker in tickers:
        alerts = ticker.pricealert_set.all()

        for alert in alerts:
            print(ticker.info)
            print(alert.trigger_price)


# if __name__ == '__main__':
