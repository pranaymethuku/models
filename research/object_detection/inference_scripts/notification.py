"""
Created on Thu Apr 27 2020
@author: rkabealo

Class: CSE 5915 - Information Systems
Section: 6pm TR, Spring 2020
Prof: Prof. Jayanti

A Python 3 script to perform notification of detections

Usage:
    * Called as a helper script from gui.py
"""

# Necessary imports for the sending and recieving of mail
import smtplib, ssl
from email.mime.text import MIMEText 
from email.utils import formataddr  
from email.mime.multipart import MIMEMultipart 
from email.mime.base import MIMEBase 
from email import encoders 
# And imghdr to find the types of our images
import imghdr
from os.path import basename 
from email.message import EmailMessage


# Other necessary imports
import sys

def send_notification_email(attachment, detected_class, best_score, average_score, detection_time): 
    # Define a file which holds a list of users to be notified
    reciever_file = "notified_users.txt"

    # Define a host and port 
    host = "smtp.gmail.com"
    port = 587

    # Configuration of our gmail account
    sender_email = "tieredobjectrecognition@gmail.com"
    sender_name = "TOR SYSTEM"
    password = "tor4zeltech"

    # Get list of recievers from file
    try: 
        with open(reciever_file) as f:
            receiver_emails = [line.rstrip() for line in f if '#' not in line]
    except Exception as e: 
        print("ERROR: {}".format(e))

    # Configure a default reciever name for people who recieve the files (since we may not have their actual names)
    default_receiver_name = "user"

    # Email body
    email_body = """
    Detected a \"{}\" at {} with a max confidence of {:0.4}% and an average confidence of {:0.4}%. 
    
    A snapshot of the detection at the time of maximum confidence is attached! 
    """.format(detected_class, detection_time, best_score*100, average_score*100)

    for receiver_email in receiver_emails:
        # Display that we're attempting to send the email in the terminal 
        print("Sending notification email...")
        print(detected_class)
        # Define the message 
        msg = EmailMessage()
        msg["To"] = formataddr((default_receiver_name, receiver_email))
        msg["From"] = formataddr((sender_name, sender_email))
        msg["Subject"] = "DETECTION: " + detected_class.upper() 
        msg.set_content(email_body)

        # Attempt to attach the file 
        try:
            print(attachment)
            print(basename(attachment))
            with open(attachment, "rb") as a:
                img_data = a.read()
            msg.add_attachment(img_data, maintype='image',
                                        subtype=imghdr.what(None, img_data), filename = basename(attachment))
        except Exception as e:
            print("ERROR: Email not sent. File {} not found.".format(attachment))
            break
        
        # Try to send the email 
        try:
            server = smtplib.SMTP(host, port) # Creating a SMTP session 
            # Encrypting the email
            context = ssl.create_default_context() 
            server.starttls(context=context)
            server.login(sender_email, password) # Login to gmail account 
            server.sendmail(sender_email, receiver_email, msg.as_string()) # Send the email! 
            print("Notification email sent!")
        except Exception as e:
            # Display the exception to the user 
            print(e)
        finally:
            # Close the server
            server.quit()