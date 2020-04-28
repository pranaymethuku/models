import smtplib, ssl
from email.mime.text import MIMEText 
from email.utils import formataddr  

from email.mime.multipart import MIMEMultipart 
from email.mime.base import MIMEBase 
from email import encoders 

host = "smtp.gmail.com"
port = 587
# User configuration
sender_email = "tieredobjectrecognition@gmail.com"
sender_name = "TOR SYSTEM"
password = "tor4zeltech"

# Email body
email_html = open("email.html")
email_body = email_html.read()

receiver_emails = ["rkabealo@gmail.com", "kabealo.21@buckeyemail.osu.edu"]
default_receiver_name = "user"

# Email body
email_body = """
  SEE ATTACHED.
"""

filename = "OUTPUT"

for receiver_email in receiver_emails:
    print("Sending notification email...")

    msg = MIMEMultipart()
    msg["To"] = formataddr((default_receiver_name, receiver_email))
    msg["From"] = formataddr((sender_name, sender_email))
    msg["Subject"] = "Detection"

    msg.attach(MIMEText(email_body, "plain"))

    try:
      with open(filename, "rb") as attachment:
              part = MIMEBase("application", "octet-stream")
              part.set_payload(attachment.read())
      # Encode file in ASCII characters to send by email
      encoders.encode_base64(part)
      # Add header as key/value pair to attachment part
      part.add_header(
          "Content-Disposition",
          f"attachment; filename= {filename}",
      )
      msg.attach(part)
    except Exception as e:
        print("ERROR: Email not sent. {} not found.".format(filename))
        break

    try:
        # Creating a SMTP session 
        server = smtplib.SMTP(host, port)
        # Encrypting the email
        context = ssl.create_default_context()
        server.starttls(context=context)

        # Login to gmail account 
        server.login(sender_email, password)

        # Send the email! 
        server.sendmail(sender_email, receiver_email, msg.as_string())
        print("Notification email sent!")
    except Exception as e:
        print(e)
    finally:
        # Close the server
        server.quit()