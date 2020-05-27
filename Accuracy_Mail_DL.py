#!/usr/bin/env python
# coding: utf-8


import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


mail_content = "Hey Developer,
the accuracy of your Deep Learning model has already been optimized! Now, you can work on your another project! 
Regards
The Automation Team!"

#The mail addresses and password
sender_address = 'namantanuj@gmail.com'
sender_pass = 'GCVEEVur'
receiver_address = '18ume029@lnmiit.ac.in'


#Setup the MIME
message = MIMEMultipart()
message['From'] = sender_address
message['To'] = receiver_address
message['Subject'] = 'Success of Model Training'   #The subject line


#The body and the attachments for the mail
message.attach(MIMEText(mail_content, 'plain'))


#Create SMTP session for sending the mail
session = smtplib.SMTP('smtp.gmail.com', 587) #use gmail with port
session.starttls() #enable security

session.login(sender_address, sender_pass) #login with mail_id and password
text = message.as_string()

session.sendmail(sender_address, receiver_address, text)
print("success mail sent")
session.quit()
