This script is used to let me know when my laptop IP was changed
due to DHCP assingment.
When it does, it send me an Email to my gmail address.
You need to create an App password in your gmail settings,
and add .env file with GMAIL_ADDR and GMAIL_PASWD values.

The script lists all IP addesses in the system, finds the one
with the requested prefix, and compare it with the IP stored
in a file. If they are not the same, the file is updated with the new
ip, and an Email is sent with the new IP.

In windows, I set the script to run twice a day using the windows's task
scheduler.