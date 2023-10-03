import smtplib
import socket
from decouple import config
import psutil
import os
import logging

def config_log() -> None:
    logging.basicConfig(filename = "C:\\update_ip.log", level = logging.INFO,
                    format = "%(asctime)s - %(levelname)s - %(message)s")

def send_mail(new_ip:str) -> None:
    email_addr = config('GMAIL_ADDR')
    email_pass = config('GMAIL_PASWD')
    from_addr = email_addr
    to_addr = email_addr
    subject = "your new IP"
    message = f"Your new IP is {new_ip}"
    msg = f"Subject: {subject}\n\n{message}"

    smtp_object = smtplib.SMTP('smtp.gmail.com',587)
    smtp_object.ehlo()
    smtp_object.starttls()
    smtp_object.login(email_addr, email_pass)

    smtp_object.sendmail(from_addr,to_addr,msg)
    smtp_object.quit()

def get_ip_from_file(file_name :str) -> str:
    file = open(file_name,'r')
    read_ip = file.read()
    file.close()
    return read_ip

def set_ip_in_file(file_name: str, new_ip : str) -> None:
    file = open(file_name, 'w')
    file.write(new_ip)
    file.close()

def get_all_ip_addresses(family: socket.AddressFamily) -> tuple[str, str]:
    for interface, nics in psutil.net_if_addrs().items():
        for nic in nics:
            if nic.family == family:
                yield (interface, nic.address)

def find_ip_by_if_name(addresses: list, if_name:str) -> str:
    for int_name, addr in addresses:
        if int_name == if_name:
            return addr
    else:
        return ""

def find_ip_by_prefix(addresses:list, prefix:str) -> str:
    for if_name, addr in addresses:
        if addr.startswith(prefix):
            return addr
    else:
        return ""

def detect_ip_change(ip_file:str) -> tuple[bool, str]:
    b_ip_changed = False
    # create file for the first time if not exist
    if not os.path.isfile(ip_file):
        set_ip_in_file(ip_file, "0.0.0.0")

    old_ip = get_ip_from_file(ip_file)
    all_ips = list(get_all_ip_addresses(socket.AF_INET))
    new_ip = find_ip_by_prefix(all_ips, '172')
    if old_ip != new_ip:
        b_ip_changed = True
        set_ip_in_file(ip_file, new_ip)

    return (b_ip_changed, new_ip)

if __name__ == "__main__":
    ip_file_name = "./ip_address.txt"
    config_log()
    b_is_changed, new_ip = detect_ip_change(ip_file_name)
    if b_is_changed == True:
        send_mail(new_ip)
        logging.info(f"ip was changed to {new_ip}")
    else:
        logging.info("IP is the same")