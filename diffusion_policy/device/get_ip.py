def get_local_ip():
    import socket


    hostname = socket.gethostname()
    ip = socket.gethostbyname(hostname)
    print(ip)
    return '192.168.2.104'

def get_local_ip1(ip='8.8.8.8'):
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect((ip, 80))
    ip = s.getsockname()[0]
    s.close()
    return ip

if __name__ == '__main__':
    print(get_local_ip1())