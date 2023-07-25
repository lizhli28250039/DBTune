import http.server
import socketserver
import random
import numpy as np
import linecache
import json
import logging
import socket
import threading


PORT = 8080
metricFile='metric.txt'
TPSFile='tps.txt'

gline = 0

state_TPS_json= {'state':[],'TPS':0}

def make_state_and_TPS_data():
    #state = np.random.randrange(1,100,2)
    open(metricFile, 'w')
    with open(metricFile, 'a')as file:
        for i in range(1000):
            state = [round(random.uniform(0, 1), 3)for _ in range(64)]
            content = str(state)
            file.write(content)
            content = '\n'
            file.write(content)

    open(TPSFile, 'w')
    with open(TPSFile, 'a')as file:
        for i in range(1000):
            TPS = [random.randint(6000,13000) for _ in range(1)]
            content = str(TPS)
            file.write(content)
            content = '\n'
            file.write(content)

def get_state_and_TPS_data():
    global gline
    gline = gline + 1
    print("gline ", gline)
    with open(metricFile, 'r') as file:
        metric = linecache.getline(metricFile, gline)
        print("metric ", metric)

    with open(TPSFile, 'r') as file:
        TPS = linecache.getline(TPSFile, gline)
        TPS.strip('\n')
        print("TPS ", TPS)

    return metric,TPS

    TPS = 10000
    temp = np.random.randint(0, 3000)
    if (temp < 1500):
        TPS = TPS - temp
    else:
        TPS = TPS + temp - 1500
    return str(state), str(TPS)




def pack_metric_tps_response():
    state, TPS = get_state_and_TPS_data()
    print("pack_metric_tps_response:state", state)
    print("pack_metric_tps_response:TPS", TPS)

    rep = [('[', ''), (']', ''), ('\n', ''),('\'','\"')]

    for c, r in rep:
        if c in state:
            state = state.replace(c, r)
        if c in TPS:
            TPS = TPS.replace(c, r)
    print("pack_metric_tps_response:state", state)
    print("pack_metric_tps_response:TPS", TPS)

    state_TPS_json["state"] = str(state)
    state_TPS_json["TPS"] = str(TPS)
    print("state_TPS_json", state_TPS_json)
    return state_TPS_json


def handle(client, addr):
    print("from ", addr)
    data = client.recv(1024)

    # 请求报文
    for k, v in enumerate(data.decode().split("\r\n")):
        print(k, v)

    state_TPS_json = pack_metric_tps_response()
    logging.debug("state_TPS_json  " + str(state_TPS_json))

    bodyText = str(state_TPS_json)
    # 响应报文
    # 响应行
    client.send(b"HTTP/1.1 200 OK\r\n")
    # 首部行
    client.send(b"Server: pdudo_web_sites\r\n")
    client.send(b"Content-Type: text/html\r\n")
    client.send(("Content-Length: %s\r\n" % (len(bodyText) + 2)).encode())
    client.send(b"\r\n")
    client.send(("%s\r\n" % (bodyText)).encode())


def main():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("127.0.0.1",8080))
        s.listen()
        print("Serving at port 8080")
        logging.debug("Serving at port 8080")

        make_state_and_TPS_data()


        while True:
            client, addr = s.accept()
            t = threading.Thread(target=handle, args=(client, addr))
            t.start()
    finally:
        s.close()


if __name__ == '__main__':
    main()