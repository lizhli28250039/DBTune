import http.server
import socketserver
import random
import numpy as np
import linecache
import json
import logging
import socket
import threading
import email
import pprint
from io import StringIO

import os

PORT = 8080
metricFile='metric.txt'
TPSFile='tps.txt'

gline = 0

state_TPS_json= {'state':[],'TPS':0}

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


def parse_request(request):
    requeststr = str(request)
    raw_list = requeststr.split("\\r\\n")
    request = {}
    for index in range(1, len(raw_list)):
        item = raw_list[index].split(":")
        if len(item) == 2:
            request.update({item[0].lstrip(' '): item[1].lstrip(' ')})
    return request

def handle(client, addr):
    print("from ", addr)
    data = client.recv(1024)

    logging.debug("handle begin")

    # 请求报文
    for k, v in enumerate(data.decode().split("\r\n")):
        print(k, v)
        logging.debug("request:",k,v)

    request = parse_request(data)
    print(request)
    print('\n')
    print(request.keys())

    # pretty-print the dictionary of headers
    logging.debug("request:",request)
    logging.debug("request keys:", request.keys())
    #logging.debug("request type:", request["{\"type\""])

    if "GET" in str(data):
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
    else:
        print("post request!!!!!")
        bodyText = ""
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

    if not os.path.exists("Logs"):
        os.makedirs("Logs")
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)-8s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S', filename='Logs/dbtune-debug.log', filemode='w')

    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("127.0.0.1",8080))
        s.listen()
        print("Serving at port 8080")
        logging.debug("Serving at port 8080")
        while True:
            client, addr = s.accept()
            t = threading.Thread(target=handle, args=(client, addr))
            t.start()
    finally:
        s.close()


if __name__ == '__main__':
    main()