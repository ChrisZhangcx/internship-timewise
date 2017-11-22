# -*- coding:utf8 -*-
import MySQLdb


def connect(host='192.168.18.63', port=3306, user='root', passwd='123321', db='goal'):
    global cur, conn
    conn = MySQLdb.connect(
        host=host,
        port=port,
        user=user,
        passwd=passwd,
        db=db
    )
    cur = conn.cursor()
    print "database connected."
    return cur


def disconnect():
    global cur, conn
    conn.commit()
    cur.close()
    conn.close()
    print "database disconnected."


cur = None
conn = None
